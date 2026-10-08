"""On-demand exact Gaussian kernel mean embeddings for structure selection.

Atomic descriptors are pooled *after* evaluating the Gaussian kernel.  This is
not an RBF applied to a structure's mean descriptor.  Readout gradients can be
passed as their corrected outer-product factors without materialising an
atom-by-parameter gradient matrix.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import OrderedDict, defaultdict
import math
import operator
from typing import Sequence

import torch

from .kernel import KernelMatrix


RawFeatures = torch.Tensor | Sequence[tuple[torch.Tensor, torch.Tensor]]


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}.") from exc
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


class ExactGaussianKMEKernelMatrix(KernelMatrix):
    """Gaussian atomic kernel, averaged over every structure's atom pairs.

    ``raw`` is ``[sum(counts), channels]`` or a nonempty sequence of ``(a, b)``
    matrices with that many rows. Each pair represents one corrected gradient
    block ``vec(a_i b_i.T)``; distinct blocks are concatenated in parameter
    space, so their inner products are summed before the Gaussian exponential.
    Atoms must be contiguous by structure in the order specified by ``counts``.

    Raw coordinates retain their input dtype; contractions, norms and reductions
    use float64. ``device`` moves raw inputs once, without expanding gradient
    factors. Inputs are detached, but not copied when no move is necessary;
    callers must not mutate them during selection. No standardisation, diagonal
    normalisation, approximate features, or full atomic Gram matrix is used.

    ``block_size`` bounds both atomic axes of each temporary cross-kernel block.
    Diagonal work is additionally batched by atom count. A bounded LRU cache can
    retain ``cache_columns`` structure-kernel columns (default 1), allowing a
    second selector to reuse columns when a larger cache is deliberately chosen.
    Cache storage is O(N * cache_columns), separate from atomic block work.
    """

    @torch.no_grad()
    def __init__(
        self,
        raw: RawFeatures,
        counts: Sequence[int] | torch.Tensor,
        sigma: float,
        block_size: int = 2048,
        device: torch.device | str | None = None,
        cache_columns: int = 1,
    ) -> None:
        self.block_size = _integer(block_size, "block_size")
        self.cache_columns = _integer(cache_columns, "cache_columns", minimum=0)
        try:
            self.sigma = float(sigma)
        except (TypeError, ValueError) as exc:
            raise ValueError("sigma must be finite and positive.") from exc
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be finite and positive.")
        if isinstance(counts, torch.Tensor):
            if counts.ndim != 1 or counts.dtype not in (
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
            ):
                raise ValueError("counts must be a one-dimensional integer sequence.")
            counts = counts.cpu().tolist()
        try:
            self.counts = tuple(_integer(n, "each atom count") for n in counts)
        except TypeError as exc:
            raise ValueError("counts must be a one-dimensional integer sequence.") from exc
        if not self.counts:
            raise ValueError("At least one nonempty structure is required.")
        self.num_atoms = sum(self.counts)
        super().__init__(len(self.counts))

        self._dense = isinstance(raw, torch.Tensor)
        if self._dense:
            tensors = [raw]
        else:
            if not isinstance(raw, (list, tuple)) or not raw:
                raise ValueError("raw must be a dense matrix or a nonempty list of factor pairs.")
            if any(not isinstance(pair, (list, tuple)) or len(pair) != 2 for pair in raw):
                raise ValueError("Each gradient block must contain two factor matrices.")
            tensors = [tensor for pair in raw for tensor in pair]
        if any(not isinstance(t, torch.Tensor) or t.ndim != 2
               or not t.is_floating_point() or t.shape[1] < 1 for t in tensors):
            raise ValueError("Coordinates must be floating-point atom-by-channel matrices.")
        first = tensors[0]
        if any(t.shape[0] != self.num_atoms for t in tensors):
            raise ValueError("sum(counts) must equal the atom count of every coordinate matrix.")
        if any(t.device != first.device or t.dtype != first.dtype for t in tensors):
            raise ValueError("All factor matrices must have the same device and dtype.")
        self.device = torch.device(device) if device is not None else first.device
        tensors = [t.detach().to(device=self.device) for t in tensors]
        self.raw = tensors[0] if self._dense else list(zip(tensors[::2], tensors[1::2]))
        self.offsets = [0]
        for count in self.counts:
            self.offsets.append(self.offsets[-1] + count)
        self._counts = torch.tensor(self.counts, dtype=torch.float64, device=self.device)
        self._blocks = []
        for start in range(0, self.num_atoms, self.block_size):
            stop = min(start + self.block_size, self.num_atoms)
            first = bisect_right(self.offsets, start) - 1
            last = bisect_left(self.offsets, stop)
            lengths = [min(stop, self.offsets[i + 1]) - max(start, self.offsets[i])
                       for i in range(first, last)]
            self._blocks.append((start, stop, first, last,
                                 torch.tensor(lengths, device=self.device)))
        self._norms = torch.empty(self.num_atoms, device=self.device, dtype=torch.float64)
        invalid = torch.zeros((), dtype=torch.bool, device=self.device)
        for start in range(0, self.num_atoms, self.block_size):
            stop = min(start + self.block_size, self.num_atoms)
            block = self._take(slice(start, stop))
            coordinates = [block] if self._dense else [t for pair in block for t in pair]
            for tensor in coordinates:
                invalid.logical_or_(~torch.isfinite(tensor).all())
            self._norms[start:stop] = self._squared_norm(block)
        if bool(invalid) or not bool(torch.isfinite(self._norms).all()):
            raise ValueError("Coordinates and their squared norms must be finite in float64.")
        self._diag: torch.Tensor | None = None
        self._columns: OrderedDict[int, torch.Tensor] = OrderedDict()

    def _take(self, indices: slice | torch.Tensor) -> RawFeatures:
        if self._dense:
            return self.raw[indices].to(dtype=torch.float64)
        return [(a[indices].to(dtype=torch.float64), b[indices].to(dtype=torch.float64))
                for a, b in self.raw]

    def _squared_norm(self, raw: RawFeatures) -> torch.Tensor:
        if self._dense:
            return raw.square().sum(dim=-1)
        return sum(a.square().sum(dim=-1) * b.square().sum(dim=-1) for a, b in raw)

    def _inner(self, x: RawFeatures, y: RawFeatures) -> torch.Tensor:
        if self._dense:
            return x @ y.transpose(-1, -2)
        return sum((a @ c.transpose(-1, -2)) * (b @ d.transpose(-1, -2))
                   for (a, b), (c, d) in zip(x, y))

    def _gaussian(
        self, inner: torch.Tensor, norm_sum: torch.Tensor, invalid: torch.Tensor,
    ) -> torch.Tensor:
        distance = norm_sum - 2 * inner
        # Defer device-to-host validation to one check per requested column/diag.
        invalid.logical_or_(~torch.isfinite(distance).all())
        invalid.logical_or_((distance < -1e-12 * norm_sum).any())
        return distance.clamp_min_(0).div_(self.sigma).div_(self.sigma).mul_(-0.5).exp_()

    @torch.no_grad()
    def get_diag(self) -> torch.Tensor:
        if self._diag is not None:
            return self._diag
        diag = torch.empty(self.num_columns, dtype=torch.float64, device=self.device)
        invalid = torch.zeros((), dtype=torch.bool, device=self.device)
        groups: dict[int, list[int]] = defaultdict(list)
        for index, count in enumerate(self.counts):
            groups[count].append(index)
        for count, structures in groups.items():
            batch_size = max(1, self.block_size // count)
            for start in range(0, len(structures), batch_size):
                ids = structures[start:start + batch_size]
                base = torch.tensor([self.offsets[i] for i in ids], device=self.device)
                totals = torch.zeros(len(ids), dtype=torch.float64, device=self.device)
                for x_start in range(0, count, self.block_size):
                    x_indices = base[:, None] + torch.arange(
                        x_start, min(x_start + self.block_size, count), device=self.device,
                    )
                    x = self._take(x_indices)
                    for y_start in range(0, count, self.block_size):
                        y_indices = base[:, None] + torch.arange(
                            y_start, min(y_start + self.block_size, count), device=self.device,
                        )
                        y = x if x_start == y_start else self._take(y_indices)
                        norm_sum = self._norms[x_indices][..., None] + self._norms[y_indices][:, None, :]
                        kernel = self._gaussian(self._inner(x, y), norm_sum, invalid)
                        if x_start == y_start:
                            kernel.diagonal(dim1=-2, dim2=-1).fill_(1.0)
                        totals.add_(kernel.sum(dim=(-2, -1)))
                diag[torch.tensor(ids, device=self.device)] = totals / (count * count)
        if bool(invalid):
            raise ValueError("Exact atomic squared distances overflowed or are materially negative.")
        self._diag = diag
        return diag

    def _index(self, index: int) -> int:
        index = _integer(index, "column index", minimum=0)
        if index >= self.num_columns:
            raise IndexError("Kernel column index is outside the structure pool.")
        return index

    @torch.no_grad()
    def get_column(self, i: int) -> torch.Tensor:
        i = self._index(i)
        if i in self._columns:
            self._columns.move_to_end(i)
            return self._columns[i]
        invalid = torch.zeros((), dtype=torch.bool, device=self.device)
        column = torch.zeros(self.num_columns, dtype=torch.float64, device=self.device)
        target_start, target_stop = self.offsets[i:i + 2]
        # Outer loop over targets reuses each converted target block for all atoms.
        for ys in range(target_start, target_stop, self.block_size):
            ye = min(ys + self.block_size, target_stop)
            y = self._take(slice(ys, ye))
            yn = self._norms[ys:ye]
            for xs, xe, first, last, lengths in self._blocks:
                x = self._take(slice(xs, xe))
                kernel = self._gaussian(
                    self._inner(x, y), self._norms[xs:xe, None] + yn[None, :], invalid,
                )
                overlap_start, overlap_stop = max(xs, ys), min(xe, ye)
                if overlap_start < overlap_stop:
                    positions = torch.arange(overlap_start, overlap_stop, device=self.device)
                    kernel[positions - xs, positions - ys] = 1.0
                # Contiguous segmented reduction avoids CUDA atomic scatter-add
                # ordering; structures crossing a block boundary are accumulated.
                column[first:last].add_(torch.segment_reduce(
                    kernel.sum(dim=1), "sum", lengths=lengths,
                ))
        if bool(invalid):
            raise ValueError("Exact atomic squared distances overflowed or are materially negative.")
        column.div_(self._counts).div_(self.counts[i])
        # The dedicated batched diagonal follows the same definition; reuse it to
        # make self-distances exactly zero despite different BLAS reduction order.
        column[i] = self.get_diag()[i]
        if self.cache_columns:
            self._columns[i] = column
            if len(self._columns) > self.cache_columns:
                self._columns.popitem(last=False)
        return column

    @torch.no_grad()
    def get_sq_dists(self, i: int) -> torch.Tensor:
        i = self._index(i)
        diag = self.get_diag()
        norm_sum = diag[i] + diag
        distances = norm_sum - 2 * self.get_column(i)
        if not bool(torch.isfinite(distances).all()) or bool((distances < -1e-12 * norm_sum).any()):
            raise ValueError("Exact KME structure squared distances are materially negative or nonfinite.")
        return distances.clamp_min_(0)

    def clear_column_cache(self) -> None:
        """Release cached structure columns while retaining the diagonal."""
        self._columns.clear()
