"""Gaussian Nyström features with fixed atomic landmarks.

Dense atomic descriptors and exact readout-gradient outer-product factors use
the same Gaussian kernel.  Factors are contracted directly, without creating
the concatenated parameter gradient.  A map is prepared once and reused for
every atom/structure; fitting it does not optimize model parameters.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn


RawFeatures = torch.Tensor | Sequence[tuple[torch.Tensor, torch.Tensor]]


def _positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _sigma(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("sigma must be positive and finite.")
    return value


def _raw_info(raw: RawFeatures) -> tuple[int, torch.device, tuple]:
    """Validate coordinates and return population, device, and coordinate layout."""
    dense = isinstance(raw, torch.Tensor)
    if dense:
        tensors = [raw]
    else:
        if not isinstance(raw, (tuple, list)) or not raw:
            raise ValueError("Features must be a matrix or a nonempty list of factor pairs.")
        if any(not isinstance(pair, (tuple, list)) or len(pair) != 2 for pair in raw):
            raise ValueError("Each gradient block must contain two factor matrices.")
        tensors = [tensor for pair in raw for tensor in pair]
    if any(not isinstance(tensor, torch.Tensor) or tensor.ndim != 2
           or not tensor.is_floating_point() or tensor.shape[1] == 0 for tensor in tensors):
        raise ValueError("Feature coordinates must be floating-point atom-by-channel matrices.")
    first = tensors[0]
    for tensor in tensors:
        if (tensor.shape[0] != first.shape[0] or tensor.device != first.device
                or tensor.dtype != first.dtype):
            raise ValueError("All feature factors must have the same atom count, device, and dtype.")
        for start in range(0, tensor.shape[0], 256):
            if not torch.isfinite(tensor[start:start + 256]).all():
                raise ValueError("Feature coordinates must be finite.")
    layout = ("dense", first.shape[1]) if dense else (
        "factors", tuple((a.shape[1], b.shape[1]) for a, b in raw)
    )
    return first.shape[0], first.device, layout


def _slice(raw: RawFeatures, start: int, stop: int) -> RawFeatures:
    if isinstance(raw, torch.Tensor):
        return raw[start:stop].to(dtype=torch.float64)
    return [(a[start:stop].to(dtype=torch.float64), b[start:stop].to(dtype=torch.float64))
            for a, b in raw]


def _norms(raw: RawFeatures) -> torch.Tensor:
    if isinstance(raw, torch.Tensor):
        return raw.square().sum(dim=1)
    return sum(a.square().sum(dim=1) * b.square().sum(dim=1) for a, b in raw)


def _kernel_block(x: RawFeatures, y: RawFeatures, sigma: float) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        inner = x @ y.T
    else:
        inner = sum((a @ c.T) * (b @ d.T) for (a, b), (c, d) in zip(x, y))
    norm_sum = _norms(x)[:, None] + _norms(y)[None, :]
    distance = norm_sum - 2 * inner
    if not torch.isfinite(distance).all():
        raise ValueError("Squared distances overflowed float64; feature magnitudes are too large.")
    if (distance < -1e-12 * norm_sum).any():
        raise ValueError("Squared distances are materially negative; exact kernel contraction failed.")
    distance.clamp_min_(0)
    # Dividing twice also handles positive sigma whose square underflows.
    return torch.exp_(distance.div_(sigma).div_(sigma).mul_(-0.5))


def _compatible(x: RawFeatures, y: RawFeatures) -> tuple[int, int, torch.device]:
    nx, device, layout = _raw_info(x)
    ny, other_device, other_layout = _raw_info(y)
    if layout != other_layout:
        raise ValueError("Features and landmarks must have the same dense width or factor layer dimensions.")
    if device != other_device:
        raise ValueError("Features and landmarks must be on the same device; move the map with .to(device).")
    return nx, ny, device


def gaussian_cross_kernel(
    x: RawFeatures, y: RawFeatures, sigma: float, block_size: int = 256,
) -> torch.Tensor:
    """Exact rectangular Gaussian kernel, in float64, with bounded temporary blocks.

    The returned matrix is necessarily ``len(x) x len(y)``.  Both temporary
    atom axes are bounded by ``block_size``; factor inputs are never expanded
    to their full parameter coordinates.  Dense and factor representations
    cannot be mixed within a call.
    """
    sigma = _sigma(sigma)
    _positive_integer(block_size, "block_size")
    nx, ny, device = _compatible(x, y)
    result = torch.empty((nx, ny), dtype=torch.float64, device=device)
    for start in range(0, nx, block_size):
        xb = _slice(x, start, start + block_size)
        for landmark_start in range(0, ny, block_size):
            yb = _slice(y, landmark_start, landmark_start + block_size)
            result[start:start + block_size, landmark_start:landmark_start + block_size] = (
                _kernel_block(xb, yb, sigma)
            )
    return result


class NystromMap(nn.Module):
    """Frozen Gaussian map ``k(x, anchors) U Lambda**(-1/2)``.

    Use :meth:`fit` or :meth:`load` to construct a map.  ``num_features`` is
    the requested output width (at most the number of landmarks).  If the
    eigenvalue cutoff removes directions, those columns are padded with zero
    and ``effective_rank`` records the number retained.  No ridge or empirical
    normalization is applied.  Buffers and outputs use float64 by default.
    """

    version = "gaussian-nystrom-v1"

    def __init__(
        self, anchors: RawFeatures, correction: torch.Tensor, eigenvalues: torch.Tensor,
        sigma: float, block_size: int, eigenvalue_rtol: float, eigenvalue_cutoff: float,
        metadata: dict[str, Any] | None,
    ) -> None:
        super().__init__()
        count, device, layout = _raw_info(anchors)
        if count == 0:
            raise ValueError("At least one landmark is required.")
        self.sigma = _sigma(sigma)
        _positive_integer(block_size, "block_size")
        self.block_size = block_size
        self.eigenvalue_rtol = float(eigenvalue_rtol)
        self.eigenvalue_cutoff = float(eigenvalue_cutoff)
        if (not math.isfinite(self.eigenvalue_rtol) or not 0 <= self.eigenvalue_rtol < 1
                or not math.isfinite(self.eigenvalue_cutoff) or self.eigenvalue_cutoff < 0):
            raise ValueError("Invalid eigenvalue cutoff settings.")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON-compatible dictionary.")
        try:
            self._metadata_json = json.dumps(metadata or {}, sort_keys=True,
                                             separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be a JSON-compatible dictionary with finite numbers.") from exc
        self._kind = layout[0]
        self._num_layers = 0 if self._kind == "dense" else len(anchors)
        if self._kind == "dense":
            self.register_buffer("landmarks", anchors.detach().to(dtype=torch.float64).clone())
        else:
            for index, (a, b) in enumerate(anchors):
                self.register_buffer(f"landmark_a_{index}", a.detach().to(dtype=torch.float64).clone())
                self.register_buffer(f"landmark_b_{index}", b.detach().to(dtype=torch.float64).clone())
        if (correction.ndim != 2 or correction.shape[0] != count
                or not 1 <= correction.shape[1] <= count or eigenvalues.ndim != 1
                or not 1 <= eigenvalues.numel() <= correction.shape[1]
                or not torch.isfinite(correction).all() or not torch.isfinite(eigenvalues).all()
                or not (eigenvalues > self.eigenvalue_cutoff).all()):
            raise ValueError("Invalid Nyström correction or retained eigenvalues.")
        if correction[:, eigenvalues.numel():].count_nonzero():
            raise ValueError("Inactive Nyström feature columns must be zero padded.")
        self.register_buffer("correction", correction.detach().to(device=device, dtype=torch.float64).clone())
        self.register_buffer("eigenvalues", eigenvalues.detach().to(device=device, dtype=torch.float64).clone())

    @property
    def num_features(self) -> int:
        return self.correction.shape[1]

    @property
    def effective_rank(self) -> int:
        return self.eigenvalues.numel()

    @property
    def num_landmarks(self) -> int:
        return self.correction.shape[0]

    @property
    def metadata(self) -> dict[str, Any]:
        """A copy of preparation provenance, so callers cannot mutate it in place."""
        return json.loads(self._metadata_json)

    def _anchors(self) -> RawFeatures:
        if self._kind == "dense":
            return self.landmarks
        return [(getattr(self, f"landmark_a_{i}"), getattr(self, f"landmark_b_{i}"))
                for i in range(self._num_layers)]

    @classmethod
    @torch.no_grad()
    def fit(
        cls, anchors: RawFeatures, sigma: float, num_features: int | None = None,
        eigenvalue_rtol: float = 1e-10, block_size: int = 256,
        metadata: dict[str, Any] | None = None,
    ) -> "NystromMap":
        """Prepare a fixed map from caller-selected, globally shared landmarks.

        ``eigenvalue_rtol`` is relative to the largest eigenvalue.  A machine
        precision floor ``m * eps(float64)`` also removes numerically null
        directions.  This is spectral truncation, not diagonal regularization.
        """
        count, _, _ = _raw_info(anchors)
        if count == 0:
            raise ValueError("At least one landmark is required.")
        num_features = count if num_features is None else num_features
        _positive_integer(num_features, "num_features")
        if num_features > count:
            raise ValueError("num_features must not exceed num_landmarks.")
        eigenvalue_rtol = float(eigenvalue_rtol)
        if not math.isfinite(eigenvalue_rtol) or not 0 <= eigenvalue_rtol < 1:
            raise ValueError("eigenvalue_rtol must be finite and in [0, 1).")
        gram = gaussian_cross_kernel(anchors, anchors, sigma, block_size)
        eigenvalues, vectors = torch.linalg.eigh((gram + gram.T) * 0.5)
        eigenvalues, vectors = eigenvalues.flip(0), vectors.flip(1)
        negative_tolerance = float(eigenvalues[0]) * max(
            1e-12, 16 * count * torch.finfo(torch.float64).eps
        )
        if float(eigenvalues[-1]) < -negative_tolerance:
            raise ValueError("Landmark Gaussian kernel is materially non-positive-semidefinite.")
        cutoff = float(eigenvalues[0]) * max(eigenvalue_rtol, count * torch.finfo(torch.float64).eps)
        rank = min(num_features, int((eigenvalues > cutoff).sum()))
        if rank == 0:
            raise ValueError("No positive landmark-kernel directions survive the eigenvalue cutoff.")
        eigenvalues, vectors = eigenvalues[:rank], vectors[:, :rank]
        # Fix the arbitrary sign for reproducible serialized coordinates.
        pivot = vectors.abs().argmax(dim=0)
        signs = vectors[pivot, torch.arange(rank, device=vectors.device)].sign()
        correction = gram.new_zeros((count, num_features))
        correction[:, :rank] = vectors * signs / eigenvalues.sqrt()
        return cls(anchors, correction, eigenvalues, sigma, block_size,
                   eigenvalue_rtol, cutoff, metadata)

    @torch.no_grad()
    def transform(self, raw: RawFeatures) -> torch.Tensor:
        """Return atomic features without materializing the full atom-landmark matrix."""
        anchors = self._anchors()
        count, _, device = _compatible(raw, anchors)
        result = torch.zeros((count, self.num_features), device=device, dtype=torch.float64)
        for start in range(0, count, self.block_size):
            xb = _slice(raw, start, start + self.block_size)
            for landmark_start in range(0, self.num_landmarks, self.block_size):
                stop = landmark_start + self.block_size
                kernel = _kernel_block(xb, _slice(anchors, landmark_start, stop), self.sigma)
                result[start:start + self.block_size].add_(
                    kernel @ self.correction[landmark_start:stop].to(dtype=torch.float64)
                )
        return result

    @torch.no_grad()
    def structure_features(
        self, raw: RawFeatures, image_idx: torch.Tensor, pooling: str = "mean",
    ) -> torch.Tensor:
        """Pool atomic kernel columns, then apply the Nyström linear correction.

        Only atom/landmark blocks and a structure-by-landmark accumulator are
        allocated; no full atom-by-landmark or full gradient matrix is built.
        Missing structure indices have zero features, matching scatter pooling.
        """
        if pooling not in ("mean", "sum"):
            raise ValueError("pooling must be 'mean' or 'sum'.")
        anchors = self._anchors()
        count, _, device = _compatible(raw, anchors)
        if (not isinstance(image_idx, torch.Tensor) or image_idx.ndim != 1
                or image_idx.numel() != count or image_idx.dtype not in
                (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
            raise ValueError("image_idx must have one integer structure index per atom.")
        if (image_idx < 0).any():
            raise ValueError("image_idx must contain nonnegative structure indices.")
        image_idx = image_idx.to(device=device, dtype=torch.long)
        num_structures = int(image_idx.max()) + 1 if count else 0
        pooled = torch.zeros((num_structures, self.num_landmarks), device=device, dtype=torch.float64)
        for start in range(0, count, self.block_size):
            xb = _slice(raw, start, start + self.block_size)
            indices = image_idx[start:start + self.block_size]
            for landmark_start in range(0, self.num_landmarks, self.block_size):
                stop = landmark_start + self.block_size
                kernel = _kernel_block(xb, _slice(anchors, landmark_start, stop), self.sigma)
                pooled[:, landmark_start:stop].index_add_(0, indices, kernel)
        if pooling == "mean" and count:
            counts = torch.bincount(image_idx, minlength=num_structures).clamp_min_(1)
            pooled.div_(counts[:, None])
        return pooled @ self.correction.to(dtype=torch.float64)

    @property
    def fingerprint(self) -> str:
        """Content identity of the mapping and provenance, independent of device."""
        digest = hashlib.sha256()
        settings = {"version": self.version, "sigma": self.sigma,
                    "eigenvalue_rtol": self.eigenvalue_rtol,
                    "eigenvalue_cutoff": self.eigenvalue_cutoff,
                    "kind": self._kind, "metadata": self.metadata}
        digest.update(json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        for name, tensor in sorted(self.state_dict().items()):
            value = tensor.detach().cpu().to(dtype=torch.float64).contiguous()
            digest.update(name.encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path) -> None:
        """Save tensors and primitive metadata; no model objects or executable pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"version": self.version, "sigma": self.sigma,
                    "block_size": self.block_size, "eigenvalue_rtol": self.eigenvalue_rtol,
                    "eigenvalue_cutoff": self.eigenvalue_cutoff, "kind": self._kind,
                    "num_layers": self._num_layers, "metadata_json": self._metadata_json,
                    "state": {key: value.detach().cpu() for key, value in self.state_dict().items()},
                    "fingerprint": self.fingerprint}, path)

    @classmethod
    def load(cls, path: str | Path) -> "NystromMap":
        """Load a saved map onto CPU using PyTorch's restricted tensor loader."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("version") != cls.version:
            raise ValueError("Unsupported Nyström artifact version.")
        try:
            state = payload["state"]
            if payload["kind"] == "dense":
                anchors = state["landmarks"]
            elif payload["kind"] == "factors":
                anchors = [(state[f"landmark_a_{i}"], state[f"landmark_b_{i}"])
                           for i in range(payload["num_layers"])]
            else:
                raise ValueError("Unsupported Nyström landmark representation.")
            result = cls(anchors, state["correction"], state["eigenvalues"], payload["sigma"],
                         payload["block_size"], payload["eigenvalue_rtol"],
                         payload["eigenvalue_cutoff"], json.loads(payload["metadata_json"]))
            if set(state) != set(result.state_dict()):
                raise ValueError("Nyström artifact contains inconsistent buffers.")
            if result.fingerprint != payload["fingerprint"]:
                raise ValueError("Nyström artifact fingerprint does not match its contents.")
            return result
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("Incomplete or invalid Nyström artifact.") from exc
