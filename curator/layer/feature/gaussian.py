"""Bounded Gaussian projection of concatenated readout parameter gradients.

Rows are parameter coordinates, in layer order with each ``a outer b`` stored
a-major. A fixed tile grid defines one iid Gaussian matrix. Layer boundaries,
atom batches, output-width prefixes, and cache limits never change that matrix.
The tile shape is part of mapping version ``readout-gaussian-v1`` and must be
recorded if callers override its defaults; it is not an execution-only option.
"""
from __future__ import annotations

import hashlib
import math

import torch


class BlockGaussianProjection:
    """Project outer-product blocks without storing N x P or P x D arrays.

    At most a parameter tile (N x row_tile), a Gaussian tile
    (row_tile x column_tile), the N x D accumulator, and ``cache_bytes`` of
    reusable Gaussian tiles are retained. Factors themselves belong to the
    caller. Cached weights are derived state and can be discarded at any time.
    """

    version = "readout-gaussian-v1"

    def __init__(
        self,
        num_features: int,
        seed: int = 0,
        row_tile: int = 256,
        column_tile: int = 256,
        cache_bytes: int = 64 * 1024**2,
    ) -> None:
        for name, value in (("num_features", num_features), ("row_tile", row_tile),
                            ("column_tile", column_tile)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(cache_bytes, int) or isinstance(cache_bytes, bool) or cache_bytes < 0:
            raise ValueError("cache_bytes must be a nonnegative integer.")
        self.num_features = num_features
        self.seed = int(seed)
        self.row_tile = row_tile
        self.column_tile = column_tile
        self.cache_bytes = cache_bytes
        self._cache: dict[tuple, torch.Tensor] = {}
        self.cached_bytes = 0

    def _seed(self, domain: str) -> int:
        payload = f"{self.version}:{self.seed}:{self.row_tile}:{self.column_tile}:{domain}"
        return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")

    def clear_cache(self) -> None:
        self._cache.clear()
        self.cached_bytes = 0

    def tile(
        self, row_index: int, column_index: int,
        device: torch.device | str, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a full fixed tile of W; indices count tiles, not coordinates."""
        if row_index < 0 or column_index < 0:
            raise ValueError("Tile indices must be nonnegative.")
        device = torch.device(device)
        key = (row_index, column_index, str(device), dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        generator = torch.Generator(device="cpu").manual_seed(
            self._seed(f"weight:{row_index}:{column_index}")
        )
        # One fixed CPU distribution makes device and precision mere casts of
        # the same map. Always draw a complete tile, including at the boundary.
        weight = torch.randn(self.row_tile, self.column_tile,
                             generator=generator, dtype=torch.float64).to(device=device, dtype=dtype)
        size = weight.numel() * weight.element_size()
        if self.cached_bytes + size <= self.cache_bytes:
            self._cache[key] = weight
            self.cached_bytes += size
        return weight

    def phase(self, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
        blocks = []
        for column_index in range(math.ceil(self.num_features / self.column_tile)):
            generator = torch.Generator(device="cpu").manual_seed(self._seed(f"phase:{column_index}"))
            blocks.append(2 * math.pi * torch.rand(self.column_tile, generator=generator, dtype=torch.float64))
        return torch.cat(blocks)[:self.num_features].to(device=device, dtype=dtype)

    @staticmethod
    def _validate_pairs(pairs) -> None:
        if not pairs:
            raise ValueError("At least one readout gradient block is required.")
        first = pairs[0][0]
        for a, b in pairs:
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0] or a.shape[0] != first.shape[0]:
                raise ValueError("Gradient factors must be aligned two-dimensional atom-by-channel matrices.")
            if a.shape[1] == 0 or b.shape[1] == 0:
                raise ValueError("Readout gradient factors must have nonempty channel axes.")
            if a.device != first.device or b.device != first.device or a.dtype != first.dtype or b.dtype != first.dtype:
                raise ValueError("All gradient factors must have the same device and dtype.")
            if not a.is_floating_point() or not b.is_floating_point():
                raise ValueError("Gradient factors must have floating-point dtype.")

    def _add_block(self, accum: torch.Tensor, block: torch.Tensor, start: int) -> None:
        """A block is bounded by one fixed row-tile boundary."""
        row_index, row_offset = divmod(start, self.row_tile)
        for column_index, column_start in enumerate(range(0, self.num_features, self.column_tile)):
            column_stop = min(column_start + self.column_tile, self.num_features)
            weight = self.tile(row_index, column_index, block.device, block.dtype)
            accum[:, column_start:column_stop].add_(
                block @ weight[row_offset:row_offset + block.shape[1], :column_stop-column_start]
            )

    def project_pairs(self, pairs) -> torch.Tensor:
        """Return concat_l(vec(a_l outer b_l)) @ W, without concatenating it."""
        pairs = list(pairs)
        self._validate_pairs(pairs)
        first = pairs[0][0]
        accum = first.new_zeros((first.shape[0], self.num_features))
        offset = 0
        for a, b in pairs:
            width = a.shape[1] * b.shape[1]
            local_start = 0
            while local_start < width:
                start = offset + local_start
                count = min(width-local_start, self.row_tile-start % self.row_tile)
                coordinates = torch.arange(local_start, local_start+count, device=a.device)
                # Gather only this tile's outer-product coordinates. Never
                # expand an entire layer or concatenate readout gradients.
                block = a[:, coordinates // b.shape[1]] * b[:, coordinates % b.shape[1]]
                self._add_block(accum, block, start)
                local_start += count
            offset += width
        return accum

    def project_dense(self, gradient: torch.Tensor) -> torch.Tensor:
        """Oracle/helper for an already materialized gradient, using the same W."""
        if gradient.ndim != 2 or not gradient.is_floating_point():
            raise ValueError("gradient must be a floating-point N x P matrix.")
        accum = gradient.new_zeros((gradient.shape[0], self.num_features))
        for start in range(0, gradient.shape[1], self.row_tile):
            self._add_block(accum, gradient[:, start:start+self.row_tile], start)
        return accum
