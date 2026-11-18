"""Volume-normalized, full-shell reciprocal q-EFA for fully periodic cells."""
import math
from typing import NamedTuple

import torch

from .core import EuclideanFastAttention, PreparedAttention
from .geometry import rotary_encoding


def reciprocal_grid(cell, cutoff, reference_volume):
    """Return Cartesian waves and weights on the centered rectangular integer grid.

    Rows of cell are direct lattice vectors in Å. Outside the sphere weights are
    zero, with two continuous derivatives at the cutoff. The bounds select
    integer topology only; wavevectors and volume retain their cell derivatives.
    """
    if cell.shape != (3, 3) or not cell.is_floating_point():
        raise ValueError("reciprocal q-EFA requires a floating-point 3x3 cell")
    volume = torch.linalg.det(cell).abs()
    if not bool(torch.isfinite(cell).all() and torch.isfinite(volume) and volume > 0):
        raise ValueError("reciprocal q-EFA requires a finite nonsingular cell")
    bounds = torch.ceil(torch.linalg.vector_norm(cell.detach(), dim=1) * (cutoff / (2 * math.pi)))
    axes = [torch.arange(-n, n + 1, dtype=cell.dtype, device=cell.device)
            for n in bounds.long().tolist()]
    indices = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    waves = indices @ (2 * math.pi * torch.linalg.inv(cell).T)
    taper = (1 - waves.square().sum(dim=-1) / cutoff**2).clamp_min(0)
    return waves, (reference_volume / volume) * taper.pow(3)


class NUFFTPreparedAttention(NamedTuple):
    query: torch.Tensor
    key: torch.Tensor
    positions: torch.Tensor
    cell: torch.Tensor
    cutoff: float
    reference_volume: float

    def apply(self, values, transpose=False):
        from .reciprocal_nufft import nufft_aggregate
        query, key = (self.key, self.query) if transpose else (self.query, self.key)
        return nufft_aggregate(query, key, values, self.positions, self.cell,
                               self.cutoff, self.reference_volume)


class ReciprocalFastAttention(EuclideanFastAttention):
    """Keep the original spherical FFF kernel and use reciprocal modes under TTT."""

    def __init__(self, *args, periodic_backend="explicit", periodic_cutoff=0.86,
                 periodic_reference_volume=45.0, **kwargs):
        super().__init__(*args, **kwargs)
        if periodic_backend not in ("explicit", "nufft"):
            raise ValueError("periodic_backend must be explicit or nufft")
        if not all(math.isfinite(x) and x > 0 for x in (periodic_cutoff, periodic_reference_volume)):
            raise ValueError("periodic_cutoff and periodic_reference_volume must be positive and finite")
        self.periodic_backend = periodic_backend
        self.periodic_cutoff = float(periodic_cutoff)
        self.periodic_reference_volume = float(periodic_reference_volume)

    @classmethod
    def from_attention(cls, source, **options):
        core = cls(source.hidden_features, source.num_directions, source.max_frequency,
                   source.max_length, contraction=source.contraction,
                   pairwise_max_atoms=source.pairwise_max_atoms, **options).to(source.frequencies)
        core.load_state_dict(source.state_dict(), strict=True)
        return core.train(source.training)

    def _explicit_modes(self, cell):
        waves, weights = reciprocal_grid(cell, self.periodic_cutoff, self.periodic_reference_volume)
        active = weights.detach() > 0
        return waves[active], weights[active]

    def rotary(self, values, positions, cell=None):
        if cell is None:
            return super().rotary(values, positions)
        waves, _ = self._explicit_modes(cell.to(positions))
        return rotary_encoding(values, positions, waves, values.new_ones(values.shape[-1] // 2))

    def prepare(self, query, key, positions, cell=None):
        if cell is None:
            return super().prepare(query, key, positions)
        cell = cell.to(positions)
        norm = math.sqrt(query.shape[1] * query.shape[2])
        if self.periodic_backend == "nufft":
            return NUFFTPreparedAttention(query.flatten(1) / norm, key.flatten(1), positions,
                                          cell, self.periodic_cutoff, self.periodic_reference_volume)
        waves, weights = self._explicit_modes(cell)
        ones = query.new_ones(query.shape[-1] // 2)
        return PreparedAttention(rotary_encoding(query, positions, waves, ones) / norm,
                                 rotary_encoding(key, positions, waves, ones), weights)

    def aggregate(self, prepared, value):
        if isinstance(prepared, NUFFTPreparedAttention):
            return prepared.apply(value)
        return super().aggregate(prepared, value)
