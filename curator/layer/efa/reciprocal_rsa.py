"""Symmetric, volume-normalized RSA for ACVR's equivariant query/key features."""
import math

import torch

from .core import PreparedAttention
from .reciprocal import ReciprocalFastAttention, reciprocal_grid


class RSAPreparedAttention(PreparedAttention):
    """Use the same three tensors as ACVR's existing transpose communication."""

    def apply(self, values, transpose=False):
        query, key = (self.key, self.query) if transpose else (self.query, self.key)
        # Reduce over atoms before the value projection. No atom-pair tensor.
        moments = key.permute(1, 2, 0) @ values
        updates = query.transpose(0, 1) @ moments
        return (self.weights[:, None, None] * updates).sum(dim=0)


class PeriodicRSA(ReciprocalFastAttention):
    """RSA with the q-EFA projections, a smooth physical cutoff and a periodic sum.

    Sigma (Å) follows the paper's real-space tolerance 1e-3 and cutoff 6 Å.
    The dimensionless Fourier weight is (Vref/V) exp(-u/2)/u with u=sigma²G²,
    multiplied by the parent's C² taper. The uniform mode is excluded.
    """

    def __init__(self, *args, rsa_sigma=6 / math.sqrt(-2 * math.log(1e-3)), **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(rsa_sigma) or rsa_sigma <= 0:
            raise ValueError("rsa_sigma must be positive and finite in angstroms")
        self.rsa_sigma = float(rsa_sigma)
        self.periodic_backend = "rsa"

    def _explicit_modes(self, cell):
        waves, taper = reciprocal_grid(cell, self.periodic_cutoff, self.periodic_reference_volume)
        squared = waves.square().sum(dim=-1)
        active = (squared.detach() > 0) & (taper.detach() > 0)
        u = self.rsa_sigma**2 * squared[active]
        return waves[active], taper[active] * torch.exp(-0.5 * u) / u

    def prepare(self, query, key, positions, cell=None):
        if cell is None:
            return super().prepare(query, key, positions)
        waves, weights = self._explicit_modes(cell.to(positions))
        # The lexicographic grid and radial mask give opposite waves in reverse
        # order. For every +/- pair the two antisymmetric rotary terms cancel.
        half = waves.size(0) // 2
        phase = waves[:half] @ (positions - positions.mean(0)).T
        harmonics = torch.cat((phase.cos(), phase.sin()), dim=0).T
        weights = 2 * weights[:half].repeat(2)
        if half == 0:
            # ACVR also consumes the standard three-tensor representation for
            # the transpose. A zero-weight slot keeps empty einsums well-defined.
            harmonics = positions[:, :1] * 0
            weights = weights.sum().reshape(1)
        query = query.flatten(1)
        key = key.flatten(1)
        return RSAPreparedAttention(harmonics[:, :, None] * (query / math.sqrt(query.size(1)))[:, None, :],
                                    harmonics[:, :, None] * key[:, None, :], weights)

    def aggregate(self, prepared, value):
        if isinstance(prepared, RSAPreparedAttention):
            return prepared.apply(value)
        return super().aggregate(prepared, value)
