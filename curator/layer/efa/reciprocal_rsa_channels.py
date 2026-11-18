"""Per-channel radial spectra for periodic RSA, shared over angular components."""
import math

import torch
from torch import nn

from .reciprocal_rsa import PeriodicRSA, RSAPreparedAttention


class ChannelSpectrumRSA(PeriodicRSA):
    @classmethod
    def from_rsa(cls, source, groups):
        if type(groups) is not int or groups < 1 or source.hidden_features % (2 * groups):
            raise ValueError("RSA groups must partition whole channel pairs equally")
        core = cls.from_attention(source, periodic_cutoff=source.periodic_cutoff,
                                  periodic_reference_volume=source.periodic_reference_volume,
                                  rsa_sigma=source.rsa_sigma)
        core.band_coefficients = nn.Parameter(core.frequencies.new_zeros(groups, 4))
        return core

    def log_gains(self, radius):
        """A cubic Bernstein profile on [0, cutoff] for each group of channel pairs."""
        x = radius / self.periodic_cutoff
        y = 1 - x
        basis = torch.stack((y**3, 3*x*y**2, 3*x**2*y, x**3), dim=-1)
        return basis @ self.band_coefficients.T

    def prepare(self, query, key, positions, cell=None):
        if cell is None:
            return super().prepare(query, key, positions)
        waves, weights = self._explicit_modes(cell.to(positions))
        half = waves.size(0) // 2
        if not half:
            return super().prepare(query, key, positions, cell)
        waves = waves[:half]
        phase = waves @ (positions - positions.mean(0)).T
        harmonics = torch.cat((phase.cos(), phase.sin()), dim=0).T
        weights = 2 * weights[:half].repeat(2)
        gain = self.log_gains(waves.norm(dim=1)).exp().repeat(2, 1)
        gain = gain.repeat_interleave(self.hidden_features // gain.size(1), dim=1)
        scaled_query = query / math.sqrt(query.size(1) * query.size(2))
        # Apply each gain once per bilinear product, equally to +/- waves and
        # all angular components. Swapping prepared Q and K gives the adjoint.
        encoded_query = harmonics[:, :, None, None] * scaled_query[:, None] * gain[None, :, None]
        encoded_key = harmonics[:, :, None] * key.flatten(1)[:, None]
        return RSAPreparedAttention(encoded_query.flatten(2), encoded_key, weights)
