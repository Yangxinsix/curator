"""A scalar charge produces a directional field via the gradient of position phases.

For four learned radial channels, k_a(x,y) = sum_m w_ma cos(G_m·(x-y)).
The energy is sum_i,j,a q_j u_ia·grad_x k_a(r_i,r_j). We collect the coefficient
of q_j first. The energy, the exact partial dU/dq and the live proposal drive
then use the very same tensor. Self derivatives of the cosine kernel are zero.
"""
import torch
from torch import nn

from model import RMSChargeEnergy
from tensor_model import TensorEnergy
from curator.layer.efa.geometry import structure_cells


class DirectionalField(nn.Module):
    def __init__(self, spec, reference):
        super().__init__()
        self.scalar_width = spec.scalar_channels
        self.receiver = nn.Parameter(reference.new_zeros((4, spec.vector_channels)))

    def forward(self, features, positions, attention, cell=None, *, cached_waves=False):
        if cell is None:
            # Four non-zero frequencies from the original molecular EFA spectrum.
            frequencies = attention.frequencies
            radius = torch.stack([frequencies[len(frequencies) * i // 4 - 1] for i in range(1, 5)])
            waves = (radius[:, None, None] * attention.grid[None]).flatten(0, 1)
            weights = attention.weights.repeat(4) / 4
            max_radius = frequencies[-1]
        else:
            # Same reciprocal wavevectors as the periodic parent, no 1/G^2 law.
            # Fixed-cell training prepares these in AuMgOGraph before CUDA capture;
            # general evaluation enumerates them from the actual (possibly new) cell.
            if cached_waves:
                waves = attention._training_waves
            else:
                waves, _ = attention._explicit_modes(cell)
                waves = waves[:waves.size(0) // 2]
            max_radius = attention.periodic_cutoff
            volume = (cell[0] * torch.linalg.cross(cell[1], cell[2])).sum().abs()
            taper = (1 - waves.square().sum(-1) / max_radius**2).clamp_min(0).pow(3)
            weights = 2 * attention.periodic_reference_volume / volume * taper

        # Each basis is a polynomial of |G| only (constant across angular directions).
        t = waves.norm(dim=-1) / max_radius
        y = 1 - t
        spectrum = weights[:, None] * torch.stack((y**3, 3*t*y*y, 3*t*t*y, t**3), -1)
        v = features[..., self.scalar_width:].reshape(*features.shape[:2], -1, 3)
        u = self.receiver @ v
        proj = ((u @ waves.T) * spectrum.T).sum(-2)
        phase = (positions - positions.mean(1, keepdim=True)) @ waves.T
        c, s = phase.cos(), phase.sin()
        # For source j: sum_i -u_i·G sin(phase_i-phase_j).
        drive = s * (proj*c).sum(1, keepdim=True) - c * (proj*s).sum(1, keepdim=True)
        return drive.sum(-1, keepdim=True)


class _DirectionalEnergy:
    def _direction_from_context(self, ctx):
        cell, = structure_cells(ctx, 1)
        return self.direction_field(ctx["features"][None], ctx["positions"][None],
                                    self.attention, cell)[0]

    def prepare_proposal_context(self, ctx, proposal):
        ctx = super().prepare_proposal_context(ctx, proposal)
        dc = self._direction_from_context(ctx)
        c, b, value = ctx["_shared_efa"]["live_physics"]
        return {**ctx, "_direction_drive": dc, "_shared_efa": {
            **ctx["_shared_efa"], "live_physics": (c + dc, b, value),
        }}

    def forward(self, ctx, state):
        dc = ctx.get("_direction_drive")
        if dc is None:
            dc = self._direction_from_context(ctx)
        return super().forward(ctx, state) + (state * dc).sum()

    def prepare_dense_batch(self, *args, **kwargs):
        raise NotImplementedError("Directional drive requires coordinates; use panel_energy_charge or the full head")

    def get_extra_state(self):
        return {**super().get_extra_state(), "direction_schema": 1}


class DirectionChargeEnergy(_DirectionalEnergy, RMSChargeEnergy):
    pass


class DirectionTensorEnergy(_DirectionalEnergy, TensorEnergy):
    pass


def attach(model):
    """Install after restoring its original scalar/STF parent; preserve existing params."""
    energy = model.output_modules[0].internal_energy
    classes = {RMSChargeEnergy: DirectionChargeEnergy, TensorEnergy: DirectionTensorEnergy}
    if type(energy) not in classes:
        raise TypeError(f"Unsupported directional parent {type(energy).__name__}")
    cls = classes[type(energy)]
    energy.direction_field = DirectionalField(energy.feature_spec, next(energy.parameters()))
    energy.__class__ = cls
    return list(energy.direction_field.parameters())
