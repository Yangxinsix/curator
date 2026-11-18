"""Strictly quadratic ACVR energies with shared local terms and selectable coupling."""
from __future__ import annotations

import itertools
import math

import torch
from torch import nn
from torch.nn import functional as F

from ._charge_internal_energy import EwaldChargeEnergy


class QuadraticChargeEnergy(EwaldChargeEnergy):
    """U(R,q) = a(R).q + .5 sum_i d_i(R) q_i^2 + U_coupling(R,q).

    ``ewald`` uses Curator's existing periodic electrostatic kernel. ``learned``
    uses .5 ||Phi(R,h)^T q||^2, with learnable invariant amplitudes multiplying
    periodic sine/cosine modes. This is a finite-rank PSD coupling, not an
    unrestricted dense Hessian and not a Coulomb asymptotic prior. Its reciprocal
    integer grid is tied to the supplied lattice basis (not invariant to arbitrary
    unimodular changes of cell basis). It is invariant to atom permutation, rigid
    translations/rotations, and independent lattice translations of each atom.

    Both modes use identical local networks/initialization. Only this scalar
    energy defines state gradients and Hessian-vector products. No boundary
    controls or state enter the geometry networks. Ewald's full curvature must
    be checked: a positive local floor alone does not ensure Ewald stability.
    """
    def __init__(self, scalar_channels=None, hidden_features=32, hardness_floor=20.0,
                 coupling="learned", coupling_channels=8, reciprocal_order=1,
                 coupling_scale=1.0, local_init_scale=0.01, ewald=None):
        if coupling not in {"ewald", "learned"}:
            raise ValueError("coupling must be ewald or learned")
        if coupling_channels < 1 or reciprocal_order < 1:
            raise ValueError("positive coupling width and reciprocal order required")
        if not math.isfinite(coupling_scale) or coupling_scale <= 0:
            raise ValueError("finite positive coupling_scale required")
        if not math.isfinite(local_init_scale) or local_init_scale <= 0:
            raise ValueError("finite positive local_init_scale required")
        self.coupling = coupling
        self.coupling_channels = int(coupling_channels)
        self.reciprocal_order = int(reciprocal_order)
        self.coupling_scale = float(coupling_scale)
        self.local_init_scale = float(local_init_scale)
        super().__init__(None, hidden_features, hardness_floor, ewald)
        modes = [v for v in itertools.product(range(-reciprocal_order, reciprocal_order + 1), repeat=3)
                 if any(v) and next(x for x in v if x) > 0]
        self.register_buffer("integer_modes", torch.tensor(modes, dtype=torch.get_default_dtype()))
        self.amplitude = None
        self.spectrum = None
        if scalar_channels is not None:
            self.bind(scalar_channels)

    def bind(self, scalar_channels, vector_channels=0, components=1, state_kind="scalar", **kwargs):
        already_bound = self.local is not None
        super().bind(scalar_channels, vector_channels, components, state_kind, **kwargs)
        if already_bound:
            return self
        with torch.no_grad():
            self.local[-1].weight.mul_(self.local_init_scale)
            self.local[-1].bias.zero_()
        if self.coupling == "learned":
            self.amplitude = nn.Sequential(nn.Linear(scalar_channels, self.hidden_features), nn.SiLU(),
                                           nn.Linear(self.hidden_features, self.coupling_channels))
            self.spectrum = nn.Sequential(nn.Linear(1, self.hidden_features), nn.SiLU(),
                                          nn.Linear(self.hidden_features, self.coupling_channels))
        return self

    def geometry_context(self, data):
        if self.coupling == "ewald":
            return super().geometry_context(data)
        from curator.data import properties
        return [{} for _ in data[properties.n_atoms]]

    def coupling_features(self, context):
        """Periodic, differentiable geometry-dependent factor Phi [atoms, modes]."""
        if self.amplitude is None:
            raise RuntimeError("learned coupling must be bound before evaluation")
        cell, positions = context["cell"], context["positions"]
        if not bool(context["pbc"].all()):
            raise ValueError("this quadratic comparison requires explicit 3D periodicity")
        if cell.shape != (3, 3):
            raise ValueError("cell must have shape [3, 3]")
        # Rows of reciprocal are wavevectors; solve preserves position/cell AD.
        reciprocal = 2 * math.pi * torch.linalg.solve(cell, self.integer_modes.T).T
        phase = positions @ reciprocal.T
        radial = F.softplus(self.spectrum(reciprocal.square().sum(-1, keepdim=True)))
        amplitudes = torch.tanh(self.amplitude(context["scalar"]))
        weighted = amplitudes[:, None, :] * radial.sqrt()[None, :, :]
        factor = torch.cat((weighted * phase.cos()[:, :, None],
                            weighted * phase.sin()[:, :, None]), dim=1).flatten(1)
        normalizer = math.sqrt(positions.shape[0] * self.integer_modes.shape[0] * self.coupling_channels)
        return factor * (math.sqrt(self.coupling_scale) / normalizer)

    def forward(self, context, state):
        if self.coupling == "ewald":
            return super().forward(context, state)
        if state.ndim != 2 or state.shape[1] != 1:
            raise ValueError("charge state must have shape [n_atoms, 1]")
        coefficients = self.coefficients(context)
        local_energy = (coefficients["electronegativity"] * state
                        + 0.5 * coefficients["hardness"] * state.square()).sum()
        modes = self.coupling_features(context).T @ state[:, 0]
        return local_energy + 0.5 * modes.square().sum()

    def get_extra_state(self):
        result = {"schema": 1, "coupling": self.coupling,
                  "hardness_floor": self.hardness_floor, "hidden_features": self.hidden_features,
                  "coupling_channels": self.coupling_channels, "reciprocal_order": self.reciprocal_order,
                  "coupling_scale": self.coupling_scale, "local_init_scale": self.local_init_scale,
                  "units": ("e", "V", "eV", "Angstrom"), "periodicity": "3D"}
        if self.coupling == "ewald":
            result["ewald"] = super().get_extra_state()
        return result

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("quadratic charge energy contract changed")
