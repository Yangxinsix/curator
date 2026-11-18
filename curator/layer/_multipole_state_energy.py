"""Learned full nonlinear energy of charge and dipole states."""
import math
import torch
from torch import nn
from ._global_state_blocks import GlobalStateNetwork, invariant_features, prepare_state_geometry


class NeuralMultipoleStateEnergy(nn.Module):
    """U(R,h,q,p), independent of boundary controls.

    State order: [q,px,py,pz], units e and e*Angstrom. This unrestricted neural
    energy promises neither convexity nor coercivity: stable reference minima
    must be checked. No Coulomb term, curvature floor or separate Hessian.
    """
    def __init__(self, hidden_irreps="64x0e + 16x1o + 8x2e", num_layers=2, num_heads=4,
                 radial_features=8, max_length=32.0, readout_scale=0.001,
                 charge_scale=1.0, dipole_scale=1.0):
        super().__init__()
        if not all(math.isfinite(v) and v > 0 for v in (readout_scale, charge_scale, dipole_scale)):
            raise ValueError("readout and state scales must be positive finite values")
        self.network = GlobalStateNetwork(hidden_irreps, num_layers, num_heads, radial_features, max_length)
        self.readout_scale = float(readout_scale)
        self.register_buffer("state_scale", torch.tensor([charge_scale, dipole_scale, dipole_scale, dipole_scale]))
        width = self.network.irreps.num_irreps
        self.readout = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 1, bias=False))
        nn.init.normal_(self.readout[-1].weight, std=self.readout_scale / math.sqrt(64))

    def bind(self, feature_spec, components=4, state_kind="multipole", **unused):
        if components != 4 or state_kind != "multipole":
            raise ValueError("multipole energy requires state_kind='multipole' and components=4")
        self.network.bind(feature_spec)
        return self

    @staticmethod
    def prepare_context(context):
        return prepare_state_geometry(context)

    def initial_state(self, context):
        return context["features"].new_zeros((context["features"].shape[0], 4))

    def forward(self, context, state):
        if state.ndim != 2 or state.shape != (context["features"].shape[0], 4):
            raise ValueError("multipole state must have shape [n_atoms, 4]")
        features = self.network(context, state / self.state_scale)
        return self.readout(invariant_features(features, self.network.irreps)).sum()

    def get_extra_state(self):
        return {"schema": "neural_multipole_energy_v1", "state_order": ("q", "px", "py", "pz"),
                "state_units": ("e", "e*Angstrom", "e*Angstrom", "e*Angstrom"),
                "hidden_irreps": str(self.network.irreps), "num_layers": self.network.num_layers,
                "num_heads": self.network.num_heads, "radial_features": self.network.radial_features,
                "max_length": self.network.max_length, "readout_scale": self.readout_scale,
                "periodicity": "finite_only", "convexity": "not_imposed"}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("multipole energy checkpoint contract differs")


__all__ = ["NeuralMultipoleStateEnergy"]
