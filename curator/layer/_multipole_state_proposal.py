"""Boundary-conditioned amortized charge/dipole state proposal."""
import math
import torch
from e3nn import o3
from torch import nn
from ._global_state_blocks import GlobalStateNetwork, STATE_IRREPS, prepare_state_geometry


class MultipoleStateProposal(nn.Module):
    """Predict [q,px,py,pz]; the outer output applies hard constraints.

    origin is feasible. network_field is the conjugate drive for U-field.state,
    ordered [charge potential,Ex,Ey,Ez]. A background potential enters the
    scalar drive with a minus sign. This module contains no physical energy.
    """
    def __init__(self, hidden_irreps="64x0e + 16x1o + 8x2e", num_layers=2, num_heads=4,
                 radial_features=8, max_length=32.0, readout_scale=0.05,
                 charge_scale=1.0, dipole_scale=1.0, potential_scale=1.0, field_scale=1.0):
        super().__init__()
        if not all(math.isfinite(v) and v > 0 for v in
                   (readout_scale, charge_scale, dipole_scale, potential_scale, field_scale)):
            raise ValueError("proposal scales must be positive finite values")
        self.network = GlobalStateNetwork(hidden_irreps, num_layers, num_heads, radial_features, max_length)
        self.readout_scale = float(readout_scale)
        self.register_buffer("state_scale", torch.tensor([charge_scale, dipole_scale, dipole_scale, dipole_scale]))
        self.register_buffer("field_scale", torch.tensor([potential_scale, field_scale, field_scale, field_scale]))
        self.field_input = o3.Linear(STATE_IRREPS, self.network.irreps)
        self.readout = o3.Linear(self.network.irreps, STATE_IRREPS)
        with torch.no_grad():
            self.readout.weight.mul_(self.readout_scale)

    def bind(self, feature_spec, components=4, state_kind="multipole", **unused):
        if components != 4 or state_kind != "multipole":
            raise ValueError("multipole proposal requires state_kind='multipole' and components=4")
        self.network.bind(feature_spec)
        return self

    @staticmethod
    def prepare_context(context):
        return prepare_state_geometry(context)

    def forward(self, context):
        origin, field = context["origin"], context["network_field"]
        if origin.shape != (context["features"].shape[0], 4) or field.shape != origin.shape:
            raise ValueError("proposal origin and conjugate field must have shape [n_atoms, 4]")
        values = self.network(context, origin / self.state_scale,
                              extra=self.field_input(field / self.field_scale))
        return origin + self.state_scale * self.readout(values)

    def checkpoint_contract(self):
        return {"schema": "multipole_state_proposal_v1", "hidden_irreps": str(self.network.irreps),
                "num_layers": self.network.num_layers, "num_heads": self.network.num_heads,
                "radial_features": self.network.radial_features, "max_length": self.network.max_length,
                "readout_scale": self.readout_scale, "state_order": ("q", "px", "py", "pz"),
                "field_convention": "U_minus_field_dot_state", "periodicity": "finite_only"}

    def get_extra_state(self):
        return self.checkpoint_contract()

    def set_extra_state(self, state):
        if state != self.checkpoint_contract():
            raise ValueError("multipole proposal checkpoint contract differs")


__all__ = ["MultipoleStateProposal"]
