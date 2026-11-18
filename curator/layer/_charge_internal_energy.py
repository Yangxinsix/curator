"""Geometry-only local charge coefficients plus the existing Ewald energy."""

from __future__ import annotations

from torch import nn

from ._ewald import EwaldSummation


class EwaldChargeEnergy(nn.Module):
    """Intrinsic charge energy in eV for charges in e and distances in Angstrom.

    Boundary controls belong to the boundary compiler. Neither electronegativity
    nor hardness reads external fields, constraint targets or feasible origins.
    Positive local hardness alone does not guarantee a positive full Hessian;
    choose its floor using the geometry family and check reduced curvature.
    """

    def __init__(self, scalar_channels=None, hidden_features=32, hardness_floor=20.0,
                 ewald=None):
        super().__init__()
        if hidden_features < 1 or hardness_floor <= 0:
            raise ValueError("hidden_features and hardness_floor must be positive")
        self.hidden_features = int(hidden_features)
        self.hardness_floor = float(hardness_floor)
        self.ewald = EwaldSummation() if ewald is None else ewald
        self.scalar_channels = None
        self.local = None
        if scalar_channels is not None:
            self.bind(scalar_channels)

    def bind(self, scalar_channels, vector_channels=0, components=1, state_kind="scalar",
             **kwargs):
        if state_kind != "scalar" or components != 1:
            raise ValueError("EwaldChargeEnergy supports scalar charge with one component only")
        if scalar_channels < 1:
            raise ValueError("EwaldChargeEnergy needs scalar geometric features")
        if self.local is not None:
            if scalar_channels != self.scalar_channels:
                raise ValueError("EwaldChargeEnergy is already bound to different feature dimensions")
            return self
        self.scalar_channels = int(scalar_channels)
        self.local = nn.Sequential(
            nn.Linear(self.scalar_channels, self.hidden_features), nn.SiLU(),
            nn.Linear(self.hidden_features, 2),
        )
        return self

    def geometry_context(self, data):
        """Build once per forward, then reuse for every condition and block."""
        from curator.variational.backends.ewald import EwaldChargeOperator

        return [{"ewald_kernel": kernel}
                for kernel in EwaldChargeOperator(self.ewald).geometry_kernels(data)]

    def coefficients(self, context):
        if self.local is None:
            raise RuntimeError("bind EwaldChargeEnergy to geometric features before use")
        coefficients = self.local(context["scalar"])
        return {
            "electronegativity": coefficients[:, :1],
            "hardness": nn.functional.softplus(coefficients[:, 1:]) + self.hardness_floor,
        }

    def forward(self, context, state):
        if state.ndim != 2 or state.shape[1] != 1:
            raise ValueError("Ewald charge state must have shape [n_atoms, 1]")
        # Always recompute through this module: functional_call then correctly
        # detaches physical weights in the residual branch. Context coefficient
        # tensors are proposal inputs only, never a shortcut around that route.
        coefficients = self.coefficients(context)
        charge = state[:, 0]
        kernel = context["ewald_kernel"]
        if kernel.shape != (charge.numel(), charge.numel()):
            raise ValueError("Ewald kernel shape must match the scalar charge state")
        return ((coefficients["electronegativity"] * state
                 + 0.5 * coefficients["hardness"] * state.square()).sum()
                + 0.5 * charge @ kernel @ charge)

    def get_extra_state(self):
        return {"schema": 1, "hardness_floor": self.hardness_floor,
                "units": ("e", "V", "eV", "Angstrom"),
                "periodicity": "3D_tinfoil", "ewald_alpha": self.ewald.alpha,
                "ewald_cutoff": self.ewald.cutoff, "ewald_k_cutoff": self.ewald.k_cutoff,
                "neutralizing_background": self.ewald.neutralizing_background}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Ewald charge energy units, boundary convention or parameters changed")
