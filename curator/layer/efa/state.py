"""Charge-conditioned equivariant EFA states and ACVR-compatible energy.

Charges are even scalars. Scalar gates let charge modulate the amplitudes of
existing directional geometry/dipole features without treating charge itself
as a vector. The scalar-only encoder is the controlled injection baseline.
"""

import math
from abc import ABC, abstractmethod

import torch
from e3nn import o3
from torch import nn

from .blocks import EquivariantEFA
from .readout import InvariantEnergyReadout, invariant_features


class ChargeStateEncoder(nn.Module, ABC):
    """Shared geometry/dipole basis and smooth charge-conditioned scalars.

    state is [q] or [q, px, py, pz], according to include_dipoles. Inputs are
    already nondimensionalized by the owner. The equivariant basis contains no
    charge: the subclasses control exactly how q reaches non-scalar channels.
    """

    def __init__(self, input_irreps, output_irreps, hidden_features=64, include_dipoles=True):
        super().__init__()
        if hidden_features < 1:
            raise ValueError("encoder hidden_features must be positive")
        self.input_irreps = o3.Irreps(input_irreps)
        self.output_irreps = o3.Irreps(output_irreps)
        self.include_dipoles = bool(include_dipoles)
        self.components = 4 if self.include_dipoles else 1
        self.scalar_width = sum(mul for mul, ir in self.output_irreps if ir == o3.Irrep("0e"))
        self.tensor_width = self.output_irreps.num_irreps - self.scalar_width
        if not self.scalar_width:
            raise ValueError("state encoder requires even scalar output channels")
        source_irreps = self.input_irreps + (o3.Irreps("1x1o") if self.include_dipoles else o3.Irreps())
        self.geometry_dipole = o3.Linear(source_irreps, self.output_irreps, biases=True)
        self.conditioner = nn.Sequential(
            nn.Linear(self.output_irreps.num_irreps + 1, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
        )
        self.scalars = nn.Linear(hidden_features, self.scalar_width)

    @abstractmethod
    def tensor_gates(self, conditioning):
        """Return one scalar gate per non-0e channel, or None for the baseline."""

    def forward(self, geometry_features, state):
        if geometry_features.ndim != 2 or geometry_features.shape[-1] != self.input_irreps.dim:
            raise ValueError("geometry features do not match encoder input irreps")
        if state.shape != (geometry_features.shape[0], self.components):
            raise ValueError(f"state must have shape [n_atoms, {self.components}]")
        source = torch.cat((geometry_features, state[:, 1:]), dim=-1) if self.include_dipoles else geometry_features
        basis = self.geometry_dipole(source)
        conditioning = self.conditioner(torch.cat(
            (invariant_features(basis, self.output_irreps), state[:, :1]), dim=-1))
        scalars = self.scalars(conditioning)
        gates = self.tensor_gates(conditioning)
        output, scalar_offset, tensor_offset = [], 0, 0
        for (mul, ir), section in zip(self.output_irreps, self.output_irreps.slices()):
            if ir == o3.Irrep("0e"):
                output.append(scalars[:, scalar_offset:scalar_offset + mul])
                scalar_offset += mul
            else:
                block = basis[:, section].reshape(len(basis), mul, ir.dim)
                if gates is not None:
                    block = block * gates[:, tensor_offset:tensor_offset + mul, None]
                output.append(block.flatten(1))
                tensor_offset += mul
        return torch.cat(output, dim=-1)


class ScalarChargeEncoder(ChargeStateEncoder):
    """Baseline: q enters 0e features; initial non-scalar features ignore q."""

    def tensor_gates(self, conditioning):
        return None


class GatedChargeEncoder(ChargeStateEncoder):
    """Main encoder: q also gates every existing non-scalar feature channel."""

    def __init__(self, input_irreps, output_irreps, hidden_features=64, include_dipoles=True):
        super().__init__(input_irreps, output_irreps, hidden_features, include_dipoles)
        self.gates = nn.Linear(hidden_features, self.tensor_width) if self.tensor_width else None

    def tensor_gates(self, conditioning):
        # Smooth bounded amplitude modulation with a unit-centered range.
        return 2 * torch.sigmoid(self.gates(conditioning)) if self.gates is not None else None


class EquivariantStateFusion(nn.Module):
    """Mix local/global features and gate a residual update, retaining irreps."""

    def __init__(self, irreps, hidden_features=64):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.mix = o3.Linear(self.irreps + self.irreps, self.irreps, biases=True)
        self.gates = nn.Sequential(
            nn.Linear(self.irreps.num_irreps, hidden_features), nn.SiLU(),
            nn.Linear(hidden_features, self.irreps.num_irreps),
        )

    def forward(self, local, global_features):
        mixed = self.mix(torch.cat((local, global_features), dim=-1))
        gates = torch.sigmoid(self.gates(invariant_features(mixed, self.irreps)))
        parts, offset = [], 0
        for (mul, ir), section in zip(self.irreps, self.irreps.slices()):
            block = mixed[:, section].reshape(len(mixed), mul, ir.dim)
            if ir == o3.Irrep("0e"):
                block = torch.nn.functional.silu(block)
            block = block * gates[:, offset:offset + mul, None]
            parts.append(block.flatten(1))
            offset += mul
        return local + torch.cat(parts, dim=-1) / math.sqrt(2.0)


class StateEFAEnergy(nn.Module):
    """Boundary-independent U(R,h,q,p) for MultipoleVariationalOutput.

    Every EFA layer computes Q, K and V from the current state features. Local
    features survive through gated residual fusion; a learned invariant tensor
    product readout jointly uses the final local and global representations.
    No prescribed electrostatic term, convexity or state solver is added.

    include_dipoles=False selects the charge-only state [N, 1] and bind contract
    (components=1, state_kind="scalar"). The default [N, 4] state plugs directly
    into the existing MultipoleVariationalOutput without changing its solver.

    This state model supports finite systems. It does not inherit the legacy
    adapter's raw-lattice periodic approximation. Bind before optimizer setup.
    """

    def __init__(self, hidden_irreps="64x0e + 16x1o + 8x2e", num_layers=2,
                 hidden_features=32, num_directions=110, max_frequency=2 * math.pi,
                 max_length=32.0, encoder="gated", tensor_integration=True,
                 integration_lmax=2, readout_features=64, coupling_features=16,
                 energy_scale=1.0, charge_scale=1.0, dipole_scale=1.0,
                 include_dipoles=True):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if encoder not in ("gated", "scalar"):
            raise ValueError("encoder must be 'gated' or 'scalar'")
        if not all(math.isfinite(v) and v > 0 for v in (energy_scale, charge_scale, dipole_scale)):
            raise ValueError("energy and state scales must be finite and positive")
        self.irreps = o3.Irreps(hidden_irreps)
        self.num_layers = int(num_layers)
        self.encoder_kind = encoder
        self.include_dipoles = bool(include_dipoles)
        self.components = 4 if self.include_dipoles else 1
        self.state_kind = "multipole" if self.include_dipoles else "scalar"
        self.readout_features = int(readout_features)
        self.energy_scale = float(energy_scale)
        self.feature_spec = None
        self.encoder = None
        scales = [charge_scale, dipole_scale, dipole_scale, dipole_scale] if self.include_dipoles else [charge_scale]
        self.register_buffer("state_scale", torch.tensor(scales))
        self.attention = nn.ModuleList(EquivariantEFA(
            self.irreps, self.irreps, hidden_features=hidden_features,
            num_directions=num_directions, max_frequency=max_frequency,
            max_length=max_length, tensor_integration=tensor_integration,
            integration_lmax=integration_lmax) for _ in range(self.num_layers))
        self.fusion = nn.ModuleList(EquivariantStateFusion(self.irreps, readout_features)
                                    for _ in range(self.num_layers))
        self.readout = InvariantEnergyReadout(self.irreps, self.irreps,
                                             readout_features, coupling_features)
        self._contract = {
            "schema": "state_efa_energy_v1",
            "state_order": ("q", "px", "py", "pz") if self.include_dipoles else ("q",),
            "state_units": ("e", "e*Angstrom", "e*Angstrom", "e*Angstrom") if self.include_dipoles else ("e",),
            "include_dipoles": self.include_dipoles,
            "hidden_irreps": str(self.irreps), "num_layers": self.num_layers,
            "hidden_features": hidden_features, "num_directions": num_directions,
            "max_frequency": float(max_frequency), "max_length": float(max_length),
            "encoder": encoder, "tensor_integration": bool(tensor_integration),
            "integration_lmax": integration_lmax, "readout_features": readout_features,
            "coupling_features": coupling_features, "energy_scale": float(energy_scale),
            "charge_scale": float(charge_scale), "dipole_scale": float(dipole_scale),
            "periodicity": "finite_only", "convexity": "not_imposed",
        }

    def bind(self, feature_spec, components=None, state_kind=None, **unused):
        if ((components is not None and components != self.components)
                or (state_kind is not None and state_kind != self.state_kind)):
            raise ValueError(f"StateEFAEnergy requires state_kind={self.state_kind!r} and components={self.components}")
        if self.feature_spec is not None:
            if self.feature_spec != feature_spec:
                raise ValueError("state EFA is already bound to a different feature specification")
            return self
        encoder_class = GatedChargeEncoder if self.encoder_kind == "gated" else ScalarChargeEncoder
        self.encoder = encoder_class(feature_spec.irreps, self.irreps,
                                     hidden_features=self.readout_features,
                                     include_dipoles=self.include_dipoles).to(self.state_scale)
        self.feature_spec = feature_spec
        return self

    @staticmethod
    def prepare_context(context):
        if "pbc" in context and bool(context["pbc"].any()):
            raise ValueError("StateEFAEnergy currently supports finite, nonperiodic systems only")
        positions = context["positions"]
        if positions.ndim != 2 or positions.shape[-1] != 3 or not positions.shape[0]:
            raise ValueError("positions must have shape [n_atoms, 3] with n_atoms > 0")
        if context["features"].shape[0] != positions.shape[0]:
            raise ValueError("feature and position atom counts differ")
        # Geometry features remain connected to positions; no N-by-N cache.
        return context

    def initial_state(self, context):
        return context["features"].new_zeros((context["features"].shape[0], self.components))

    def forward(self, context, state):
        if self.encoder is None:
            raise RuntimeError("bind StateEFAEnergy before constructing the optimizer")
        context = self.prepare_context(context)
        if state.shape != (context["features"].shape[0], self.components):
            raise ValueError(f"state must have shape [n_atoms, {self.components}]")
        values = self.encoder(context["features"], state / self.state_scale)
        for attention, fusion in zip(self.attention, self.fusion):
            global_features = attention(values, context["positions"])
            values = fusion(values, global_features)
        return self.energy_scale * self.readout(values, global_features).sum()

    def get_extra_state(self):
        specification = None
        if self.feature_spec is not None:
            specification = {
                "type": type(self.feature_spec).__name__,
                "fields": {key: str(value) if key == "irreps" else value
                           for key, value in vars(self.feature_spec).items()},
            }
        return {**self._contract, "feature_spec": specification}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("state EFA energy checkpoint contract differs; bind the matching feature specification first")


__all__ = ["ChargeStateEncoder", "ScalarChargeEncoder", "GatedChargeEncoder",
           "EquivariantStateFusion", "StateEFAEnergy"]
