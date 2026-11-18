"""Unconstrained latent-q EFA trained through energy and coordinate forces."""

import math

import torch
from torch import nn

from curator.data import properties
from .readout import invariant_features
from .state import StateEFAEnergy


class LatentChargeEFAAdapter(nn.Module):
    """Add U(h(R), R, q(h(R))) without a variational proposal or constraints.

    q is an unconstrained latent scalar, not a calibrated atomic charge. A local
    MLP reads invariant MPNN features; no total charge, charge labels, external
    field, projection, dipoles, solver or stationarity residual is used. The
    coordinate-force output must follow this adapter so all q(R) derivatives
    remain in the energy/force training graph.

    ``charge_mode='zero'`` retains the same parameter layout but freezes and
    bypasses the q head. Geometry still reaches every StateEFA energy block.
    ``zero_init_energy`` zeros only the final energy readout, preserving an
    existing backbone's initial energy and forces. Earlier layers start learning
    after this readout's first update. Bind before constructing the optimizer.

    ``charge_scale`` is a latent-coordinate normalization, not a physical unit.
    This adapter supports finite, nonperiodic structures only.
    """

    repr_callback = True

    def __init__(self, feature_spec=None, charge_mode="learned",
                 charge_hidden_features=32, zero_init_energy=True,
                 hidden_irreps="64x0e + 16x1o + 8x2e", num_layers=2,
                 hidden_features=32, num_directions=110,
                 max_frequency=2 * math.pi, max_length=32.0, encoder="gated",
                 tensor_integration=True, integration_lmax=2,
                 readout_features=64, coupling_features=16,
                 energy_scale=1.0, charge_scale=1.0):
        super().__init__()
        if charge_mode not in ("learned", "zero"):
            raise ValueError("charge_mode must be 'learned' or 'zero'")
        if charge_hidden_features < 1:
            raise ValueError("charge_hidden_features must be positive")
        self.charge_mode = charge_mode
        self.charge_hidden_features = int(charge_hidden_features)
        self.zero_init_energy = bool(zero_init_energy)
        self.feature_spec = None
        self.q_head = None
        self.energy_model = StateEFAEnergy(
            hidden_irreps=hidden_irreps, num_layers=num_layers,
            hidden_features=hidden_features, num_directions=num_directions,
            max_frequency=max_frequency, max_length=max_length, encoder=encoder,
            tensor_integration=tensor_integration, integration_lmax=integration_lmax,
            readout_features=readout_features, coupling_features=coupling_features,
            energy_scale=energy_scale, charge_scale=charge_scale,
            include_dipoles=False,
        )
        if self.zero_init_energy:
            # Do this once at construction, never in bind or checkpoint loading.
            nn.init.zeros_(self.energy_model.readout.mlp[-1].weight)
        self.model_outputs = [properties.energy, "efa_energy", "efa_latent_q"]
        if feature_spec is not None:
            self.bind(feature_spec)

    def bind(self, feature_spec):
        if self.feature_spec is not None:
            if self.feature_spec != feature_spec:
                raise ValueError("latent EFA is already bound to a different feature specification")
            return self
        self.energy_model.bind(feature_spec)
        self.q_head = nn.Sequential(
            nn.Linear(feature_spec.irreps.num_irreps, self.charge_hidden_features),
            nn.SiLU(),
            nn.Linear(self.charge_hidden_features, 1),
        ).to(self.energy_model.state_scale)
        self.q_head.requires_grad_(self.charge_mode == "learned")
        self.feature_spec = feature_spec
        return self

    def register_repr_callback(self, model):
        self.bind(model.representation.response_feature_spec())
        reference = next(model.representation.parameters(), None)
        if reference is not None:
            self.to(device=reference.device, dtype=reference.dtype)

    def predict_charge(self, features):
        """Return differentiable latent q in the input atom order, shape [N, 1]."""
        if self.feature_spec is None:
            raise RuntimeError("bind latent EFA before constructing the optimizer")
        if features.ndim != 2 or features.shape[-1] != self.feature_spec.irreps.dim:
            raise ValueError("features do not match latent EFA input irreps")
        if self.charge_mode == "zero":
            return features.new_zeros((features.shape[0], 1))
        return self.q_head(invariant_features(features, self.feature_spec.irreps))

    def forward(self, data):
        if self.feature_spec is None:
            raise RuntimeError("bind latent EFA before constructing the optimizer")
        positions = data[properties.positions]
        features = self.feature_spec.flatten(data)
        if positions.ndim != 2 or positions.shape != (len(features), 3):
            raise ValueError("positions must have shape [n_atoms, 3] matching the features")
        counts = data[properties.n_atoms].reshape(-1)
        if (counts.dtype not in (torch.int32, torch.int64) or not counts.numel()
                or bool((counts <= 0).any()) or int(counts.sum()) != len(positions)):
            raise ValueError("n_atoms must describe nonempty structures")
        if properties.pbc not in data:
            raise ValueError("latent EFA requires explicit pbc flags")
        pbc = data[properties.pbc]
        allowed_shapes = [(len(counts), 3)] + ([(3,)] if len(counts) == 1 else [])
        if pbc.dtype != torch.bool or tuple(pbc.shape) not in allowed_shapes:
            raise ValueError("pbc must be boolean with shape [n_structures, 3]")
        if bool(pbc.any()):
            raise ValueError("latent EFA supports finite, nonperiodic systems only")
        pbc = pbc.reshape(len(counts), 3)
        batch = data.get(properties.image_idx)
        if batch is None:
            if len(counts) != 1:
                raise ValueError("batched latent EFA requires image_idx")
            batch = torch.zeros(len(positions), device=positions.device, dtype=torch.long)
        if (batch.dtype not in (torch.int32, torch.int64)
                or batch.shape != (len(positions),)
                or bool(((batch < 0) | (batch >= len(counts))).any())):
            raise ValueError("invalid latent EFA image_idx")
        if data[properties.energy].numel() != len(counts):
            raise ValueError("base energy must contain one value per structure")
        q = self.predict_charge(features)
        corrections = []
        for graph, count in enumerate(counts):
            mask = batch == graph
            if int(mask.sum()) != int(count):
                raise ValueError("image_idx and n_atoms disagree")
            context = {"features": features[mask], "positions": positions[mask], "pbc": pbc[graph]}
            corrections.append(self.energy_model(context, q[mask]))
        correction = torch.stack(corrections)
        data["efa_energy"] = correction
        data["efa_latent_q"] = q
        data[properties.energy] = data[properties.energy] + correction.reshape_as(data[properties.energy])
        return data

    def get_extra_state(self):
        # StateEFA separately serializes its architecture and feature contract.
        return {
            "schema": "latent_charge_efa_adapter_v1",
            "charge_mode": self.charge_mode,
            "charge_hidden_features": self.charge_hidden_features,
            "zero_init_energy": self.zero_init_energy,
            "q_semantics": "unconstrained_latent_scalar",
            "training_path": "full_energy_and_coordinate_forces",
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("latent EFA checkpoint contract differs; instantiate the matching configuration")


__all__ = ["LatentChargeEFAAdapter"]
