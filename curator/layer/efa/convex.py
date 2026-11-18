"""Geometry EFA plus a strongly convex charge energy using the same attention."""

import copy
import math

import torch
from e3nn import o3
from torch import nn
from torch.nn import functional as F

from curator.data import properties
from .adapter import EFAAdapter
from .geometry import structure_cells


class ConvexQConditionedEFAEnergy(EFAAdapter):
    """An unrestricted geometry branch and a convex charge branch.

    Geometry alone determines Q and K. One attention call carries both the
    ordinary value V(h) and the charge-gated value q * B(h). Its two outputs,
    g and z, enter an ordinary geometry readout and a separate affine charge
    readout a = Linear([g, z, q]). Positive output weights, softplus and a
    positive local hardness make the charge Hessian at least ``hardness * I``.
    The q=0 charge energy is subtracted without another attention evaluation.

    ``forward(context, state)`` returns only the EFA correction; the existing
    variational output adds the MACE baseline and boundary energy once. State
    has shape [N, 1] in e, energy is in eV, and hardness is in eV/e². The floor
    and initial hardness are configurable model settings, not measured atomic
    hardnesses. The complete declared backbone feature tensor is retained.
    """

    def __init__(self, feature_spec=None, hidden_features=32, num_directions=110,
                 max_frequency=2 * math.pi, max_length=32.0, energy_scale=0.01,
                 use_equivariant_queries=True, charge_features=None,
                 curvature_floor=0.01, hardness_init=0.1, output_weight_init=None):
        charge_features = hidden_features if charge_features is None else charge_features
        if not isinstance(charge_features, int) or charge_features < 1:
            raise ValueError("charge_features must be a positive integer")
        if not math.isfinite(curvature_floor) or curvature_floor <= 0:
            raise ValueError("curvature_floor must be finite and positive")
        if not math.isfinite(hardness_init) or hardness_init <= curvature_floor:
            raise ValueError("hardness_init must be finite and exceed curvature_floor")
        output_weight_init = (1 / charge_features if output_weight_init is None
                              else output_weight_init)
        if not math.isfinite(output_weight_init) or output_weight_init <= 0:
            raise ValueError("output_weight_init must be finite and positive")
        super().__init__(
            hidden_features=hidden_features, num_directions=num_directions,
            max_frequency=max_frequency, max_length=max_length,
            energy_scale=energy_scale, use_equivariant_queries=use_equivariant_queries,
        )
        self.charge_features = charge_features
        self.curvature_floor = float(curvature_floor)
        self.charge_value = self.charge_affine = None
        self.raw_output_weights = nn.Parameter(torch.full(
            (charge_features,), self._inverse_softplus(output_weight_init)))
        self.raw_hardness = nn.Parameter(torch.tensor(
            self._inverse_softplus(hardness_init - curvature_floor)))
        if feature_spec is not None:
            self.bind(feature_spec)

    @staticmethod
    def _inverse_softplus(value):
        return value + math.log(-math.expm1(-value))

    @property
    def hardness(self):
        """Current lower bound on the charge Hessian eigenvalues, in eV/e²."""
        return self.curvature_floor + F.softplus(self.raw_hardness)

    @property
    def positive_output_weights(self):
        return F.softplus(self.raw_output_weights)

    def bind(self, feature_spec, *, components=1, state_kind="scalar", **unused):
        if components != 1 or state_kind != "scalar":
            raise ValueError("convex q-EFA requires one scalar charge per atom")
        if self.feature_spec is not None:
            return super().bind(feature_spec)
        super().bind(feature_spec)
        self.charge_value = o3.Linear(
            feature_spec.irreps, f"{self.hidden_features}x0e", biases=True,
        ).to(self._dtype_anchor)
        self.charge_affine = nn.Linear(
            2 * self.hidden_features + 1, self.charge_features,
        ).to(self._dtype_anchor)
        return self

    @classmethod
    def from_adapter(cls, adapter, **charge_options):
        """Retain a plain EFA adapter's entire learned geometry branch."""
        if type(adapter) is not EFAAdapter:
            raise TypeError("expected a plain EFAAdapter")
        if adapter.feature_spec is None:
            raise ValueError("bind the EFA adapter before converting it")
        reference = next(adapter.parameters())
        energy = cls(
            hidden_features=adapter.hidden_features,
            num_directions=adapter.attention.num_directions,
            max_frequency=adapter.attention.max_frequency,
            max_length=adapter.attention.max_length,
            energy_scale=adapter.energy_scale,
            use_equivariant_queries=adapter.use_equivariant_queries,
            **charge_options,
        ).to(device=reference.device, dtype=reference.dtype)
        energy.bind(copy.deepcopy(adapter.feature_spec))
        for name in ("attention", "query", "key", "value", "readout"):
            setattr(energy, name, copy.deepcopy(getattr(adapter, name)))
        energy.train(adapter.training)
        return energy

    def _attention_terms(self, context, state):
        """Return geometry, q-independent offset and the affine q response."""
        if self.feature_spec is None:
            raise RuntimeError("bind convex q-EFA energy before evaluation")
        features, positions = context["features"], context["positions"]
        if features.ndim != 2 or features.shape[1] != self.feature_spec.irreps.dim:
            raise ValueError("feature shape does not match its specification")
        if len(features) == 0 or positions.shape != (len(features), 3):
            raise ValueError("positions must describe the same nonempty structure")
        if state.shape != (len(features), 1):
            raise ValueError("convex q-EFA state must have shape [N, 1]")
        if state.dtype != features.dtype or state.device != features.device:
            raise ValueError("charge state must match feature dtype and device")
        geometry_data = {properties.pbc: context["pbc"]}
        if context.get("cell") is not None:
            geometry_data[properties.cell] = context["cell"]
        cell = structure_cells(geometry_data, 1)[0]
        query, key, value = super()._project_features(features, context)
        values = torch.cat((value, state * self.charge_value(features)), dim=-1)
        geometry, charge = self.attention(
            query, key, values, positions, cell=cell,
        ).split(self.hidden_features, dim=-1)
        # Splitting one affine map caches a(q=0) while keeping z and q affine.
        width = self.hidden_features
        offset = F.linear(geometry, self.charge_affine.weight[:, :width],
                          self.charge_affine.bias)
        response = F.linear(torch.cat((charge, state), dim=-1),
                            self.charge_affine.weight[:, width:])
        return geometry, offset, response

    def forward(self, context, state):
        geometry, offset, response = self._attention_terms(context, state)
        charge_energy = (
            (F.softplus(offset + response) - F.softplus(offset))
            * self.positive_output_weights
        ).sum()
        return (self.energy_scale * (self.readout(geometry).sum() + charge_energy)
                + 0.5 * self.hardness * state.square().sum())

    def get_extra_state(self):
        return {
            "schema": "convex_q_efa_energy_v1",
            "state_unit": "e", "energy_unit": "eV", "hardness_unit": "eV/e^2",
            "hidden_features": self.hidden_features,
            "charge_features": self.charge_features,
            "energy_scale": self.energy_scale,
            "curvature_floor": self.curvature_floor,
            "use_equivariant_queries": self.use_equivariant_queries,
            "num_directions": self.attention.num_directions,
            "max_frequency": self.attention.max_frequency,
            "max_length": self.attention.max_length,
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("convex q-EFA checkpoint contract differs")


__all__ = ["ConvexQConditionedEFAEnergy"]
