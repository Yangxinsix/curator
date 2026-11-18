"""One shared geometric EFA communication followed by local charge decoding."""

import copy
import math

import torch
from e3nn import o3
from torch import nn
from torch.nn import functional as F

from .._variational_proposal import EquivariantFeatureNorm
from .geometry import structure_cells
from .proposal import EFAChargeProposal
from .readout import invariant_features
from .state import EquivariantStateFusion
from .two_scale import TwoScaleQConditionedEFAEnergy


class SharedGeometryChargeProposal(EFAChargeProposal):
    """Decode q and positive Fukui weights from shared geometric attention.

    The energy owns Q/K. This proposal owns only its equivariant Values and
    local gated decoder, including the uncompressed backbone-feature bypass.
    Boundary potentials enter Values before communication, so a remote
    potential can change a local raw charge even without a charge constraint.
    This is a new architecture, not an equivalent two-layer proposal.
    """

    def __init__(self, feature_spec=None, hidden_irreps="64x0e + 16x1o",
                 readout_features=64, charge_scale=1.0, potential_scale=1.0):
        nn.Module.__init__(self)
        if not isinstance(readout_features, int) or readout_features < 1:
            raise ValueError("readout_features must be a positive integer")
        if not all(math.isfinite(v) and v > 0 for v in (charge_scale, potential_scale)):
            raise ValueError("charge and potential scales must be finite and positive")
        self.irreps = o3.Irreps(hidden_irreps)
        if not any(ir == o3.Irrep("0e") for _, ir in self.irreps):
            raise ValueError("proposal hidden irreps require even scalar channels")
        self.readout_features = readout_features
        self.charge_scale, self.potential_scale = float(charge_scale), float(potential_scale)
        self.feature_spec = None
        self.input = self.readout = None
        self.boundary_input = o3.Linear("2x0e", self.irreps)
        self.potential_input = o3.Linear("2x0e", self.irreps, biases=False)
        self.norm = EquivariantFeatureNorm(self.irreps)
        self.fusion = EquivariantStateFusion(self.irreps, readout_features)
        self.register_buffer("_dtype_anchor", torch.empty(0))
        if feature_spec is not None:
            self.bind(feature_spec)

    def local_features(self, context):
        if self.feature_spec is None:
            raise RuntimeError("bind shared charge proposal before use")
        condition, potential = self._conditions(context)
        local = (self.input(context["features"]) + self.boundary_input(condition)
                 + self.potential_input(potential))
        return local, condition

    def predict_components(self, context):
        if "_shared_efa" not in context:
            raise RuntimeError("shared proposal requires energy.prepare_proposal_context(context, proposal)")
        cache = context["_shared_efa"]
        values = self.fusion(cache["local"], cache["global"])
        invariants = torch.cat((
            invariant_features(context["features"], self.feature_spec.irreps),
            invariant_features(values, self.irreps), cache["condition"],
        ), dim=-1)
        output = self.readout(invariants)
        return self.charge_scale * output[:, :1], torch.softmax(output[:, 1:], dim=0)

    def checkpoint_contract(self):
        specification = None if self.feature_spec is None else {
            "type": type(self.feature_spec).__name__,
            "fields": {key: str(value) if key == "irreps" else value
                       for key, value in vars(self.feature_spec).items()},
        }
        return {
            "schema": "shared_geometry_charge_proposal_v1", "feature_spec": specification,
            "hidden_irreps": str(self.irreps), "readout_features": self.readout_features,
            "charge_scale": self.charge_scale, "potential_scale": self.potential_scale,
            "communication": "energy_geometry_query_key", "state_unit": "e",
            "projection": "positive_fukui_when_total_charge_fixed",
            "conditioning": "total_charge_gauge_projected_potential_open_charge",
        }


class SharedTwoScaleQConditionedEFAEnergy(TwoScaleQConditionedEFAEnergy):
    """Keep U(R,q) unchanged while sharing its geometric kernel with proposal.

    Prepare Q/K once and aggregate concatenated geometry/proposal Values.
    After the proposal predicts q, aggregate q*B(h) with the same Q/K. Thus
    the full model uses two aggregations, not one. Caches belong to one
    forward's context, never to the module or its checkpoint. The energy reads
    neither proposal features nor boundary conditions at fixed R,h,q.
    """

    requires_shared_context_reuse = True

    @classmethod
    def from_two_scale(cls, energy):
        """Copy every existing two-scale energy weight; no proposal is copied."""
        if type(energy) is not TwoScaleQConditionedEFAEnergy or energy.feature_spec is None:
            raise TypeError("expected a bound TwoScaleQConditionedEFAEnergy")
        reference = next(energy.parameters())
        converted = cls(
            hidden_features=energy.hidden_features,
            num_directions=energy.attention.num_directions,
            max_frequency=energy.attention.max_frequency,
            max_length=energy.attention.max_length,
            energy_scale=energy.energy_scale,
            use_equivariant_queries=energy.use_equivariant_queries,
            charge_features=energy.charge_features,
            curvature_floor=energy.curvature_floor,
            hardness_init=energy.curvature_floor + 1.0,
        ).to(reference)
        converted.bind(copy.deepcopy(energy.feature_spec))
        state = energy.state_dict()
        state["_extra_state"] = converted.get_extra_state()
        converted.load_state_dict(state)
        converted.attention = copy.deepcopy(energy.attention)
        converted.train(energy.training)
        return converted

    def prepare_proposal_context(self, context, proposal):
        if not isinstance(proposal, SharedGeometryChargeProposal):
            raise TypeError("shared energy requires SharedGeometryChargeProposal")
        features, positions = context["features"], context["positions"]
        if (self.feature_spec is None or features.ndim != 2
                or features.shape[1] != self.feature_spec.irreps.dim
                or not len(features) or positions.shape != (len(features), 3)):
            raise ValueError("features and positions must match the bound nonempty structure")
        query, key, value = self._project_features(features, context)
        prepared = self.attention.prepare(
            query, key, positions, cell=structure_cells(context, 1)[0])
        local, condition = proposal.local_features(context)
        geometry, global_features = self.attention.aggregate(
            prepared, torch.cat((value, proposal.norm(local)), dim=-1),
        ).split((self.hidden_features, proposal.irreps.dim), dim=-1)
        return {**context, "_shared_efa": {
            "prepared": prepared, "geometry": geometry, "local": local,
            "global": global_features, "condition": condition,
        }}

    def _attention_terms(self, context, state):
        if "_shared_efa" not in context:
            # Independent U(R,q) remains available for equilibrium diagnostics.
            return super()._attention_terms(context, state)
        features = context["features"]
        if (state.shape != (len(features), 1) or state.dtype != features.dtype
                or state.device != features.device):
            raise ValueError("charge state must be [N, 1] and match feature dtype/device")
        cache = context["_shared_efa"]
        geometry = cache["geometry"]
        charge = self.attention.aggregate(
            cache["prepared"], state * self.charge_value(features))
        width = self.hidden_features
        offset = F.linear(geometry, self.charge_affine.weight[:, :width], self.charge_affine.bias)
        response = F.linear(torch.cat((charge, state), dim=-1), self.charge_affine.weight[:, width:])
        return geometry, offset, response

    def get_extra_state(self):
        return {**super().get_extra_state(), "schema": "shared_two_scale_q_efa_energy_v1"}


__all__ = ["SharedGeometryChargeProposal", "SharedTwoScaleQConditionedEFAEnergy"]
