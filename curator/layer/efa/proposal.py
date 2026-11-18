"""Global EFA charge proposal conditioned on charge and potential boundaries."""

import copy
import math

import torch
from e3nn import o3
from torch import nn

from .._variational_proposal import EquivariantFeatureNorm
from .blocks import EquivariantEFA
from .geometry import structure_cells
from .readout import invariant_features
from .state import EquivariantStateFusion


class EFAChargeProposal(nn.Module):
    """Predict physical charges using global EFA and positive Fukui weights.

    The full backbone features bypass the proposal's hidden representation and
    enter its scalar readout. Total charge and charge per atom condition the
    features before global attention. Gauge-projected potentials and an open
    total-charge flag enter a separate scalar branch. The final output is
    unbounded. A fixed total charge uses learned positive Fukui weights;
    without that equality the raw prediction is returned unchanged.

    ``forward(context)`` returns an already feasible [N, 1] state. The existing
    VariationalEnergyOutput projection is retained as a numerical feasibility
    guard; in exact arithmetic it leaves both this state and its derivatives
    unchanged. Its stationarity residual remains the ordinary tangent energy
    gradient, not a Fukui-weighted residual.

    Supported boundaries are fixed total charge, fixed potentials (including
    group potentials), and potentials with one fixed total-charge equality.
    All EFA intermediates grow linearly with N at fixed width and quadrature
    size. Periodic geometry uses the same raw-lattice encoding as EFAAdapter;
    partial PBC is unsupported.
    """

    def __init__(self, feature_spec=None, hidden_irreps="64x0e + 16x1o",
                 num_layers=2, hidden_features=32, readout_features=64,
                 num_directions=110, max_frequency=2 * math.pi,
                 max_length=32.0, charge_scale=1.0, potential_scale=1.0):
        super().__init__()
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be a positive integer")
        if not isinstance(readout_features, int) or readout_features < 1:
            raise ValueError("readout_features must be a positive integer")
        if not math.isfinite(charge_scale) or charge_scale <= 0:
            raise ValueError("charge_scale must be finite and positive")
        if not math.isfinite(potential_scale) or potential_scale <= 0:
            raise ValueError("potential_scale must be finite and positive")
        self.irreps = o3.Irreps(hidden_irreps)
        if not any(ir == o3.Irrep("0e") for _, ir in self.irreps):
            raise ValueError("proposal hidden irreps require even scalar channels")
        self.num_layers = num_layers
        self.readout_features = readout_features
        self.charge_scale = float(charge_scale)
        self.potential_scale = float(potential_scale)
        self.feature_spec = None
        self.input = self.readout = None
        self.boundary_input = o3.Linear("2x0e", self.irreps)
        self.potential_input = o3.Linear("2x0e", self.irreps, biases=False)
        self.norm = EquivariantFeatureNorm(self.irreps)
        self.attention = nn.ModuleList(EquivariantEFA(
            self.irreps, self.irreps, hidden_features=hidden_features,
            num_directions=num_directions, max_frequency=max_frequency,
            max_length=max_length, tensor_integration=False,
        ) for _ in range(num_layers))
        self.fusion = nn.ModuleList(EquivariantStateFusion(self.irreps, readout_features)
                                    for _ in range(num_layers))
        self.register_buffer("_dtype_anchor", torch.empty(0))
        if feature_spec is not None:
            self.bind(feature_spec)

    @classmethod
    def from_fixed_charge(cls, proposal, *, potential_scale=1.0):
        """Explicitly upgrade a v1 proposal before constructing its optimizer.

        Old serialized proposals retain their v1 contract and parameter set.
        Upgrading copies all learned weights and adds a zero-initialized
        potential branch; fixed-charge predictions and derivatives are kept.
        """
        if not isinstance(proposal, cls) or hasattr(proposal, "potential_input"):
            raise ValueError("expected a legacy fixed-charge EFA proposal")
        if not math.isfinite(potential_scale) or potential_scale <= 0:
            raise ValueError("potential_scale must be finite and positive")
        upgraded = copy.deepcopy(proposal)
        upgraded.potential_scale = float(potential_scale)
        upgraded.potential_input = o3.Linear("2x0e", upgraded.irreps, biases=False).to(
            upgraded._dtype_anchor)
        with torch.no_grad():
            for parameter in upgraded.potential_input.parameters():
                parameter.zero_()
        upgraded.potential_input.train(upgraded.training)
        return upgraded

    def bind(self, feature_spec, components=1, state_kind="scalar", **unused):
        if components != 1 or state_kind != "scalar":
            raise ValueError("EFAChargeProposal requires one scalar charge per atom")
        if self.feature_spec is not None:
            if self.feature_spec != feature_spec:
                raise ValueError("proposal is already bound to different features")
            return self
        self.feature_spec = feature_spec
        self.input = o3.Linear(feature_spec.irreps, self.irreps, biases=True).to(self._dtype_anchor)
        width = feature_spec.irreps.num_irreps + self.irreps.num_irreps + 2
        self.readout = nn.Sequential(nn.Linear(width, self.readout_features), nn.SiLU(),
                                     nn.Linear(self.readout_features, 2)).to(self._dtype_anchor)
        return self

    @staticmethod
    def _total_charge(context):
        features = context["features"]
        matrix, target = context["constraint_matrix"], context["constraint_target"]
        if (matrix.shape != (1, len(features)) or target.shape != (1,)
                or not torch.equal(matrix, torch.ones_like(matrix))):
            raise ValueError("EFAChargeProposal requires exactly one fixed total-charge equality")
        if (matrix.dtype != features.dtype or target.dtype != features.dtype
                or matrix.device != features.device or target.device != features.device
                or not bool(torch.isfinite(target).all())):
            raise ValueError("total-charge constraint must be finite and match feature dtype/device")
        for name in ("control_matrix", "environment_matrix"):
            if name in context and context[name].numel():
                raise ValueError("EFAChargeProposal currently supports fixed charge without other controls")
        for name in ("field", "network_field"):
            if name in context and bool(torch.count_nonzero(context[name])):
                raise ValueError("EFAChargeProposal currently supports fixed charge without external fields")
        return target.reshape(())

    def _conditions(self, context):
        features = context["features"]
        matrix, target = context["constraint_matrix"], context["constraint_target"]
        if not hasattr(self, "potential_input"):
            charge = self._total_charge(context)
            potential = None
        else:
            if (matrix.shape not in ((0, len(features)), (1, len(features)))
                    or target.shape != (matrix.shape[0],)
                    or (matrix.shape[0] and not torch.equal(matrix, torch.ones_like(matrix)))):
                raise ValueError("EFAChargeProposal accepts only an optional fixed total-charge equality")
            if (matrix.dtype != features.dtype or target.dtype != features.dtype
                    or matrix.device != features.device or target.device != features.device
                    or not bool(torch.isfinite(target).all())):
                raise ValueError("charge constraints must be finite and match feature dtype/device")
            if context.get("environment_matrix", matrix[:0]).numel():
                raise ValueError("EFAChargeProposal does not support Robin environments")
            field = context["network_field"]
            if (field.shape != (len(features), 1) or field.dtype != features.dtype
                    or field.device != features.device or not bool(torch.isfinite(field).all())):
                raise ValueError("network_field must be finite [N, 1] and match feature dtype/device")
            charge = target.reshape(()) if matrix.shape[0] else features.new_zeros(())
            potential = torch.cat((field / self.potential_scale,
                                   torch.full_like(field, float(matrix.shape[0] == 0))), dim=-1)
        condition = torch.stack((charge, charge / len(features))) / self.charge_scale
        return condition.expand(len(features), -1), potential

    def predict_components(self, context):
        """Return raw charges and normalized Fukui weights, each shaped [N, 1]."""
        if self.feature_spec is None:
            raise RuntimeError("bind EFAChargeProposal before use")
        features, positions = context["features"], context["positions"]
        if (features.ndim != 2 or features.shape[1] != self.feature_spec.irreps.dim
                or positions.shape != (len(features), 3) or not len(features)):
            raise ValueError("proposal features and positions must describe the same nonempty atom set")
        condition, potential = self._conditions(context)
        cell = structure_cells(context, 1)[0]
        values = self.input(features) + self.boundary_input(condition)
        if potential is not None:
            values = values + self.potential_input(potential)
        for attention, fusion in zip(self.attention, self.fusion):
            update = attention(self.norm(values), positions, cell=cell)
            values = fusion(values, update)
        invariants = torch.cat((invariant_features(features, self.feature_spec.irreps),
                                invariant_features(values, self.irreps), condition), dim=-1)
        output = self.readout(invariants)
        return self.charge_scale * output[:, :1], torch.softmax(output[:, 1:], dim=0)

    def forward(self, context):
        raw, weights = self.predict_components(context)
        if not context["constraint_matrix"].shape[0]:
            return raw
        return raw + weights * (context["constraint_target"].reshape(()) - raw.sum())

    def checkpoint_contract(self):
        core = self.attention[0].attention
        specification = None if self.feature_spec is None else {
            "type": type(self.feature_spec).__name__,
            "fields": {key: str(value) if key == "irreps" else value
                       for key, value in vars(self.feature_spec).items()},
        }
        contract = {
            "schema": "efa_charge_proposal_v1", "feature_spec": specification,
            "hidden_irreps": str(self.irreps), "num_layers": self.num_layers,
            "hidden_features": core.hidden_features, "num_directions": core.num_directions,
            "max_frequency": core.max_frequency, "max_length": core.max_length,
            "readout_features": self.readout_features, "charge_scale": self.charge_scale,
            "state_unit": "e", "projection": "positive_normalized_fukui_fixed_total_charge",
            "periodicity": "finite_or_raw_lattice", "tensor_integration": False,
        }
        if hasattr(self, "potential_input"):
            contract.update(schema="efa_charge_proposal_v2", potential_scale=self.potential_scale,
                            projection="positive_fukui_when_total_charge_fixed",
                            conditioning="total_charge_gauge_projected_potential_open_charge")
        return contract

    def get_extra_state(self):
        return self.checkpoint_contract()

    def set_extra_state(self, state):
        if state != self.checkpoint_contract():
            raise ValueError("EFA charge proposal checkpoint contract differs")


__all__ = ["EFAChargeProposal"]
