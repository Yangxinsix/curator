"""Internal-energy labels keep boundary-potential forces and residuals."""

import copy
import io
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from curator.data import properties as p
from curator.data.variational_data import canonical_boundary_hash
from curator.layer import VariationalEnergyOutput
from curator.model.features import ScalarVectorFeatureSpec
from curator.train import VariationalJointLoss
from curator.variational.spec import (
    BoundarySpec, ConstraintSpec, ControlSpec, GroupSpec, VariableSpec,
)


class QuadraticEnergy(nn.Module):
    def __init__(self):
        super().__init__()
        self.hardness = nn.Parameter(torch.tensor(0.8, dtype=torch.float64))

    def forward(self, context, state):
        return (0.5 * self.hardness * state.square() + 0.3 * context["scalar"] * state).sum()


class CoordinateProposal(nn.Module):
    def __init__(self):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(0.35, dtype=torch.float64))

    def forward(self, context):
        return self.slope * context["scalar"] + 0.6 * context["network_field"] + context["origin"]


def head(**kwargs):
    result = VariationalEnergyOutput(
        internal_energy=QuadraticEnergy(), proposal=CoordinateProposal(),
        state_unit="e", conjugate_unit="V", **kwargs,
    ).double()
    result.bind(ScalarVectorFeatureSpec("scalar", "", 1, 0), cutoff=5.0)
    return result


def batch(kind="potential", positions=None, count=1):
    if positions is None:
        positions = torch.tensor([[0.2, 0.0, 0.0], [1.1, 0.1, -0.2]], dtype=torch.float64)
        positions = torch.cat([positions + 0.3 * i for i in range(count)])
    positions = positions.clone().requires_grad_(True)
    variable = VariableSpec("charge", "scalar", 1, "e", "V")
    kwargs = ({"controls": (ControlSpec.group_potential("potential", "all", 0.4, unit="V"),)}
              if kind == "potential" else
              {"equalities": (ConstraintSpec.group_sum("charge", "all", 0.7, unit="e"),)})
    boundary = asdict(BoundarySpec(variable, groups=(GroupSpec.all(),), **kwargs))
    scalar = positions[:, :1].square()
    return {
        p.positions: positions, p.n_atoms: torch.full((count,), 2, dtype=torch.long),
        p.energy: 0.2 * scalar.reshape(count, 2).sum(-1), "scalar": scalar,
        p.edge_idx: torch.empty((0, 2), dtype=torch.long),
        p.edge_dist: positions.new_empty((0,)), p.edge_diff: positions.new_empty((0, 3)),
        "variational": {
            "condition_geometry_index": torch.arange(count),
            "condition_ptr": torch.arange(count + 1), "condition_weight": torch.ones(count),
            "condition_state_ptr": 2 * torch.arange(count + 1),
            "boundary_spec": (boundary,) * count,
            "boundary_hash": (canonical_boundary_hash(boundary),) * count, "labels": {},
        },
    }


def predict(output, data):
    return output(data)["variational_prediction"]


@pytest.mark.parametrize("kind", ["charge", "potential"])
def test_internal_labels_change_neither_state_force_nor_stationarity(kind):
    thermodynamic = head()
    internal = head(energy_convention="internal")
    data = batch(kind)
    predicted = predict(internal, data)
    reference = predict(thermodynamic, batch(kind))
    state = predicted["labels"]["variable_value"]["values"]
    scalar = data["scalar"].reshape(-1)
    expected_u = 0.2 * scalar.sum() + (0.4 * state.square() + 0.3 * scalar * state).sum()
    torch.testing.assert_close(predicted["labels"]["condition_energy"]["values"].sum(), expected_u)
    work = 0.4 * state.sum() if kind == "potential" else state.sum() * 0.0
    torch.testing.assert_close(reference["labels"]["condition_energy"]["values"].sum(), expected_u - work)
    for name in ("condition_forces", "variable_value"):
        torch.testing.assert_close(predicted["labels"][name]["values"], reference["labels"][name]["values"], atol=0, rtol=0)
    expected_residual = 0.8 * state + 0.3 * scalar
    expected_residual = (expected_residual - 0.4 if kind == "potential"
                         else expected_residual - expected_residual.mean())
    torch.testing.assert_close(predicted["residuals"]["stationarity"]["values"], expected_residual)
    assert predicted["metadata"]["energy_convention"] == "internal"
    assert predicted["metadata"]["force_potential"] == "thermodynamic"


def test_fixed_potential_internal_labels_use_grand_potential_forces():
    output = head(energy_convention="internal").eval()
    data = batch()
    positions = data[p.positions].detach()
    result = predict(output, data)
    direction = torch.tensor([[0.2, -0.1, 0.3], [-0.3, 0.1, 0.2]], dtype=torch.float64)
    step = 1e-5
    adjacent = [predict(output, batch(positions=positions + sign * step * direction)) for sign in (-1, 1)]
    internal = [value["labels"]["condition_energy"]["values"].sum() for value in adjacent]
    potentials = [energy - 0.4 * value["labels"]["variable_value"]["values"].sum()
                  for energy, value in zip(internal, adjacent)]
    force = result["labels"]["condition_forces"]["values"].reshape(-1, 3)
    projected_force = -(force * direction).sum()
    torch.testing.assert_close(projected_force, (potentials[1] - potentials[0]) / (2 * step), atol=1e-9, rtol=1e-7)
    assert abs(float((projected_force - (internal[1] - internal[0]) / (2 * step)).detach())) > 1e-3


def test_converged_reference_preserves_internal_labels_and_envelope_forces():
    output = head(energy_convention="internal", state_mode="converged").eval()
    data = batch()
    result = predict(output, data)
    positions, scalar = data[p.positions], data["scalar"].reshape(-1)
    stationary = (0.4 - 0.3 * scalar) / 0.8
    torch.testing.assert_close(result["labels"]["variable_value"]["values"], stationary)
    internal = 0.2 * scalar.sum() + (0.4 * stationary.square() + 0.3 * scalar * stationary).sum()
    torch.testing.assert_close(result["labels"]["condition_energy"]["values"].sum(), internal)
    expected_force = torch.zeros_like(positions)
    expected_force[:, 0] = -2 * positions[:, 0] * (0.2 + 0.3 * stationary)
    torch.testing.assert_close(result["labels"]["condition_forces"]["values"].reshape(-1, 3), expected_force)
    assert result["metadata"]["force_derivative"] == "stationary_envelope"


def test_batched_deployment_forces_do_not_differentiate_internal_energy_labels():
    output = head(energy_convention="internal", batch_observables=True).eval()
    reference = predict(output, batch(count=2))
    output.configure_deployment(compute_residuals=False, batch_forces=True)
    actual = predict(output, batch(count=2))
    for name in ("condition_energy", "condition_forces", "variable_value"):
        torch.testing.assert_close(actual["labels"][name]["values"], reference["labels"][name]["values"])
    assert actual["metadata"]["batched_force_derivative"]


@pytest.mark.parametrize("kind", ["charge", "potential"])
def test_same_joint_loss_and_residual_gradient_policy_for_both_boundaries(kind):
    output = head(energy_convention="internal")
    data = batch(kind)
    result = output(data)
    criterion = VariationalJointLoss({}, {}, residual_weights={"stationarity": 1.0})
    criterion(result, data)["loss"].backward()
    assert output.proposal.slope.grad is not None and output.proposal.slope.grad.abs() > 0
    assert output.internal_energy.hardness.grad is None


def test_dataset_configuration_checkpoint_contract_and_legacy_pickle():
    default = head()
    state = copy.deepcopy(default.state_dict())
    assert "energy_convention" not in state["_extra_state"]["contract"]
    explicit = head(energy_convention="thermodynamic")
    explicit.load_state_dict(state, strict=True)
    configured = head()
    configured.setup_from_datamodule(SimpleNamespace(metadata={
        "energy_convention": "internal", "force_potential": "thermodynamic"}))
    assert configured.energy_convention == "internal"
    assert configured.get_extra_state()["contract"]["energy_convention"] == "internal"
    with pytest.raises(ValueError, match="contract differs"):
        explicit.load_state_dict(configured.state_dict(), strict=True)
    restored = head(energy_convention="internal")
    restored.load_state_dict(configured.state_dict(), strict=True)
    assert restored._dataset_metadata == configured._dataset_metadata
    with pytest.raises(ValueError, match="dataset metadata differs"):
        restored.setup_from_datamodule(SimpleNamespace(metadata={"energy_convention": "thermodynamic"}))

    del default.energy_convention
    stream = io.BytesIO()
    torch.save(default, stream)
    stream.seek(0)
    legacy = torch.load(stream, weights_only=False)
    assert legacy.energy_convention == "thermodynamic"
    legacy.load_state_dict(state, strict=True)
    actual, expected = predict(legacy, batch()), predict(explicit, batch())
    for name in ("condition_energy", "condition_forces", "variable_value"):
        torch.testing.assert_close(actual["labels"][name]["values"], expected["labels"][name]["values"], atol=0, rtol=0)


@pytest.mark.parametrize("convention", [None, "source_total_energy", "wrong"])
def test_invalid_convention_rejected(convention):
    with pytest.raises(ValueError, match="energy_convention"):
        head(energy_convention=convention)
