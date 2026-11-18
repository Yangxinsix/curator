"""Condition-local controls, residual-only augmentation and thermodynamic AD."""

from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from curator.data import AtomsData, collate_atoms_data, properties
from curator.data.public_response import PublicResponseDataModule
from curator.data.variational_data import (
    VariationalCondition, VariationalData, canonical_boundary_hash, condition_input,
)
from curator.train.variational_losses import VariationalJointLoss
from curator.variational.observables import (
    CHARGE_BOUNDARY_OBSERVABLES, BatchedEnergyObservables, EnergyObservables, ObservableSpec,
)


DTYPE = torch.float64


@pytest.fixture
def public_module(tmp_path, monkeypatch):
    import curator.data.public_response as public

    root = tmp_path / "field_schnet" / "ethanol_methanol_qmmm"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text('{"files": []}')
    records = []
    for index, magnitude in enumerate((1.0, 3.0, 100.0, 200.0)):
        record = public._record(
            [1, 1], [[0, 0, 0], [index + 1, 0, 0]], np.zeros((3, 3)), False,
            {"condition_energy": [float(index)], "dipole": [index, 0, 0]}, {}, str(index),
        )
        record["local_external_field"] = np.full((2, 3), magnitude)
        records.append(record)
    monkeypatch.setitem(public._READERS, "sqlite", lambda paths, dataset: (
        records, {"field_semantics": "fixed_per_atom_field", "disabled_labels": {}}))
    monkeypatch.setattr(public, "_make_splits", lambda records, dataset, seed: (
        {"train": [0, 1], "val": [2], "test": [3]}, {"seed": seed}))
    module = PublicResponseDataModule(
        "field_schnet/ethanol_methanol_qmmm", root=tmp_path, batch_size=2,
        local_field_conditions=2, local_field_std_ratio=0.1, local_field_seed=29,
    )
    module.setup()
    return module


def test_augmentation_uses_training_statistics_and_resumes_exactly(public_module):
    module = public_module
    metadata = module.metadata["local_field_augmentation"]
    assert metadata["training_rms"] == pytest.approx(np.sqrt(5))
    assert metadata["fit_indices"] == [0, 1]
    assert metadata["conditions_per_geometry"] == 3
    first = module.train_dataset[0]
    second = module.train_dataset[1]
    assert len(first.variational.conditions) == 3
    assert len(module.val_dataset[0].variational.conditions) == 1
    assert len(module.test_dataset[0].variational.conditions) == 1
    for condition in first.variational.conditions[1:]:
        assert condition.labels == {} and condition.responses == ()
        assert condition.label_eligible == {"condition_energy": False, "dipole": False}
    assert first.variational.conditions[0].labels.keys() == {"condition_energy", "dipole"}

    batch = collate_atoms_data([first, second])["variational"]
    assert batch["condition_ptr"].tolist() == [0, 3, 6]
    assert batch["labels"]["dipole"]["present"].tolist() == [True, False, False] * 2
    assert batch["labels"]["dipole"]["eligible"].tolist() == [True, False, False] * 2
    assert condition_input(batch, "local_external_field", 0) is None
    augmented = condition_input(batch, "local_external_field", 1)
    assert not torch.equal(augmented, first.atoms["local_external_field"])
    assert not torch.equal(augmented, condition_input(batch, "local_external_field", 2))
    state = module.state_dict()
    restored = PublicResponseDataModule(
        module.dataset_id, root=module.root, batch_size=2,
        local_field_conditions=2, local_field_std_ratio=0.1, local_field_seed=29,
    )
    restored.load_state_dict(state)
    torch.testing.assert_close(torch.rand(4, generator=module.generator),
                               torch.rand(4, generator=restored.generator), rtol=0, atol=0)
    replay = collate_atoms_data([restored.train_dataset[0]])["variational"]
    torch.testing.assert_close(condition_input(replay, "local_external_field", 1), augmented, rtol=0, atol=0)
    changed = PublicResponseDataModule(
        module.dataset_id, root=module.root, local_field_conditions=2, local_field_seed=30,
    )
    with pytest.raises(ValueError, match="differ from checkpoint"):
        changed.load_state_dict(state)
    module.train_dataset[1]  # Access order cannot alter the local sampling stream.
    replay = collate_atoms_data([module.train_dataset[0]])["variational"]
    torch.testing.assert_close(condition_input(replay, "local_external_field", 1), augmented, rtol=0, atol=0)


def test_disabled_augmentation_preserves_legacy_metadata(public_module):
    module = PublicResponseDataModule(public_module.dataset_id, root=public_module.root)
    module.setup()
    assert "local_field_augmentation" not in module.metadata
    assert len(module.train_dataset[0].variational.conditions) == 1
    assert module.train_dataset[0].variational.conditions[0].inputs == {}


@pytest.mark.parametrize("options", [
    {"local_field_conditions": -1}, {"local_field_conditions": True},
    {"local_field_std_ratio": 0}, {"local_field_std_ratio": float("nan")},
    {"local_field_conditions": 1, "state_kind": "scalar"},
])
def test_invalid_augmentation_is_rejected(options):
    with pytest.raises(ValueError):
        PublicResponseDataModule("field_schnet/ethanol_methanol_qmmm", **options)


def test_ragged_inputs_keep_live_graph_and_move_dtype(public_module):
    sample = public_module.train_dataset[0]
    control = torch.tensor([0.3, -0.3], dtype=DTYPE, requires_grad=True)
    target = torch.tensor([0.2], dtype=DTYPE, requires_grad=True)
    conditions = list(sample.variational.conditions)
    conditions[0] = replace(conditions[0], inputs={"control_value": control, "constraint_target": target})
    sample.variational = replace(sample.variational, conditions=tuple(conditions))
    moved = sample.to(dtype=torch.float32)
    assert moved.variational.conditions[0].inputs["control_value"].dtype == torch.float32
    batch = collate_atoms_data([sample])["variational"]
    u = condition_input(batch, "control_value", 0)
    b = condition_input(batch, "constraint_target", 0)
    energy = u.square().sum() + b.pow(3).sum()
    du, db = torch.autograd.grad(energy, (control, target))
    torch.testing.assert_close(du, 2 * control)
    torch.testing.assert_close(db, 3 * target.square())
    assert condition_input(batch, "control_value", 1) is None
    assert condition_input({}, "control_value", 0) is None


def test_residual_only_conditions_preserve_geometry_balance_and_label_checks(public_module):
    first, second = public_module.train_dataset[0], public_module.train_dataset[1]
    second.variational = replace(second.variational, conditions=second.variational.conditions[:1])
    batch = collate_atoms_data([first, second])
    residual = torch.arange(1.0, 5.0, dtype=DTYPE).repeat_interleave(6).requires_grad_()
    energy = torch.tensor([first.variational.conditions[0].labels["condition_energy"].item(),
                           1e6, -1e6, second.variational.conditions[0].labels["condition_energy"].item()],
                          dtype=DTYPE, requires_grad=True)
    prediction = {"variational_prediction": {"value_space": "physical", "labels": {
        "condition_energy": {"values": energy, "ptr": torch.arange(5), "shape": ((1,),) * 4, "unit": "eV"},
    }, "residuals": {"stationarity": {
        "values": residual, "ptr": batch["variational"]["condition_state_ptr"],
        "source": "variational_energy_gradient", "gradient_policy": "proposal_only",
        "normalization": {"scale": 1.0},
    }}}}
    criterion = VariationalJointLoss(
        {"condition_energy": 1}, {"condition_energy": 1}, required_labels=("condition_energy",),
        residual_weights={"stationarity": 1},
    )
    result = criterion(prediction, batch)
    assert result["terms"]["condition_energy"]["unweighted"] == 0
    assert result["loss"].item() == pytest.approx(((1 + 4 + 9) / 3 + 16) / 2)
    result["loss"].backward()
    torch.testing.assert_close(energy.grad, torch.zeros_like(energy))
    batch["variational"]["labels"]["condition_energy"]["present"][0] = False
    with pytest.raises(ValueError, match="absent for an eligible condition"):
        criterion(prediction, batch)


def test_conp_derivative_includes_nonstationary_proposal_path():
    u = torch.tensor([0.4], dtype=DTYPE, requires_grad=True)
    weight = torch.tensor(0.2, dtype=DTYPE, requires_grad=True)
    direction = torch.tensor([1.0, -1.0], dtype=DTYPE)
    state = weight * u * direction
    energy = state[0].square() + 2 * state[1].square() - u[0] * (direction @ state)
    result = EnergyObservables(energy, {"control_value": u}, create_graph=True).evaluate([
        ObservableSpec("conjugate_charge", ("control_value",), factor=-1),
    ])["conjugate_charge"]
    expected = (4 * weight - 6 * weight.square()) * u
    torch.testing.assert_close(result, expected)
    assert not torch.allclose(result, (direction @ state).reshape(1))
    actual_gradient = torch.autograd.grad(result.sum(), weight, retain_graph=True)[0]
    torch.testing.assert_close(actual_gradient, torch.autograd.grad(expected.sum(), weight)[0])


def test_conq_derivative_includes_target_projection_and_tangent_proposal_path():
    target = torch.tensor([0.4], dtype=DTYPE, requires_grad=True)
    weight = torch.tensor(0.2, dtype=DTYPE, requires_grad=True)
    state = torch.cat((target / 2 + weight * target.square(), target / 2 - weight * target.square()))
    energy = state[0].square() + 2 * state[1].square()
    derivative = EnergyObservables(energy, {"constraint_target": target}, create_graph=True).derivative(("constraint_target",))
    gradient = torch.stack((2 * state[0], 4 * state[1]))
    expected = (gradient.mean() + (gradient[0] - gradient[1]) * 2 * weight * target)
    torch.testing.assert_close(derivative, expected)
    assert not torch.allclose(derivative, gradient.mean().reshape(1))
    derivative.square().sum().backward()
    assert weight.grad is not None and weight.grad.abs() > 0


@pytest.mark.parametrize("batched", [False, True])
def test_absent_control_has_empty_derivatives_in_mixed_boundary_schema(batched):
    empty = torch.empty(0, dtype=DTYPE, requires_grad=True)
    energy = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    specs = [ObservableSpec("first", ("control_value",)),
             ObservableSpec("second", ("control_value", "control_value"))]
    if batched:
        result = BatchedEnergyObservables([energy], {"control_value": [empty]}, create_graph=True).evaluate(specs)
        result = {key: value[0] for key, value in result.items()}
    else:
        result = EnergyObservables(energy, {"control_value": empty}, create_graph=True).evaluate(specs)
    assert result["first"].shape == (0,)
    assert result["second"].shape == (0, 0)


def _model(kind, observables=()):
    from curator.layer import PairwiseDistance, VariationalEnergyOutput
    from curator.layer._differentiable_inputs import DifferentiableInputs
    from curator.model import NeuralNetworkPotential, Painn

    torch.manual_seed(37)
    return NeuralNetworkPotential(
        Painn(1, 4, 5.0, num_basis=4),
        input_modules=[DifferentiableInputs(), PairwiseDistance(compute_distance_from_R=True)],
        output_modules=[VariationalEnergyOutput(
            state_kind=kind, components=3 if kind == "vector" else 1, hidden_features=4,
            state_unit="e*Angstrom" if kind == "vector" else "e",
            conjugate_unit="V/Angstrom" if kind == "vector" else "V",
            compute_forces=False, observables=observables,
        )],
    ).double()


def _model_batch(samples):
    batch = collate_atoms_data(samples)
    edges, offset = [], 0
    for sample in samples:
        count = int(sample.atoms[properties.n_atoms].item())
        edges.extend((offset + i, offset + j) for i in range(count) for j in range(count) if i != j)
        offset += count
    batch[properties.edge_idx] = torch.tensor(edges, dtype=torch.long)
    return batch


def test_output_uses_condition_local_field_without_mixing_conditions(public_module):
    model = _model("vector").eval()
    sample = public_module.train_dataset[0]
    joint = model(_model_batch([sample]))["variational_prediction"]["labels"]
    for index, condition in enumerate(sample.variational.conditions):
        isolated = replace(sample, variational=replace(sample.variational, conditions=(condition,)))
        single = model(_model_batch([isolated]))["variational_prediction"]["labels"]
        for name in ("condition_energy", "variable_value"):
            start, stop = joint[name]["ptr"][index:index + 2]
            torch.testing.assert_close(joint[name]["values"][start:stop], single[name]["values"], rtol=0, atol=0)
    first, second = joint["variable_value"]["values"].reshape(3, -1)[:2]
    assert not torch.allclose(first, second)


def _charge_sample(mode, value):
    from curator.variational.spec import BoundarySpec, ConstraintSpec, ControlSpec, GaugeSpec, GroupSpec, VariableSpec

    variable = VariableSpec("charge", "scalar", 1, "e", "V")
    groups = (GroupSpec.from_indices("left", [0, 1]), GroupSpec.from_indices("right", [2, 3]), GroupSpec.all())
    if mode == "conq":
        equalities = tuple(ConstraintSpec.group_sum(name, name, target, unit="e")
                           for name, target in zip(("left", "right"), (0.4, -0.4)))
        controls = ()
        inputs = {"constraint_target": value}
    else:
        equalities = (ConstraintSpec.group_sum("neutral", "all", 0.0, unit="e"),)
        controls = tuple(ControlSpec.group_potential(name, name, potential, unit="V")
                         for name, potential in zip(("left", "right"), (0.4, -0.4)))
        inputs = {"control_value": value}
    boundary = asdict(BoundarySpec(variable, groups=groups, equalities=equalities,
                                  controls=controls, gauge=GaugeSpec(mode="allow")))
    condition = VariationalCondition(
        mode, None, None, None, boundary, canonical_boundary_hash(boundary), {}, {}, {}, inputs=inputs,
    )
    payload = VariationalData(
        "1", "four-sites", "test-geometry", asdict(variable), torch.tensor([0, 2, 4, 8]),
        torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]), ("left", "right", "all"),
        ("electrode", "electrode", "system"), (condition,),
    )
    atoms = {
        properties.atomic_numbers: torch.tensor([1, 6, 8, 1]),
        properties.positions: torch.tensor([[0., 0., 0.], [1., .1, 0.], [0., 2., .2], [1., 2., 0.]], dtype=DTYPE),
        properties.n_atoms: torch.tensor([4]), properties.cell: torch.zeros((3, 3), dtype=DTYPE),
        properties.pbc: torch.zeros((1, 3), dtype=torch.bool),
    }
    return AtomsData(atoms, {}, "boundary_derivative", variational=payload)


@pytest.mark.parametrize("boundary_mode", ["conq", "conp"])
def test_joint_output_boundary_derivatives_match_full_energy_finite_difference(boundary_mode):
    model = _model("scalar", CHARGE_BOUNDARY_OBSERVABLES)
    values = torch.tensor([0.4, -0.4], dtype=DTYPE, requires_grad=True)
    prediction = model(_model_batch([_charge_sample(boundary_mode, values)]))["variational_prediction"]
    name = "constraint_derivative" if boundary_mode == "conq" else "control_derivative"
    derivative = prediction["labels"][name]["values"]
    energy = prediction["labels"]["condition_energy"]["values"].sum()
    torch.testing.assert_close(derivative, torch.autograd.grad(energy, values, retain_graph=True)[0])
    direction = torch.tensor([1.0, -1.0], dtype=DTYPE)
    eps = 1e-5
    neighbors = [model(_model_batch([_charge_sample(boundary_mode, values.detach() + sign * eps * direction)]))
                 ["variational_prediction"]["labels"]["condition_energy"]["values"].sum() for sign in (-1, 1)]
    torch.testing.assert_close(derivative @ direction, (neighbors[1] - neighbors[0]) / (2 * eps), rtol=2e-5, atol=1e-8)
    derivative.square().sum().backward()
    proposal = next(group for group in model.parameter_groups() if group.name == "proposal")
    assert any(parameter.grad is not None and bool(parameter.grad.abs().max() > 0) for parameter in proposal.params)
