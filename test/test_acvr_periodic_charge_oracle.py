"""Physical Ewald, boundary labels and dataset separation for the charge task."""

import copy
from dataclasses import replace

import pytest
import torch

from curator.data import properties
from curator.data.collate_atoms_data import collate_atoms_data
from curator.data.periodic_charge_oracle import (
    PeriodicChargeOracleDataModule, PeriodicChargeOracleDataset,
    periodic_charge_coefficients, periodic_charge_reference,
)
from curator.layer._charge_internal_energy import EwaldChargeEnergy
from curator.layer._ewald import EwaldSummation
from curator.layer._variational_output import _boundary_from_dict
from curator.variational.backends.ewald import EwaldChargeOperator
from curator.variational.compiler import compile_boundary
from curator.variational.spec import VariableLayout


@pytest.fixture(scope="module")
def dataset():
    return PeriodicChargeOracleDataset(num_train=2, num_val=1, num_test=1, plane_side=2)


def test_geometry_ewald_ignores_incomplete_gnn_edges_and_checks_periodicity(dataset):
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    data = dataset[0].atoms.copy()
    ewald = EwaldSummation(acc_factor=8)
    operator = EwaldChargeOperator(ewald)
    actual = operator.geometry_kernels(data)[0]
    atoms = Atoms(numbers=data[properties.atomic_numbers].numpy(),
                  positions=data[properties.positions].numpy(),
                  cell=data[properties.cell].numpy(), pbc=True)
    center, neighbor, shifts = neighbor_list("ijS", atoms, ewald.cutoff, self_interaction=False)
    edges = torch.stack((torch.as_tensor(center), torch.as_tensor(neighbor)), dim=1).long()
    difference = (data[properties.positions][edges[:, 1]] - data[properties.positions][edges[:, 0]]
                  + torch.as_tensor(shifts, dtype=torch.float64) @ data[properties.cell])
    complete = {**data, properties.edge_idx: edges,
                properties.edge_dist: torch.linalg.vector_norm(difference, dim=-1)}
    torch.testing.assert_close(actual, operator.kernels(complete)[0], atol=1e-12, rtol=1e-12)
    assert not torch.allclose(actual, operator.kernels(data)[0], atol=1e-10, rtol=1e-10)
    data[properties.pbc] = torch.tensor([[True, True, False]])
    with pytest.raises(ValueError, match="3D periodicity"):
        operator.geometry_kernels(data)


def test_ewald_geometry_has_live_coordinate_cell_gradients_and_no_stale_cache(dataset):
    data = dataset[0].atoms.copy()
    operator = EwaldChargeOperator(EwaldSummation(acc_factor=8))
    data[properties.positions] = data[properties.positions].clone().requires_grad_()
    data[properties.cell] = data[properties.cell].clone().requires_grad_()
    charge = dataset[0].variational.conditions[0].labels["variable_value"].flatten()
    kernel = operator.geometry_kernels(data)[0]
    energy = 0.5 * charge @ kernel @ charge
    position_gradient, cell_gradient = torch.autograd.grad(
        energy, (data[properties.positions], data[properties.cell]), create_graph=True)
    generator = torch.Generator().manual_seed(41)
    for key, gradient in ((properties.positions, position_gradient), (properties.cell, cell_gradient)):
        direction = torch.randn(data[key].shape, generator=generator, dtype=torch.float64)
        epsilon = 1e-5
        energies = []
        for sign in (-1, 1):
            moved = {**data, key: data[key].detach() + sign * epsilon * direction}
            updated = operator.geometry_kernels(moved)[0]
            assert not torch.equal(kernel, updated)
            energies.append(0.5 * charge @ updated @ charge)
        torch.testing.assert_close((gradient * direction).sum(),
                                   (energies[1] - energies[0]) / (2 * epsilon), rtol=3e-5, atol=1e-9)
    # One atom may be represented by another periodic image.
    shifted = data[properties.positions].detach().clone()
    shifted[1] += data[properties.cell][0].detach()
    torch.testing.assert_close(operator.geometry_kernels({**data, properties.positions: shifted})[0],
                               kernel, atol=1e-11, rtol=1e-11)
    # Higher derivatives needed by response training remain available.
    second = torch.autograd.grad(position_gradient.square().sum(), data[properties.positions])[0]
    assert torch.isfinite(second).all() and second.abs().max() > 0


def test_ewald_charge_energy_intrinsic_coefficients_and_residual_detachment(dataset):
    torch.manual_seed(13)
    energy = EwaldChargeEnergy(scalar_channels=4, hidden_features=6).double()
    scalar = torch.randn((8, 4), dtype=torch.float64, requires_grad=True)
    context = {"scalar": scalar, **energy.geometry_context(dataset[0].atoms)[0]}
    state = torch.randn((8, 1), dtype=torch.float64, requires_grad=True)
    state = state - state.mean()
    result = energy(context, state)
    coefficients = energy.coefficients(context)
    from curator.layer._ewald import apply_ewald
    expected_gradient = (coefficients["electronegativity"] + coefficients["hardness"] * state
                         + apply_ewald(context, state))
    torch.testing.assert_close(torch.autograd.grad(result, state, create_graph=True)[0], expected_gradient)
    changed_boundary = {**context, "field": torch.ones_like(state), "origin": state * 3,
                        "network_field": state * 2, "constraint_target": state.new_tensor([4.])}
    torch.testing.assert_close(energy(changed_boundary, state), result)
    result.backward(retain_graph=True)
    assert scalar.grad.abs().max() > 0
    assert any(parameter.grad is not None for parameter in energy.parameters())
    energy.zero_grad(set_to_none=True)
    scalar.grad = None
    detached = {key: value.detach() for key, value in context.items()}
    detached.update(coefficients)  # Deliberately supply live, unused coefficients.
    weights = {name: parameter.detach() for name, parameter in energy.named_parameters()}
    value = torch.func.functional_call(energy, weights, (detached, state))
    residual = torch.autograd.grad(value, state, create_graph=True)[0]
    residual.square().mean().backward()
    assert scalar.grad is None
    assert all(parameter.grad is None for parameter in energy.parameters())
    restored = EwaldChargeEnergy(scalar_channels=4, hidden_features=6).double()
    restored.load_state_dict(copy.deepcopy(energy.state_dict()))
    with pytest.raises(ValueError, match="boundary convention"):
        EwaldChargeEnergy(scalar_channels=4, hidden_features=6,
                          ewald=EwaldSummation(alpha=0.3)).double().load_state_dict(energy.state_dict())


def test_periodic_oracle_exact_states_derivative_labels_and_remote_response(dataset):
    for sample in dataset:
        sample.variational.validate(8)
        atoms = sample.atoms
        chi, hardness, constant = periodic_charge_coefficients(atoms)
        kernel = EwaldChargeOperator(EwaldSummation(acc_factor=8)).geometry_kernels(atoms)[0]
        hessian = kernel + torch.diag(hardness)
        for condition in sample.variational.conditions:
            boundary = _boundary_from_dict(condition.boundary_spec)
            compiled = compile_boundary(boundary, VariableLayout(boundary.variable, 8))
            constraint, target = compiled.constraint_matrix, condition.inputs["constraint_target"]
            drive = compiled.control_matrix.T @ compiled.control_value
            drive = drive + condition.inputs["local_external_field"].flatten()
            charge = condition.labels["variable_value"].flatten()
            torch.testing.assert_close(constraint @ charge, target, atol=1e-13, rtol=1e-13)
            gradient = chi + hessian @ charge - drive
            multiplier = -condition.labels["constraint_derivative"]
            torch.testing.assert_close(gradient + constraint.T @ multiplier,
                                       torch.zeros(8, dtype=torch.float64), atol=1e-13, rtol=0)
            torch.testing.assert_close(constant + (chi - drive) @ charge + 0.5 * charge @ hessian @ charge,
                                       condition.labels["condition_energy"].squeeze())
            # Independent envelope signs against finite differences of the exact energy.
            for index in range(target.numel()):
                delta = torch.zeros_like(target)
                delta[index] = 1e-5
                energies = [periodic_charge_reference(hessian, chi, constant, constraint,
                                                      target + sign * delta, drive)[2]
                            for sign in (-1, 1)]
                torch.testing.assert_close((energies[1] - energies[0]) / 2e-5,
                                           condition.labels["constraint_derivative"][index],
                                           rtol=1e-6, atol=1e-9)
            if condition.provenance["boundary_kind"] == "conp":
                torch.testing.assert_close(condition.labels["control_derivative"],
                                           -compiled.control_matrix @ charge)
                for index in range(compiled.control_value.numel()):
                    delta = 1e-5 * compiled.control_matrix[index]
                    energies = [periodic_charge_reference(hessian, chi, constant, constraint,
                                                          target, drive + sign * delta)[2]
                                for sign in (-1, 1)]
                    torch.testing.assert_close((energies[1] - energies[0]) / 2e-5,
                                               condition.labels["control_derivative"][index],
                                               rtol=1e-6, atol=1e-9)
        for first in (0, 3):
            baseline = sample.variational.conditions[first]
            perturbed = sample.variational.conditions[first + 2]
            torch.testing.assert_close(baseline.inputs["local_external_field"][:4],
                                       perturbed.inputs["local_external_field"][:4])
            response = perturbed.labels["variable_value"][:4] - baseline.labels["variable_value"][:4]
            assert response.abs().max() > 1e-8


def test_periodic_oracle_splits_batch_isolation_and_shuffle_resume():
    kwargs = dict(num_train=2, num_val=1, num_test=1, plane_side=2, batch_size=2)
    data = PeriodicChargeOracleDataModule(**kwargs)
    data.setup()
    splits = [set(dataset.indices) for dataset in data.datasets.values()]
    assert sum(map(len, splits)) == len(set.union(*splits)) == 4
    rebuilt = PeriodicChargeOracleDataModule(**kwargs)
    rebuilt.setup()
    assert data.metadata == rebuilt.metadata
    for split in data.datasets:
        assert data.datasets[split].metadata == data.metadata
        for first, second in zip(data.datasets[split], rebuilt.datasets[split]):
            assert first.variational.geometry_hash == second.variational.geometry_hash
            for left, right in zip(first.variational.conditions, second.variational.conditions):
                assert left.boundary_hash == right.boundary_hash
                torch.testing.assert_close(left.labels["variable_value"], right.labels["variable_value"],
                                           rtol=0, atol=0)
    batch = next(iter(data.train_dataloader()))
    packed = batch["variational"]
    assert packed["condition_geometry_index"].tolist() == [0] * 6 + [1] * 6
    assert packed["labels"]["control_derivative"]["present"].tolist() == [False] * 3 + [True] * 3 + [False] * 3 + [True] * 3
    state = data.state_dict()
    list(data.val_dataloader())
    assert torch.equal(state["generator_state"], data.generator.get_state())
    rebuilt.load_state_dict(state)
    assert next(iter(data.train_dataloader()))["meta"] == next(iter(rebuilt.train_dataloader()))["meta"]
    operator = EwaldChargeOperator(EwaldSummation(acc_factor=8))
    samples = [data.datasets["train"][0], data.datasets["val"][0]]
    batched = operator.geometry_kernels(collate_atoms_data(samples))
    for sample, actual in zip(samples, batched):
        torch.testing.assert_close(actual, operator.geometry_kernels(sample.atoms)[0])


def test_joint_ewald_head_routes_gradients_and_differentiates_live_boundaries(dataset, monkeypatch):
    from curator.layer import PairwiseDistance, VariationalEnergyOutput
    from curator.layer._variational_proposal import BoundaryConditionedProposal
    from curator.model import NeuralNetworkPotential, Painn
    from curator.train import VariationalJointLoss
    from curator.variational.observables import CHARGE_BOUNDARY_OBSERVABLES
    import curator.layer._variational_output as output_module

    torch.manual_seed(17)
    model = NeuralNetworkPotential(
        Painn(2, 8, 3.5, num_basis=4),
        input_modules=[PairwiseDistance(compute_distance_from_R=True)],
        output_modules=[VariationalEnergyOutput(
            state_kind="scalar", components=1, state_unit="e", conjugate_unit="V",
            compute_forces=True, retain_response_graph=True, observables=CHARGE_BOUNDARY_OBSERVABLES,
            internal_energy=EwaldChargeEnergy(hidden_features=8, ewald=EwaldSummation(acc_factor=8)),
            proposal=BoundaryConditionedProposal(communication="ewald", hidden_features=8,
                                                   num_layers=2, num_heads=2),
        )],
    ).double()
    sample = dataset[0]
    conditions = (sample.variational.conditions[0], sample.variational.conditions[3])
    sample = replace(sample, variational=replace(sample.variational, conditions=conditions))

    def predict(item):
        return model(collate_atoms_data([item]))

    def forbidden(*args, **kwargs):
        raise AssertionError("predicted training must not invoke a reference state solve")

    monkeypatch.setattr(output_module, "solve_functional_reference", forbidden)
    prediction = predict(sample)["variational_prediction"]
    prediction["residuals"]["stationarity"]["values"].square().mean().backward()
    for group in model.parameter_groups():
        if group.name == "proposal":
            assert any(parameter.grad is not None and parameter.grad.abs().max() > 0
                       for parameter in group.params)
        else:
            assert all(parameter.grad is None for parameter in group.params)
    model.zero_grad(set_to_none=True)
    prediction = predict(sample)
    weights = {"condition_energy": 1., "control_derivative": 1., "constraint_derivative": 1.}
    loss = VariationalJointLoss(supervised_weights=weights, supervised_scales=weights,
                               required_labels=tuple(weights))
    loss(prediction, collate_atoms_data([sample]))["loss"].backward()
    for group in model.parameter_groups():
        assert any(parameter.grad is not None and parameter.grad.abs().max() > 0
                   for parameter in group.params), group.name

    labels = prediction["variational_prediction"]["labels"]
    epsilon = 1e-5
    # Compare the predicted thermodynamic potential, including its proposal
    # derivative. No equilibrium envelope shortcut is valid at this state.
    for condition_index, input_name, label_name, direction in (
        (0, "constraint_target", "constraint_derivative", torch.tensor([1., -1.], dtype=torch.float64)),
        (1, "control_value", "control_derivative", torch.tensor([1., -1.], dtype=torch.float64)),
    ):
        condition = conditions[condition_index]
        energies = []
        for sign in (-1, 1):
            changed = replace(condition, inputs={**condition.inputs,
                              input_name: condition.inputs[input_name] + sign * epsilon * direction})
            updated_conditions = list(conditions)
            updated_conditions[condition_index] = changed
            updated = replace(sample, variational=replace(sample.variational,
                                                         conditions=tuple(updated_conditions)))
            energies.append(predict(updated)["variational_prediction"]["labels"]
                            ["condition_energy"]["values"][condition_index])
        start, stop = labels[label_name]["ptr"][condition_index:condition_index + 2]
        derivative = labels[label_name]["values"][start:stop] @ direction
        torch.testing.assert_close(derivative, (energies[1] - energies[0]) / (2 * epsilon),
                                   atol=1e-7, rtol=3e-5)
    direction = torch.randn((8, 3), dtype=torch.float64)
    energies = []
    for sign in (-1, 1):
        updated = replace(sample, atoms={**sample.atoms, properties.positions:
                          sample.atoms[properties.positions] + sign * epsilon * direction})
        energies.append(predict(updated)["variational_prediction"]["labels"]["condition_energy"]["values"].sum())
    forces = labels["condition_forces"]["values"].reshape(2, 8, 3).sum(0)
    torch.testing.assert_close(-(forces * direction).sum(), (energies[1] - energies[0]) / (2 * epsilon),
                               atol=1e-7, rtol=3e-5)
