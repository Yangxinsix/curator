"""Joint GNN energy semantics, gradient ownership and coordinate response."""

import copy
from dataclasses import asdict

import pytest
import torch

from curator.data import properties
from curator.layer import PairwiseDistance, StableInternalEnergy, VariationalEnergyOutput
from curator.model import MACE, NeuralNetworkPotential, Painn
from curator.data.variational_data import canonical_boundary_hash
from curator.variational.spec import BoundarySpec, ConstraintSpec, GroupSpec, VariableSpec


def _batch(positions=None, *, field=None, charge=None):
    if positions is None:
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.2, 0.0],
                                  [0.1, 1.0, 0.3]], dtype=torch.float64)
    result = {
        properties.Z: torch.tensor([1, 6, 8]), properties.positions: positions,
        properties.edge_idx: torch.tensor([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]]),
        properties.n_atoms: torch.tensor([3]), properties.image_idx: torch.tensor([0, 0, 0]),
    }
    if field is not None:
        result["external_field"] = torch.as_tensor(field, dtype=positions.dtype).reshape(1, -1)
    if charge is not None:
        result[properties.total_charge] = torch.tensor([charge], dtype=positions.dtype)
    return result


def _model(kind="scalar", backbone="painn", **output_kwargs):
    torch.manual_seed(7)
    if backbone == "painn":
        representation = Painn(2, 8, 5.0, num_basis=4)
    else:
        representation = MACE(cutoff=5.0, num_interactions=2, correlation=2,
                              species=["H", "C", "O"], lmax=1, parity=True,
                              num_features=4, num_basis=4, avg_num_neighbors=2.0)
    return NeuralNetworkPotential(
        representation,
        input_modules=[PairwiseDistance(compute_distance_from_R=True)],
        output_modules=[VariationalEnergyOutput(state_kind=kind, components=3 if kind == "vector" else 1,
                                                hidden_features=8, **output_kwargs)],
    ).double()


def _prediction(model, batch):
    result = model(batch)
    assert set(result) == {"variational_prediction"}
    return result["variational_prediction"]


@pytest.mark.parametrize("kind", ["scalar", "vector"])
def test_physical_supervision_reaches_all_four_parameter_groups(kind):
    model = _model(kind)
    prediction = _prediction(model, _batch(field=[0.2] if kind == "scalar" else [0.2, -0.1, 0.3]))
    labels = prediction["labels"]
    loss = labels["condition_energy"]["values"].square().mean()
    loss = loss + labels["condition_forces"]["values"].square().mean()
    loss.backward()
    groups = model.parameter_groups()
    assert [group.name for group in groups] == ["backbone", "baseline", "functional", "proposal"]
    all_ids = [id(parameter) for group in groups for parameter in group.params]
    assert len(all_ids) == len(set(all_ids)) == len(list(model.parameters()))
    for group in groups:
        gradients = [parameter.grad for parameter in group.params if parameter.grad is not None]
        assert gradients, group.name
        assert sum(float(value.square().sum()) for value in gradients) > 0, group.name


@pytest.mark.parametrize("kind", ["scalar", "vector"])
def test_residual_backward_and_adamw_update_only_proposal(kind):
    model = _model(kind)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    prediction = _prediction(model, _batch())
    prediction["residuals"]["stationarity"]["values"].square().mean().backward()
    for group in model.parameter_groups():
        if group.name == "proposal":
            assert any(parameter.grad is not None and bool(parameter.grad.abs().max() > 0)
                       for parameter in group.params)
        else:
            assert all(parameter.grad is None for parameter in group.params)
    optimizer.step()
    changed = [name for name, parameter in model.named_parameters()
               if not torch.equal(parameter, before[name])]
    assert changed
    assert all(name.startswith("output_modules.0.proposal.") for name in changed)


@pytest.mark.parametrize("kind", ["scalar", "vector"])
def test_forces_differentiate_the_predicted_state_coordinate_path(kind):
    model = _model(kind).eval()
    positions = _batch()[properties.positions]
    field = [0.3] if kind == "scalar" else [0.2, -0.1, 0.3]
    prediction = _prediction(model, _batch(positions.clone(), field=field))
    force = prediction["labels"]["condition_forces"]["values"].reshape(-1, 3)
    direction = torch.tensor([[0.1, -0.2, 0.3], [-0.3, 0.2, 0.1],
                              [0.2, 0.1, -0.2]], dtype=torch.float64)
    eps = 1.0e-5
    energies = [_prediction(model, _batch(positions + sign * eps * direction, field=field))
                ["labels"]["condition_energy"]["values"].sum() for sign in (-1, 1)]
    torch.testing.assert_close((energies[1] - energies[0]) / (2 * eps),
                               -(force * direction).sum(), rtol=2e-5, atol=1e-8)


@pytest.mark.parametrize("backbone", ["painn", "mace"])
@pytest.mark.parametrize("kind", ["scalar", "vector"])
def test_energy_state_residual_and_force_rotation_and_translation(kind, backbone):
    model = _model(kind, backbone).eval()
    rotation = torch.tensor([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]], dtype=torch.float64)
    positions = _batch()[properties.positions]
    field = torch.tensor([0.2, -0.1, 0.3] if kind == "vector" else [0.2], dtype=torch.float64)
    rotated_field = rotation @ field if kind == "vector" else field
    first = _prediction(model, _batch(positions.clone(), field=field))
    second = _prediction(model, _batch(positions @ rotation.T + 2.5, field=rotated_field))
    torch.testing.assert_close(first["labels"]["condition_energy"]["values"],
                               second["labels"]["condition_energy"]["values"], rtol=1e-8, atol=1e-9)
    for key in ("variable_value", "condition_forces"):
        initial = first["labels"][key]["values"]
        expected = initial.reshape(-1, 3) @ rotation.T if key == "condition_forces" or kind == "vector" else initial
        torch.testing.assert_close(second["labels"][key]["values"], expected.reshape(-1), rtol=1e-7, atol=1e-9)
    initial = first["residuals"]["stationarity"]["values"]
    expected = initial.reshape(-1, 3) @ rotation.T if kind == "vector" else initial
    torch.testing.assert_close(second["residuals"]["stationarity"]["values"], expected.reshape(-1), rtol=1e-7, atol=1e-9)


def test_scalar_charge_is_projected_and_external_field_conditions_proposal():
    model = _model()
    prediction = _prediction(model, _batch(charge=0.7, field=[0.2]))
    torch.testing.assert_close(prediction["labels"]["variable_value"]["values"].sum(), torch.tensor(0.7, dtype=torch.float64))
    # Unconstrained field conditions must reach the state head, not just energy.
    zero = _prediction(model, _batch(field=[0.0]))["labels"]["variable_value"]["values"]
    field = _prediction(model, _batch(field=[0.4]))["labels"]["variable_value"]["values"]
    assert not torch.allclose(zero, field)


def test_predicted_training_never_calls_teacher_solve_or_full_hessian(monkeypatch):
    import curator.layer._variational_output as output_module
    import curator.variational.amortized_training as amortized
    import curator.variational.backends.dense as dense
    import curator.variational.implicit as implicit

    def forbidden(*args, **kwargs):
        raise AssertionError("joint training called a teacher/inner solve/full Hessian")

    monkeypatch.setattr(output_module, "solve_functional_reference", forbidden)
    monkeypatch.setattr(amortized, "prepare_amortized_training_batch", forbidden)
    monkeypatch.setattr(dense, "solve_dense", forbidden)
    monkeypatch.setattr(implicit, "solve_implicit", forbidden)
    monkeypatch.setattr(torch.autograd.functional, "hessian", forbidden)
    monkeypatch.setattr(torch.func, "hessian", forbidden)
    model = _model()
    prediction = _prediction(model, _batch())
    energy = prediction["labels"]["condition_energy"]["values"]
    force = prediction["labels"]["condition_forces"]["values"]
    residual = prediction["residuals"]["stationarity"]["values"]
    (energy.square().sum() + force.square().sum() + residual.square().sum()).backward()


def test_validation_no_grad_and_explicit_converged_reference():
    model = _model().eval()
    with torch.no_grad():
        predicted = _prediction(model, _batch())
    model.output_modules[0].state_mode = "converged"
    with torch.no_grad():
        converged = _prediction(model, _batch())
    assert converged["metadata"]["force_derivative"] == "stationary_envelope"
    assert all(value["converged"] and value["stable"] for value in converged["diagnostics"])
    assert float(converged["residuals"]["stationarity"]["values"].norm()) < 1e-7
    assert float(converged["labels"]["condition_energy"]["values"].sum()) <= float(
        predicted["labels"]["condition_energy"]["values"].sum()) + 1e-10
    model.train()
    with pytest.raises(RuntimeError, match="evaluation only"):
        model(_batch())


class _CustomEnergy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.curvature = torch.nn.Parameter(torch.tensor(0.8))

    def forward(self, context, state):
        target = context["scalar"][:, :state.shape[-1]]
        return torch.nn.functional.softplus(self.curvature) * (state - target).square().sum()


def test_custom_functional_is_replaceable_and_checkpoint_reconstructs():
    model = _model(internal_energy=_CustomEnergy())
    prediction = _prediction(model, _batch())
    rebuilt = _model(internal_energy=_CustomEnergy())
    rebuilt.load_state_dict(copy.deepcopy(model.state_dict()))
    second = _prediction(rebuilt, _batch())
    torch.testing.assert_close(prediction["labels"]["condition_energy"]["values"],
                               second["labels"]["condition_energy"]["values"])
    prediction["residuals"]["stationarity"]["values"].square().sum().backward()
    assert model.output_modules[0].internal_energy.curvature.grad is None
    assert any(parameter.grad is not None for parameter in model.output_modules[0].proposal.parameters())


def test_external_field_preserves_input_derivatives_and_optional_proposal_response():
    model = _model(compute_responses=True).eval()
    field = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    prediction = _prediction(model, _batch(field=field))
    energy = prediction["labels"]["condition_energy"]["values"].sum()
    energy_derivative = torch.autograd.grad(energy, field, retain_graph=True)[0]
    eps = 1e-5
    adjacent = [_prediction(model, _batch(field=[0.2 + sign * eps])) for sign in (-1, 1)]
    energy_fd = (adjacent[1]["labels"]["condition_energy"]["values"]
                 - adjacent[0]["labels"]["condition_energy"]["values"]) / (2 * eps)
    torch.testing.assert_close(energy_derivative, energy_fd, rtol=2e-5, atol=1e-8)
    response = prediction["responses"]["values"].reshape(3, 3)
    state_fd = (adjacent[1]["labels"]["variable_value"]["values"]
                - adjacent[0]["labels"]["variable_value"]["values"]) / (2 * eps)
    torch.testing.assert_close(response.sum(-1), state_fd, rtol=2e-5, atol=1e-8)
    assert prediction["metadata"]["response_derivative"] == "proposal_jacobian"


def test_converged_field_response_uses_implicit_response_of_same_functional():
    model = _model(internal_energy=_CustomEnergy(), compute_responses=True).eval()
    predicted = _prediction(model, _batch(field=[0.2]))
    model.output_modules[0].state_mode = "converged"
    converged = _prediction(model, _batch(field=[0.2]))
    curvature = 2 * torch.nn.functional.softplus(model.output_modules[0].internal_energy.curvature)
    response = converged["responses"]["values"].reshape(3, 3)
    torch.testing.assert_close(response, torch.eye(3, dtype=torch.float64) / curvature, rtol=1e-8, atol=1e-9)
    assert not torch.allclose(response, predicted["responses"]["values"].reshape(3, 3))
    assert converged["metadata"]["response_derivative"] == "implicit_stationary_response"


def test_converged_force_includes_geometry_dependent_constraint_derivative():
    model = _model(internal_energy=_CustomEnergy()).eval()
    model.output_modules[0].state_mode = "converged"
    variable = VariableSpec("internal", "scalar", 1, "dimensionless", "eV")
    boundary = BoundarySpec(variable, groups=(GroupSpec.all(),), equalities=(
        ConstraintSpec.weighted_sum("weighted_total", "all", 0.7, weight_field="weights"),))
    payload = asdict(boundary)

    def batch(positions):
        positions = positions.detach().requires_grad_(True)
        result = _batch(positions)
        result["weights"] = positions[:, 0] + 2.0
        result["variational"] = {
            "condition_geometry_index": torch.tensor([0]), "boundary_spec": (payload,),
            "boundary_hash": (canonical_boundary_hash(payload),),
            "condition_state_ptr": torch.tensor([0, 3]),
        }
        return result

    positions = _batch()[properties.positions]
    prediction = _prediction(model, batch(positions))
    force = prediction["labels"]["condition_forces"]["values"].reshape(3, 3)
    direction = torch.zeros_like(positions)
    direction[1, 0] = 1.0
    eps = 1e-5
    adjacent = [_prediction(model, batch(positions + sign * eps * direction))
                ["labels"]["condition_energy"]["values"].sum() for sign in (-1, 1)]
    torch.testing.assert_close((adjacent[1] - adjacent[0]) / (2 * eps),
                               -force[1, 0], rtol=2e-5, atol=1e-8)


def test_default_graph_coupling_vanishes_through_second_derivative_at_cutoff():
    energy = StableInternalEnergy(2, 0, 1, "scalar", 4).double()
    distance = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    context = {
        "scalar": torch.zeros((2, 2), dtype=torch.float64),
        "vector": torch.zeros((2, 3, 0), dtype=torch.float64),
        "field": torch.zeros((2, 1), dtype=torch.float64),
        "network_field": torch.zeros((2, 1), dtype=torch.float64),
        "origin": torch.zeros((2, 1), dtype=torch.float64),
        "edge_index": torch.tensor([[0, 1]]), "edge_distance": distance,
        "cutoff": torch.tensor(2.0, dtype=torch.float64),
    }
    state = torch.tensor([[0.2], [-0.4]], dtype=torch.float64)
    local_context = dict(context, edge_index=torch.empty((0, 2), dtype=torch.long))
    coupling = energy(context, state) - energy(local_context, state)
    first = torch.autograd.grad(coupling, distance, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), distance)[0]
    assert coupling.item() == 0.0
    assert first.item() == 0.0
    assert second.item() == 0.0


def test_converged_envelope_rejects_a_misleading_second_derivative_graph():
    model = _model(retain_response_graph=True).eval()
    model.output_modules[0].state_mode = "converged"
    with pytest.raises(RuntimeError, match="linear-response/HVP"):
        model(_batch())


def test_proposal_may_share_gnn_features_but_not_physical_parameters():
    energy = _CustomEnergy()
    proposal = torch.nn.Module()
    proposal.shared_curvature = energy.curvature
    with pytest.raises(ValueError, match="proposal parameters must be exclusive"):
        _model(internal_energy=energy, proposal=proposal)


def test_predicted_auxiliary_state_cannot_be_given_a_physical_stationarity_loss():
    variable = VariableSpec("internal", "scalar", 1, "dimensionless", "eV",
                            semantics="predicted_auxiliary")
    payload = asdict(BoundarySpec(variable))
    batch = _batch()
    batch["variational"] = {
        "condition_geometry_index": torch.tensor([0]), "boundary_spec": (payload,),
        "boundary_hash": (canonical_boundary_hash(payload),),
        "condition_state_ptr": torch.tensor([0, 3]),
    }
    with pytest.raises(ValueError, match="variational_internal"):
        _model()(batch)
