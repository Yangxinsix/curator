"""Deployment skips diagnostics without changing the conservative model."""
import copy

import pytest
import torch

from test_acvr_joint_model import _batch, _model, _prediction
from curator.variational.observables import ObservableSpec


@pytest.mark.parametrize("kind", ["scalar", "vector"])
@pytest.mark.parametrize("charge", [None, 0.7])
def test_deployment_preserves_energy_force_state_and_uses_one_proposal(kind, charge):
    if kind == "vector" and charge is not None:
        pytest.skip("total charge applies to scalar states")
    model = _model(kind).eval()
    head = model.output_modules[0]
    calls = []
    hook = head.proposal.register_forward_hook(lambda *_: calls.append(1))
    original = _prediction(model, _batch(charge=charge))
    assert len(calls) == 2
    before = copy.deepcopy(model.state_dict())
    head.configure_deployment()
    calls.clear()
    actual = _prediction(model, _batch(charge=charge))
    assert len(calls) == 1
    for name in ("condition_energy", "condition_forces", "variable_value"):
        torch.testing.assert_close(actual["labels"][name]["values"],
                                   original["labels"][name]["values"], rtol=0, atol=0)
    assert actual["residuals"] == {}
    assert actual["metadata"]["residuals_computed"] is False
    assert actual["metadata"]["force_derivative"] == "full_proposal_path"
    assert actual["metadata"]["energy_state"] == "predicted"
    for name, value in before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        else:
            assert model.state_dict()[name] == value
    head.reset_deployment()
    restored = _prediction(model, _batch(charge=charge))
    torch.testing.assert_close(restored["residuals"]["stationarity"]["values"],
                               original["residuals"]["stationarity"]["values"], rtol=0, atol=0)
    hook.remove()


def test_deployment_keeps_optional_diagnostics():
    model = _model().eval()
    original = _prediction(model, _batch())
    model.output_modules[0].configure_deployment(compute_residuals=True)
    result = _prediction(model, _batch())
    torch.testing.assert_close(result["residuals"]["stationarity"]["values"],
                               original["residuals"]["stationarity"]["values"], rtol=0, atol=0)


def test_energy_only_skips_force_response_and_configured_observables(monkeypatch):
    model = _model(compute_responses=True, observables=(
        ObservableSpec("reported_energy", unit="eV"),)).eval()
    head = model.output_modules[0]
    original = _prediction(model, _batch())
    saved_specs = head.observable_specs
    head.configure_deployment(energy_only=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("energy-only called derivative observables")
    monkeypatch.setattr(head, "_observables", forbidden)
    monkeypatch.setattr(head, "_field_response", forbidden)
    result = _prediction(model, _batch())
    assert head.observable_specs == saved_specs and head.compute_forces and head.compute_responses
    assert set(result["labels"]) == {"condition_energy", "variable_value"}
    assert "responses" not in result
    assert result["metadata"]["derivative_order"] == 0
    torch.testing.assert_close(result["labels"]["condition_energy"]["values"],
                               original["labels"]["condition_energy"]["values"], rtol=0, atol=0)


def test_deployment_training_guard_and_old_checkpoint_defaults():
    model = _model()
    head = model.output_modules[0]
    with pytest.raises(RuntimeError, match="eval"):
        head.configure_deployment()
    model.eval()
    head.configure_deployment()
    model.train()
    with pytest.raises(RuntimeError, match="reset_deployment"):
        _prediction(model, _batch())
    head.reset_deployment()
    _prediction(model, _batch())["residuals"]["stationarity"]["values"].square().sum().backward()
    assert any(p.grad is not None for p in head.proposal.parameters())
    del head._deployment_options
    assert "stationarity" in _prediction(model.eval(), _batch())["residuals"]


@pytest.mark.parametrize("constraints", [0, 1, 2])
def test_reused_projection_matches_general_live_boundary_first_and_second_derivatives(constraints):
    from curator.variational.deployment import DeploymentFunctionalProblem
    from curator.variational.functionals import FunctionalProblem
    matrix = torch.tensor([[1.0, 0.8, -0.2], [0.1, 0.3, 1.2]], dtype=torch.float64)[:constraints].requires_grad_()
    target = torch.tensor([0.7, -0.4], dtype=torch.float64)[:constraints].requires_grad_()
    point = torch.tensor([0.2, 0.3, -0.5], dtype=torch.float64, requires_grad=True)
    original = FunctionalProblem(lambda x: x.square().sum(), 3, matrix, target)
    deployed = DeploymentFunctionalProblem(lambda x: x.square().sum(), 3, matrix, target)
    assert deployed.with_energy(lambda x: x.sum())._schur is deployed._schur
    outputs = [problem.project_state(point) for problem in (original, deployed)]
    torch.testing.assert_close(*outputs, rtol=1e-13, atol=1e-14)
    if constraints:
        gradients = [torch.autograd.grad(value.square().sum(), (matrix, target, point),
                                        create_graph=True, retain_graph=True) for value in outputs]
        for first, second in zip(*gradients):
            torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-13)
        second = [torch.autograd.grad(sum(g.square().sum() for g in values),
                                     (matrix, target, point), retain_graph=True) for values in gradients]
        for first, other in zip(*second):
            torch.testing.assert_close(first, other, rtol=1e-11, atol=1e-12)


def test_projection_reuse_rejects_bad_constraints_and_training():
    from curator.variational.deployment import DeploymentFunctionalProblem
    matrix = torch.ones((2, 3), dtype=torch.float64)
    with pytest.raises(ValueError):
        DeploymentFunctionalProblem(lambda x: x.sum(), 3, matrix, matrix.new_zeros(2))
    model = _model().eval()
    head = model.output_modules[0]
    with pytest.raises(ValueError, match="diagnostics-off"):
        head.configure_deployment(compute_residuals=True, reuse_constraint_projection=True)
    ordinary = _prediction(model, _batch(charge=0.7))
    head.configure_deployment(reuse_constraint_projection=True)
    result = _prediction(model, _batch(charge=0.7))
    for name in ("condition_energy", "condition_forces", "variable_value"):
        torch.testing.assert_close(result["labels"][name]["values"],
                                   ordinary["labels"][name]["values"], rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("mode", ["full_hvp", "diagonal_ewald"])
def test_analytic_energy_only_supports_no_grad_without_changing_energy(mode):
    from test_acvr_correction_mode import _correction, _periodic
    model = _correction(mode).eval()
    head = model.output_modules[0]
    head.internal_energy.derivative_backend = "analytic"
    ordinary = _prediction(model, _periodic())["labels"]["condition_energy"]["values"]
    head.configure_deployment(energy_only=True, reuse_constraint_projection=True)
    with torch.no_grad():
        result = _prediction(model, _periodic())
    value = result["labels"]["condition_energy"]["values"]
    torch.testing.assert_close(value, ordinary, rtol=1e-13, atol=1e-14)
    assert not value.requires_grad and value.grad_fn is None
    with torch.inference_mode(), pytest.raises(RuntimeError):
        _prediction(model, _periodic())
    head.internal_energy.derivative_backend = "autograd"
    with torch.no_grad():
        automatic = _prediction(model, _periodic())
    assert automatic["labels"]["condition_energy"]["values"].requires_grad



def test_model_unknown_derivative_module_keeps_grad_enabled():
    from test_acvr_correction_mode import _correction, _periodic
    model = _correction("full_hvp").eval()
    head = model.output_modules[0]
    head.internal_energy.derivative_backend = "analytic"
    head.configure_deployment(energy_only=True)
    unknown = torch.nn.Identity()
    unknown.requires_energy_grad = True
    model.add_module("unknown_derivative_module", unknown)
    with torch.no_grad():
        result = _prediction(model, _periodic())
    assert result["labels"]["condition_energy"]["values"].requires_grad


def _four_geometries():
    from curator.data import properties
    from curator.data.collate_atoms_data import collate_atoms_data
    samples = []
    for index, size in enumerate([3, 2, 3, 2]):
        sample = _batch(charge=0.2 * index)
        sample[properties.positions] = sample[properties.positions][:size] * (1 + 0.1 * index)
        sample[properties.Z] = sample[properties.Z][:size]
        edges = sample[properties.edge_idx]
        sample[properties.edge_idx] = edges[(edges < size).all(dim=1)]
        sample[properties.n_pairs] = torch.tensor([len(sample[properties.edge_idx])])
        sample[properties.n_atoms] = torch.tensor([size])
        sample[properties.image_idx] = torch.zeros(size, dtype=torch.long)
        sample[properties.pbc] = torch.zeros((1, 3), dtype=torch.bool)
        sample[properties.cell] = torch.eye(3, dtype=torch.float64)[None] * 8
        samples.append(sample)
    return collate_atoms_data(samples)


@pytest.mark.parametrize("order", [[0, 1, 2, 3], [3, 0, 2, 1], [2, 0], [0, 0, 2, 3]])
@pytest.mark.parametrize("backbone", ["painn", "mace"])
def test_batched_force_is_same_full_derivative_and_falls_back_for_repeated_geometry(
        order, backbone, monkeypatch):
    model = _model(backbone=backbone, batch_observables=True).eval()
    head = model.output_modules[0]
    _, boundaries = head._conditions(_four_geometries())
    condition_boundaries = tuple(boundaries[index] for index in range(len(order)))
    monkeypatch.setattr(head, "_conditions", lambda data: (torch.tensor(order), condition_boundaries))
    calls = []
    original_grad = torch.autograd.grad
    def record(*args, **kwargs):
        inputs = args[1] if len(args) > 1 else kwargs["inputs"]
        if isinstance(inputs, torch.Tensor) and inputs.shape == (10, 3):
            calls.append(1)
        return original_grad(*args, **kwargs)
    monkeypatch.setattr(torch.autograd, "grad", record)
    head.batch_observables = False
    ordinary = _prediction(model, _four_geometries())
    assert len(calls) == len(order)
    head.batch_observables = True
    head.configure_deployment(batch_forces=True)
    calls.clear()
    result = _prediction(model, _four_geometries())
    unique = len(set(order)) == len(order)
    assert len(calls) == (1 if unique else len(order))
    assert result["metadata"]["batched_force_derivative"] is unique
    for name in ("condition_energy", "condition_forces", "variable_value"):
        torch.testing.assert_close(result["labels"][name]["values"],
                                   ordinary["labels"][name]["values"], rtol=1e-11, atol=1e-12)
        assert result["labels"][name]["shape"] == ordinary["labels"][name]["shape"]


def test_batched_force_requires_geometry_independence_assertion():
    model = _model().eval()
    with pytest.raises(ValueError, match="independent geometries"):
        model.output_modules[0].configure_deployment(batch_forces=True)



@pytest.mark.parametrize("mode", ["full_hvp", "diagonal_ewald"])
def test_batched_force_keeps_moving_anchor_and_proposal_correction_chain(mode):
    from test_acvr_correction_mode import _correction
    model = _correction(mode, backbone="mace").eval()
    head = model.output_modules[0]
    head.batch_observables = True
    head.internal_energy.derivative_backend = "analytic"
    ordinary = _prediction(model, _four_geometries())
    head.configure_deployment(batch_forces=True, reuse_constraint_projection=True)
    result = _prediction(model, _four_geometries())
    assert result["metadata"]["batched_force_derivative"] is True
    for name in ("condition_energy", "condition_forces", "variable_value",
                 "initial_variable_value", "proposal_delta"):
        torch.testing.assert_close(result["labels"][name]["values"],
                                   ordinary["labels"][name]["values"], rtol=1e-10, atol=1e-11)

