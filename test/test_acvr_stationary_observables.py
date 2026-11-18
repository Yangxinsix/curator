"""Independent analytic/finite-difference checks of relaxed observables."""

import pytest
import torch

from curator.variational.functionals import FunctionalProblem, solve_functional_reference
from curator.variational.observables import EnergyObservables, ObservableSpec
from curator.variational.stationary_observables import StationaryEnergyObservables


def _quadratic():
    geometry = torch.tensor([.2, -.1], dtype=torch.float64, requires_grad=True)
    field = torch.tensor([.3, -.4, .2], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([.7], dtype=torch.float64, requires_grad=True)
    hessian = torch.diag(torch.stack((2 + geometry[0].square(), 3 + geometry[1].exp(),
                                     4 + .2 * geometry.sum().square())))
    matrix = torch.stack((1 + geometry[0], geometry[1], geometry.new_tensor(1.))).reshape(1, 3)
    rhs = target + .3 * geometry[:1].square()
    constant = .2 * geometry.square().sum() + .1 * field.square().sum()
    problem = FunctionalProblem(lambda state: .5 * state @ hessian @ state - field @ state + constant,
                                3, matrix, rhs)
    # Independent small dense solve is the analytic truth, not the new executor.
    kkt = torch.cat((torch.cat((hessian, matrix.T), dim=1),
                     torch.cat((matrix, matrix.new_zeros((1, 1))), dim=1)), dim=0)
    solution = torch.linalg.solve(kkt, torch.cat((field, rhs)))
    exact_energy = problem.energy(solution[:3])
    inputs = {"geometry": geometry, "field": field, "target": target}
    return problem, solution[:3], solution[3:], inputs, exact_energy


def test_relaxed_first_second_and_mixed_derivatives_include_live_constraint_geometry(monkeypatch):
    problem, state, multipliers, inputs, exact_energy = _quadratic()
    expected = EnergyObservables(exact_energy, inputs)
    actual = StationaryEnergyObservables(problem, state, multipliers, inputs,
                                         normalizers={"volume": 2.})

    def forbidden(*args, **kwargs):
        raise AssertionError("stationary observable built a dense Hessian/Jacobian")

    monkeypatch.setattr(torch.autograd.functional, "hessian", forbidden)
    monkeypatch.setattr(torch.autograd.functional, "jacobian", forbidden)
    monkeypatch.setattr(torch.func, "hessian", forbidden)
    specs = [ObservableSpec("energy")]
    for first in inputs:
        specs.append(ObservableSpec(first, (first,)))
        for second in inputs:
            specs.append(ObservableSpec(first + "_" + second, (first, second)))
    reference = expected.evaluate(specs)
    with torch.no_grad():
        values = actual.evaluate(specs)
    for name in reference:
        torch.testing.assert_close(values[name], reference[name], rtol=1e-9, atol=1e-10)
        assert not values[name].requires_grad
    # An unrelaxed partial Hessian omits the state redistribution altogether.
    partial = EnergyObservables(problem.energy(state.detach()), inputs).derivative(("field", "field"))
    assert not torch.allclose(partial, values["field_field"])
    formatted = actual.evaluate([ObservableSpec("mixed", ("field", "geometry"), factor=-1,
                                               permutation=(1, 0), reshape=(2, 3), normalization="volume")])
    torch.testing.assert_close(formatted["mixed"], -reference["field_geometry"].T / 2)


def test_nonlinear_polarizability_agrees_with_new_equilibria_finite_difference():
    stiffness = torch.tensor([2.], dtype=torch.float64, requires_grad=True)
    field = torch.tensor([2.7], dtype=torch.float64, requires_grad=True)

    def solve(a, f):
        problem = FunctionalProblem(lambda state: .5 * a[0] * state.square().sum()
                                    + .175 * state.pow(4).sum() - f @ state, 1)
        solved = solve_functional_reference(problem, field.new_tensor([1.]))
        assert solved.converged and solved.stable
        return StationaryEnergyObservables(problem, solved.state, solved.multipliers,
                                           {"field": f, "stiffness": a})

    executor = solve(stiffness, field)
    values = executor.evaluate([ObservableSpec("dipole", ("field",), factor=-1),
                                ObservableSpec("alpha", ("field", "field"), factor=-1),
                                ObservableSpec("mixed", ("field", "stiffness"))])
    torch.testing.assert_close(values["dipole"], field.new_tensor([1.]))
    torch.testing.assert_close(values["alpha"], field.new_tensor([[1 / 4.1]]))
    torch.testing.assert_close(values["mixed"], field.new_tensor([[1 / 4.1]]))
    eps = 1e-4
    adjacent = [solve(stiffness, (field.detach() + sign * eps).requires_grad_(True))
                .derivative(("field",)) for sign in (-1, 1)]
    torch.testing.assert_close(-(adjacent[1] - adjacent[0]) / (2 * eps),
                               values["alpha"].reshape(-1), rtol=2e-6, atol=1e-9)


def test_empty_boundary_axes_unused_inputs_and_response_cache(monkeypatch):
    problem, state, multipliers, inputs, _ = _quadratic()
    inputs["empty"] = state.new_empty(0).requires_grad_(True)
    inputs["unused"] = state.new_ones(2).requires_grad_(True)
    executor = StationaryEnergyObservables(problem, state, multipliers, inputs)
    calls = []
    import curator.variational.stationary_observables as implementation

    original = implementation.solve_functional_linear_response

    def counted(*args, **kwargs):
        calls.append(args[2].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(implementation, "solve_functional_linear_response", counted)
    torch.testing.assert_close(executor.derivative(("empty",)), state.new_empty(0))
    assert executor.derivative(("empty", "field")).shape == (0, 3)
    assert executor.derivative(("field", "empty")).shape == (3, 0)
    assert not calls
    executor.derivative(("field", "field"))
    executor.derivative(("field", "geometry"))
    assert calls == [torch.Size([3, 3])]
    torch.testing.assert_close(executor.derivative(("unused", "field")), state.new_zeros(2, 3))


def test_large_coordinate_hvp_includes_relaxation_without_dense_hessian():
    coordinates = torch.linspace(-.2, .3, 30, dtype=torch.float64).requires_grad_(True)
    coupling = torch.linspace(.1, .8, 60, dtype=torch.float64).reshape(2, 30)
    matrix = torch.tensor([[2., .3], [.3, 3.]], dtype=torch.float64)
    problem = FunctionalProblem(lambda state: .5 * state @ matrix @ state
                                - state @ coupling @ coordinates + .2 * coordinates.square().sum(), 2)
    state = torch.linalg.solve(matrix, coupling @ coordinates)
    executor = StationaryEnergyObservables(problem, state, state.new_empty(0), {"positions": coordinates})
    with pytest.raises(ValueError, match="max_jacobian_rows"):
        executor.derivative(("positions", "positions"))
    direction = coordinates.square()
    expected = .4 * direction - coupling.T @ torch.linalg.solve(matrix, coupling @ direction)
    result = executor.hvp("positions", direction)
    torch.testing.assert_close(result, expected)
    assert not result.requires_grad
    with pytest.raises(ValueError, match="evaluation-only"):
        executor.derivative(("positions",), create_graph=True)
    with pytest.raises(ValueError, match="evaluation-only"):
        executor.hvp("positions", direction, create_graph=True)


def test_nonpositive_linear_response_is_reported():
    field = torch.tensor([0.], dtype=torch.float64, requires_grad=True)
    problem = FunctionalProblem(lambda state: -.5 * state.square().sum() - field @ state, 1)
    executor = StationaryEnergyObservables(problem, field.new_zeros(1), field.new_empty(0), {"field": field})
    with pytest.raises(RuntimeError, match="nonpositive_curvature"):
        executor.derivative(("field", "field"))


def test_vector_head_reference_observables_use_relaxed_polarizability():
    from test_acvr_global_proposal import _model
    from test_acvr_joint_model import _batch

    specs = [ObservableSpec("dipole", ("external_field",), factor=-1, unit="dimensionless"),
             ObservableSpec("polarizability", ("external_field", "external_field"), factor=-1, unit="1/eV")]
    model = _model("attention", kind="vector", compute_forces=False, retain_response_graph=True,
                   state_mode="converged", observables=specs).eval()
    field = torch.tensor([.2, -.1, .3], dtype=torch.float64)

    def predict(value):
        return model(_batch(field=value))["variational_prediction"]

    prediction = predict(field)
    alpha = prediction["labels"]["polarizability"]["values"].reshape(3, 3)
    assert prediction["metadata"]["energy_state"] == "converged"
    assert prediction["metadata"]["derivative_order"] == 2
    torch.testing.assert_close(alpha, alpha.T, rtol=1e-7, atol=1e-8)
    direction = field.new_tensor([.2, -.3, .1])
    eps = 1e-4
    adjacent = [predict(field + sign * eps * direction)["labels"]["dipole"]["values"] for sign in (-1, 1)]
    torch.testing.assert_close((adjacent[1] - adjacent[0]) / (2 * eps), alpha @ direction,
                               rtol=3e-5, atol=2e-7)
