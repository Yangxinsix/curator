"""Scalar-energy derivatives and teacher-free equality residuals."""

from dataclasses import replace

import pytest
import torch

from curator.variational import (
    BoundarySpec, ControlSpec, EnvironmentSpec, GroupSpec, LinearTermSpec,
    VariableLayout, VariableSpec, compile_boundary,
)
from curator.variational.functionals import FunctionalProblem, solve_functional_reference
from curator.variational.validation import (
    InconsistentConstraintsError, RankDeficientConstraintsError, null_space,
    thermodynamic_gauge_basis,
)


DTYPE = torch.float64


def test_nonlinear_gradient_and_lazy_hvp_match_finite_differences():
    def energy(state):
        return 0.25 * state.pow(4).sum() + state.square().sum() + 0.3 * state[0] * state[1]

    problem = FunctionalProblem(energy, 2)
    point = torch.tensor([0.4, -0.7], dtype=DTYPE)
    direction = torch.tensor([1.2, -0.5], dtype=DTYPE)
    step = 1e-5
    basis = torch.eye(2, dtype=DTYPE)
    gradient_fd = torch.stack([
        (energy(point + step * vector) - energy(point - step * vector)) / (2 * step)
        for vector in basis
    ])
    hvp_fd = (
        problem.state_gradient(point + step * direction)
        - problem.state_gradient(point - step * direction)
    ) / (2 * step)
    torch.testing.assert_close(problem.state_gradient(point), gradient_fd, atol=2e-10, rtol=1e-9)
    torch.testing.assert_close(problem.hvp(point, direction), hvp_fd, atol=2e-9, rtol=1e-9)


@pytest.mark.parametrize("constant", [False, True])
def test_affine_and_state_independent_energies_have_zero_hvp(constant):
    offset = torch.tensor(3.0, dtype=DTYPE, requires_grad=True)
    point = torch.tensor([0.4, 0.6], dtype=DTYPE, requires_grad=True)
    problem = FunctionalProblem(
        (lambda state: offset) if constant else (lambda state: offset + 2.0 * state.sum()), 2,
    )
    gradient = problem.state_gradient(point, create_graph=True)
    torch.testing.assert_close(gradient, torch.full_like(point, 0.0 if constant else 2.0))
    torch.testing.assert_close(problem.hvp(point, torch.ones_like(point)), torch.zeros_like(point))
    gradient.square().sum().backward()
    torch.testing.assert_close(point.grad, torch.zeros_like(point))


def test_residual_backward_preserves_proposal_and_context_gradients_without_dense_hessian(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("dense Hessian or null-space basis reached the residual path")

    monkeypatch.setattr(torch.autograd.functional, "hessian", forbidden)
    monkeypatch.setattr(torch.linalg, "svd", forbidden)
    stiffness = torch.tensor(2.0, dtype=DTYPE, requires_grad=True)
    proposal = torch.tensor(0.3, dtype=DTYPE, requires_grad=True)
    state = torch.stack((proposal, -proposal))
    problem = FunctionalProblem(
        lambda value: 0.5 * stiffness * value.square().sum() + value[0], 2,
        torch.ones((1, 2), dtype=DTYPE), torch.zeros(1, dtype=DTYPE),
    )
    projected = problem.project_state(state)
    # Explicit derivative requests also work inside an evaluation no_grad block.
    with torch.no_grad():
        result = problem.stationarity(projected, create_graph=True)
    loss = 0.5 * result.stationarity.square().sum()
    grad_proposal, grad_stiffness = torch.autograd.grad(loss, (proposal, stiffness))
    # Residual = (+a*p+0.5, -a*p-0.5); both derivative routes are available.
    torch.testing.assert_close(grad_proposal, proposal.new_tensor(4.4))
    torch.testing.assert_close(grad_stiffness, stiffness.new_tensor(0.66))
    torch.testing.assert_close(problem.hvp(state, torch.ones_like(state)), state.new_full((2,), 2.0))


def test_hvp_direction_is_a_cotangent_and_retains_requested_derivatives():
    parameter = torch.tensor(1.7, dtype=DTYPE, requires_grad=True)
    state = torch.tensor([0.4, -0.2], dtype=DTYPE, requires_grad=True)
    # The direction depends on state; H*v must not include the derivative of v
    # that differentiating the scalar grad(Phi).dot(v(state)) would introduce.
    direction = state.square()
    problem = FunctionalProblem(lambda value: parameter * value.pow(3).sum(), 2)
    product = problem.hvp(state, direction, create_graph=True)
    torch.testing.assert_close(product, 6.0 * parameter * state.pow(3))
    grad_state, grad_parameter = torch.autograd.grad(product.sum(), (state, parameter))
    torch.testing.assert_close(grad_state, 18.0 * parameter * state.square())
    torch.testing.assert_close(grad_parameter, 6.0 * state.pow(3).sum())


def test_constraint_stationarity_distinguishes_tangent_gradient_and_feasibility():
    problem = FunctionalProblem(
        lambda value: 0.5 * value.square().sum() + torch.tensor([1.0, 3.0], dtype=DTYPE) @ value,
        2, torch.ones((1, 2), dtype=DTYPE), torch.zeros(1, dtype=DTYPE),
    )
    state = torch.tensor([1.0, -1.0], dtype=DTYPE)
    residual = problem.stationarity(state)
    torch.testing.assert_close(residual.gradient, torch.tensor([2.0, 2.0], dtype=DTYPE))
    torch.testing.assert_close(residual.stationarity, torch.zeros(2, dtype=DTYPE))
    torch.testing.assert_close(residual.multipliers, torch.tensor([-2.0], dtype=DTYPE))
    torch.testing.assert_close(residual.feasibility, torch.zeros(1, dtype=DTYPE))
    # A normal displacement leaves projected stationarity zero but is infeasible.
    infeasible = problem.stationarity(state + 1.0)
    torch.testing.assert_close(infeasible.stationarity, torch.zeros(2, dtype=DTYPE))
    torch.testing.assert_close(infeasible.feasibility, torch.tensor([2.0], dtype=DTYPE))
    torch.testing.assert_close(problem.project_state(state + 1.0), state)


@pytest.mark.parametrize("inconsistent", [False, True])
def test_redundant_or_inconsistent_equalities_are_rejected(inconsistent):
    matrix = torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=DTYPE)
    target = torch.tensor([0.0, 1.0 if inconsistent else 0.0], dtype=DTYPE)
    error = InconsistentConstraintsError if inconsistent else RankDeficientConstraintsError
    with pytest.raises(error):
        FunctionalProblem(lambda state: state.square().sum(), 2, matrix, target)


def test_shapes_nonfinite_values_and_dtypes_are_rejected():
    problem = FunctionalProblem(lambda state: state.square().sum(), 2)
    with pytest.raises(ValueError, match="length n_state"):
        problem.state_gradient(torch.zeros(3, dtype=DTYPE))
    with pytest.raises(ValueError, match="finite"):
        problem.hvp(torch.zeros(2, dtype=DTYPE), torch.tensor([float("nan"), 0.0], dtype=DTYPE))
    with pytest.raises(ValueError, match="scalar tensor"):
        FunctionalProblem(lambda state: state.square(), 2).energy(torch.zeros(2, dtype=DTYPE))
    with pytest.raises(ValueError, match="supplied together"):
        FunctionalProblem(lambda state: state.sum(), 2, torch.ones(1, 2, dtype=DTYPE))
    with pytest.raises(ValueError, match="finite"):
        FunctionalProblem(lambda state: state.sum(), 2, torch.ones(1, 2, dtype=DTYPE), torch.tensor([float("nan")], dtype=DTYPE))
    with pytest.raises(ValueError, match="share dtype"):
        FunctionalProblem(lambda state: state.sum(), 2, torch.ones(1, 2, dtype=DTYPE), torch.zeros(1, dtype=DTYPE)).energy(torch.zeros(2))


def _compiled_boundary():
    variable = VariableSpec(name="q", kind="atomic_charge", components=1, unit="e", conjugate_unit="V")
    return compile_boundary(
        BoundarySpec(
            variable=variable,
            groups=(GroupSpec.from_indices("left", [0]), GroupSpec.from_indices("right", [1])),
            controls=(ControlSpec.group_potential("left_field", "left", 0.7, unit="V"),),
            environment=(EnvironmentSpec(
                name="reservoir", terms=(LinearTermSpec("right"),), target=-0.3,
                stiffness=1.5, field=-0.2, observable_unit="e",
            ),),
        ),
        VariableLayout(variable=variable, n_entities=2),
    )


def test_boundary_energy_matches_quadratic_adapter_values_and_context_derivatives():
    geometry = torch.tensor(0.4, dtype=DTYPE, requires_grad=True)
    boundary = replace(
        _compiled_boundary(),
        control_value=(geometry + 0.7).reshape(1),
        environment_matrix=torch.stack((geometry * 0.0, geometry + 1.0)).reshape(1, 2),
        environment_target=(geometry - 0.3).reshape(1),
        environment_field=(geometry - 0.2).reshape(1),
    )
    point = torch.tensor([0.2, -0.5], dtype=DTYPE, requires_grad=True)
    hessian = torch.diag(torch.tensor([2.0, 3.0], dtype=DTYPE))
    linear = torch.tensor([0.1, -0.4], dtype=DTYPE)
    base_energy = 0.5 * point @ hessian @ point + linear @ point + 0.6
    scalar_energy = base_energy + boundary.energy(point)
    quadratic_energy = boundary.to_problem(hessian, linear, constant=0.6).energy(point)
    torch.testing.assert_close(scalar_energy, quadratic_energy)
    scalar_derivatives = torch.autograd.grad(scalar_energy, (point, geometry), retain_graph=True)
    quadratic_derivatives = torch.autograd.grad(quadratic_energy, (point, geometry))
    for actual, expected in zip(scalar_derivatives, quadratic_derivatives):
        torch.testing.assert_close(actual, expected)


def test_reference_solver_converges_and_returns_detached_local_result():
    problem = FunctionalProblem(
        lambda state: 0.25 * state.pow(4).sum() + 0.5 * state.square().sum()
        + torch.tensor([-2.0, 2.0], dtype=DTYPE) @ state,
        2, torch.ones((1, 2), dtype=DTYPE), torch.zeros(1, dtype=DTYPE),
    )
    result = solve_functional_reference(problem, torch.tensor([0.8, -0.8], dtype=DTYPE))
    assert result.converged and result.stable and result.reliable
    torch.testing.assert_close(result.state, torch.tensor([1.0, -1.0], dtype=DTYPE), atol=1e-9, rtol=0)
    assert not result.state.requires_grad and not result.energy.requires_grad


def test_zero_residual_is_not_a_minimum_certificate():
    problem = FunctionalProblem(lambda state: -0.5 * state.square().sum(), 2)
    state = torch.zeros(2, dtype=DTYPE)
    torch.testing.assert_close(problem.stationarity(state).stationarity, state)
    result = solve_functional_reference(problem, state)
    assert result.converged and not result.stable and not result.reliable
    assert result.status == "unstable_stationary_point"


@pytest.mark.parametrize("n_constraints,n_controls", [(0, 3), (2, 3), (2, 9), (6, 3)])
def test_gauge_rowspace_matches_dense_reference(n_constraints, n_controls):
    generator = torch.Generator().manual_seed(31)
    matrix = torch.randn(n_constraints, 6, generator=generator, dtype=DTYPE)
    controls = torch.randn(n_controls, 6, generator=generator, dtype=DTYPE)
    if 0 < n_constraints < 6:
        controls[0] = matrix[0]
    tangent = null_space(matrix)
    reference = null_space(tangent.T @ controls.T, tolerance=1e-11)
    result = thermodynamic_gauge_basis(matrix, controls, tolerance=1e-11)
    # The basis is nonunique; compare control-space projectors.
    torch.testing.assert_close(result @ result.T, reference @ reference.T, atol=1e-10, rtol=0)


def test_large_layout_gauge_and_projection_never_build_state_square_matrices(monkeypatch):
    n_state = 2048
    matrix = torch.ones((1, n_state), dtype=DTYPE)
    controls = torch.stack((
        torch.cat((torch.ones(n_state // 2), torch.zeros(n_state // 2))),
        torch.cat((torch.zeros(n_state // 2), torch.ones(n_state // 2))),
    )).to(dtype=DTYPE)
    original_svd, original_eye = torch.linalg.svd, torch.eye

    def thin_svd(value, *args, **kwargs):
        assert not kwargs.get("full_matrices", True), "large layout requested full SVD"
        return original_svd(value, *args, **kwargs)

    def small_eye(n, *args, **kwargs):
        assert n != n_state, "state-space identity was materialized"
        return original_eye(n, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", thin_svd)
    monkeypatch.setattr(torch, "eye", small_eye)
    basis = thermodynamic_gauge_basis(matrix, controls)
    assert basis.shape == (2, 1)
    torch.testing.assert_close(basis @ basis.T, torch.full((2, 2), 0.5, dtype=DTYPE))
    problem = FunctionalProblem(lambda state: state.square().sum(), n_state, matrix, torch.zeros(1, dtype=DTYPE))
    point = problem.project_state(torch.ones(n_state, dtype=DTYPE))
    torch.testing.assert_close(point, torch.zeros_like(point))
    torch.testing.assert_close(problem.stationarity(point).stationarity, torch.zeros_like(point))


def test_all_gauge_controls_remain_gauge_after_projection_roundoff():
    matrix = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=DTYPE)
    controls = torch.cat((matrix, 1.7 * matrix), dim=0)
    result = thermodynamic_gauge_basis(matrix, controls)
    assert result.shape == (2, 2)
    torch.testing.assert_close(result @ result.T, torch.eye(2, dtype=DTYPE))
