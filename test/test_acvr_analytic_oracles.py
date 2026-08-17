"""Analytic oracle tests for the dense ACVR quadratic core.

These tests deliberately use only the public ``curator.variational`` API.  The
reference answers are either closed-form expressions or independently assembled
NumPy KKT systems.  In particular, no production helper is used to construct an
expected solution.

Conventions frozen by this test contract
----------------------------------------

* ``A`` has shape ``[m, n]`` and encodes ``A @ x = b``.
* The objective is ``0.5 * x.T @ H @ x + c.T @ x + constant``.
* The Lagrangian is ``L = objective + lambda.T @ (A @ x - b)``.
* Consequently, stationarity is ``H @ x + c + A.T @ lambda = 0``.
"""

import numpy as np
import pytest
import torch

from curator.variational import (
    QuadraticProblem,
    evaluate_kkt,
    project_to_equalities,
    solve_dense,
    solve_linear_response,
    solve_schur,
)


DTYPE = torch.float64
ATOL = 2.0e-11


def _tensor(value):
    return torch.as_tensor(value, dtype=DTYPE)


def _empty_constraints(num_variables):
    return torch.empty((0, num_variables), dtype=DTYPE), torch.empty(0, dtype=DTYPE)


def _numpy_kkt_oracle(hessian, linear, constraint_matrix, constraint_target):
    """Solve the explicitly assembled KKT system with NumPy."""

    hessian = np.asarray(hessian, dtype=np.float64)
    linear = np.asarray(linear, dtype=np.float64)
    constraint_matrix = np.asarray(constraint_matrix, dtype=np.float64)
    constraint_target = np.asarray(constraint_target, dtype=np.float64)
    num_variables = hessian.shape[0]
    num_constraints = constraint_matrix.shape[0]

    if num_constraints == 0:
        return np.linalg.solve(hessian, -linear), np.empty(0, dtype=np.float64)

    kkt = np.block(
        [
            [hessian, constraint_matrix.T],
            [constraint_matrix, np.zeros((num_constraints, num_constraints))],
        ]
    )
    rhs = np.concatenate((-linear, constraint_target))
    solution = np.linalg.solve(kkt, rhs)
    return solution[:num_variables], solution[num_variables:]


def _random_problem(seed, num_variables=8, num_constraints=3):
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(num_variables, num_variables))
    hessian = factor.T @ factor + 0.75 * np.eye(num_variables)
    linear = rng.normal(size=num_variables)

    # Orthonormal rows make the constraints unambiguously full row rank while
    # leaving the conditioning of the physical Hessian nontrivial.
    basis, _ = np.linalg.qr(rng.normal(size=(num_variables, num_constraints)))
    constraint_matrix = basis.T
    constraint_target = rng.normal(size=num_constraints)
    return hessian, linear, constraint_matrix, constraint_target


@pytest.mark.parametrize("seed", [3, 19, 71])
def test_random_spd_constrained_problem_matches_independent_kkt_oracle(seed):
    hessian, linear, constraint_matrix, constraint_target = _random_problem(seed)
    constant = 1.375
    expected_state, expected_multipliers = _numpy_kkt_oracle(
        hessian,
        linear,
        constraint_matrix,
        constraint_target,
    )
    problem = QuadraticProblem(
        hessian=_tensor(hessian),
        linear=_tensor(linear),
        constraint_matrix=_tensor(constraint_matrix),
        constraint_target=_tensor(constraint_target),
        constant=_tensor(constant),
    )

    result = solve_dense(problem)

    torch.testing.assert_close(result.state, _tensor(expected_state), atol=ATOL, rtol=0.0)
    torch.testing.assert_close(
        result.multipliers,
        _tensor(expected_multipliers),
        atol=ATOL,
        rtol=0.0,
    )
    expected_energy = (
        0.5 * expected_state @ hessian @ expected_state
        + linear @ expected_state
        + constant
    )
    torch.testing.assert_close(result.energy, _tensor(expected_energy), atol=ATOL, rtol=0.0)

    residual = evaluate_kkt(problem, result.state, result.multipliers)
    torch.testing.assert_close(
        residual.stationarity,
        torch.zeros_like(result.state),
        atol=ATOL,
        rtol=0.0,
    )
    torch.testing.assert_close(
        residual.primal,
        torch.zeros_like(problem.constraint_target),
        atol=ATOL,
        rtol=0.0,
    )
    assert residual.stationarity_norm <= 1.0e-10
    assert residual.primal_norm <= 1.0e-10


def test_fixed_total_constraint_matches_closed_form_solution():
    diagonal = _tensor([1.25, 2.0, 4.5, 7.0])
    hessian = torch.diag(diagonal)
    linear = _tensor([0.4, -0.7, 0.2, 1.1])
    total = _tensor(0.65)
    constraint_matrix = torch.ones((1, diagonal.numel()), dtype=DTYPE)
    problem = QuadraticProblem(
        hessian=hessian,
        linear=linear,
        constraint_matrix=constraint_matrix,
        constraint_target=total.reshape(1),
        constant=_tensor(0.0),
    )

    result = solve_dense(problem)

    unconstrained = -linear / diagonal
    inverse_hessian_ones = diagonal.reciprocal()
    correction = (total - unconstrained.sum()) / inverse_hessian_ones.sum()
    expected_state = unconstrained + inverse_hessian_ones * correction
    expected_multiplier = -correction.reshape(1)
    torch.testing.assert_close(result.state, expected_state, atol=ATOL, rtol=0.0)
    torch.testing.assert_close(
        result.multipliers,
        expected_multiplier,
        atol=ATOL,
        rtol=0.0,
    )
    torch.testing.assert_close(result.state.sum(), total, atol=ATOL, rtol=0.0)


def test_dense_direct_and_schur_reference_paths_agree():
    hessian, linear, constraint_matrix, constraint_target = _random_problem(37)
    problem = QuadraticProblem(
        hessian=_tensor(hessian),
        linear=_tensor(linear),
        constraint_matrix=_tensor(constraint_matrix),
        constraint_target=_tensor(constraint_target),
    )

    direct = solve_dense(problem)
    schur = solve_schur(problem)

    torch.testing.assert_close(direct.state, schur.state, atol=ATOL, rtol=0.0)
    torch.testing.assert_close(
        direct.multipliers, schur.multipliers, atol=ATOL, rtol=0.0
    )
    torch.testing.assert_close(direct.energy, schur.energy, atol=ATOL, rtol=0.0)


def test_fragment_total_constraints_equilibrate_each_domain_independently():
    diagonal = _tensor([1.0, 3.0, 2.0, 5.0, 8.0])
    linear = _tensor([0.3, -0.2, 0.7, -0.4, 0.1])
    fragment_matrix = _tensor(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
        ]
    )
    targets = _tensor([0.4, -0.25])
    problem = QuadraticProblem(
        hessian=torch.diag(diagonal),
        linear=linear,
        constraint_matrix=fragment_matrix,
        constraint_target=targets,
        constant=_tensor(0.0),
    )

    result = solve_dense(problem)

    expected_state = -linear / diagonal
    for indices, target in ((slice(0, 2), targets[0]), (slice(2, 5), targets[1])):
        weights = diagonal[indices].reciprocal()
        expected_state[indices] += weights * (
            target - expected_state[indices].sum()
        ) / weights.sum()

    torch.testing.assert_close(result.state, expected_state, atol=ATOL, rtol=0.0)
    torch.testing.assert_close(
        fragment_matrix @ result.state,
        targets,
        atol=ATOL,
        rtol=0.0,
    )


def test_intensive_linear_control_enters_as_a_legendre_term():
    hessian = _tensor(
        [
            [3.0, 0.2, -0.1, 0.0],
            [0.2, 2.5, 0.3, 0.1],
            [-0.1, 0.3, 2.0, 0.2],
            [0.0, 0.1, 0.2, 1.75],
        ]
    )
    base_linear = _tensor([0.3, -0.5, 0.8, -0.2])
    # Rows map the internal state to two extensive observables y = B @ x.
    control_matrix = _tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])
    control = _tensor([0.7, -0.35])
    constraint_matrix, constraint_target = _empty_constraints(hessian.shape[0])

    # Phi_h(x) = Phi_0(x) - h.T @ (B @ x).
    effective_linear = base_linear - control_matrix.T @ control
    problem = QuadraticProblem(
        hessian=hessian,
        linear=effective_linear,
        constraint_matrix=constraint_matrix,
        constraint_target=constraint_target,
        constant=_tensor(0.0),
    )

    result = solve_dense(problem)
    expected_state = torch.linalg.solve(
        hessian,
        control_matrix.T @ control - base_linear,
    )

    torch.testing.assert_close(result.state, expected_state, atol=ATOL, rtol=0.0)
    assert result.multipliers.shape == (0,)


def test_robin_environment_renormalizes_curvature_and_linear_term():
    base_hessian = _tensor(
        [
            [2.0, 0.1, 0.0, -0.1],
            [0.1, 2.5, 0.2, 0.0],
            [0.0, 0.2, 3.0, 0.15],
            [-0.1, 0.0, 0.15, 2.25],
        ]
    )
    base_linear = _tensor([0.2, -0.4, 0.1, 0.5])
    observable_matrix = _tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    environment_curvature = _tensor([[1.2, 0.15], [0.15, 0.8]])
    environment_target = _tensor([0.3, -0.2])
    environment_field = _tensor([0.4, -0.1])
    base_constant = _tensor(0.6)
    constraint_matrix, constraint_target = _empty_constraints(base_hessian.shape[0])

    # F_env = 1/2 (B x-y_ref)^T K (B x-y_ref) - h0^T B x.
    effective_hessian = (
        base_hessian
        + observable_matrix.T @ environment_curvature @ observable_matrix
    )
    effective_linear = base_linear - observable_matrix.T @ (
        environment_field + environment_curvature @ environment_target
    )
    effective_constant = (
        base_constant
        + 0.5 * environment_target @ environment_curvature @ environment_target
    )
    problem = QuadraticProblem(
        hessian=effective_hessian,
        linear=effective_linear,
        constraint_matrix=constraint_matrix,
        constraint_target=constraint_target,
        constant=effective_constant,
    )

    result = solve_dense(problem)
    expected_state = torch.linalg.solve(effective_hessian, -effective_linear)
    direct_energy = (
        0.5 * expected_state @ base_hessian @ expected_state
        + base_linear @ expected_state
        + base_constant
        + 0.5
        * (observable_matrix @ expected_state - environment_target)
        @ environment_curvature
        @ (observable_matrix @ expected_state - environment_target)
        - environment_field @ (observable_matrix @ expected_state)
    )

    torch.testing.assert_close(result.state, expected_state, atol=ATOL, rtol=0.0)
    torch.testing.assert_close(result.energy, direct_energy, atol=ATOL, rtol=0.0)


def test_metric_projection_matches_closed_form_and_is_exactly_feasible():
    hessian, _, constraint_matrix, constraint_target = _random_problem(
        seed=107,
        num_variables=7,
        num_constraints=2,
    )
    raw_state = _tensor([0.1, -0.2, 0.7, 0.4, -0.3, 0.2, 0.9])
    metric = _tensor(hessian)
    constraints = _tensor(constraint_matrix)
    targets = _tensor(constraint_target)

    projection = project_to_equalities(
        raw_state,
        constraints,
        targets,
        metric=metric,
    )
    projected = projection.state

    metric_inverse_constraints = torch.linalg.solve(metric, constraints.T)
    schur = constraints @ metric_inverse_constraints
    expected = raw_state + metric_inverse_constraints @ torch.linalg.solve(
        schur,
        targets - constraints @ raw_state,
    )
    torch.testing.assert_close(projected, expected, atol=ATOL, rtol=0.0)
    torch.testing.assert_close(
        constraints @ projected,
        targets,
        atol=ATOL,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "bad_metric",
    [
        _tensor([1.0, -0.5, 2.0]),
        _tensor([[1.0, 0.4, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        _tensor([[1.0, 0.0, 0.0], [0.0, -0.25, 0.0], [0.0, 0.0, 1.0]]),
    ],
    ids=["negative-diagonal", "nonsymmetric-dense", "indefinite-dense"],
)
def test_projection_rejects_metrics_that_do_not_define_a_positive_norm(bad_metric):
    state = _tensor([0.2, -0.1, 0.4])
    constraints = torch.ones((1, 3), dtype=DTYPE)
    target = _tensor([0.0])

    with pytest.raises(ValueError, match="(?i)metric|positive|symmetric|definite"):
        project_to_equalities(
            state,
            constraints,
            target,
            metric=bad_metric,
        )


def test_redundant_constraints_are_rejected_with_a_clear_diagnostic():
    hessian = torch.eye(3, dtype=DTYPE)
    linear = torch.zeros(3, dtype=DTYPE)
    dependent_constraints = _tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    consistent_targets = _tensor([1.0, 2.0])

    with pytest.raises(ValueError, match="(?i)rank|dependent|redundant"):
        solve_dense(
            QuadraticProblem(
                hessian=hessian,
                linear=linear,
                constraint_matrix=dependent_constraints,
                constraint_target=consistent_targets,
                constant=_tensor(0.0),
            )
        )


def test_conflicting_constraints_are_rejected_as_infeasible():
    hessian = torch.eye(3, dtype=DTYPE)
    linear = torch.zeros(3, dtype=DTYPE)
    dependent_constraints = _tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    conflicting_targets = _tensor([1.0, 2.5])

    with pytest.raises(ValueError, match="(?i)inconsistent|conflict|infeasible|feasible"):
        solve_dense(
            QuadraticProblem(
                hessian=hessian,
                linear=linear,
                constraint_matrix=dependent_constraints,
                constraint_target=conflicting_targets,
                constant=_tensor(0.0),
            )
        )


def test_constraint_may_remove_an_unstable_global_hessian_direction():
    # The objective is not positive definite in the full space, but the only
    # negative direction is fixed by the equality.  Stability must therefore
    # be assessed on null(A), not by requiring H itself to be positive definite.
    problem = QuadraticProblem(
        hessian=torch.diag(_tensor([-1.0, 2.0, 3.0])),
        linear=_tensor([0.4, -0.6, 0.9]),
        constraint_matrix=_tensor([[1.0, 0.0, 0.0]]),
        constraint_target=_tensor([0.25]),
        constant=_tensor(0.0),
    )

    result = solve_dense(problem)

    torch.testing.assert_close(
        result.state,
        _tensor([0.25, 0.3, -0.3]),
        atol=ATOL,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.minimum_reduced_curvature,
        _tensor(2.0),
        atol=ATOL,
        rtol=0.0,
    )


def test_negative_curvature_in_the_constraint_tangent_space_is_rejected():
    problem = QuadraticProblem(
        hessian=torch.diag(_tensor([1.0, -0.25, 2.0])),
        linear=torch.zeros(3, dtype=DTYPE),
        constraint_matrix=_tensor([[1.0, 0.0, 0.0]]),
        constraint_target=_tensor([0.0]),
        constant=_tensor(0.0),
    )

    with pytest.raises(ValueError, match="(?i)stable|convex|curvature"):
        solve_dense(problem)


def test_linearized_kkt_response_matches_analytic_and_finite_difference():
    hessian, base_linear, constraint_matrix, constraint_target = _random_problem(
        seed=211,
        num_variables=6,
        num_constraints=2,
    )
    rng = np.random.default_rng(223)
    control_matrix = rng.normal(size=(3, 6))
    control = rng.normal(size=3)
    effective_linear = base_linear - control_matrix.T @ control
    problem = QuadraticProblem(
        hessian=_tensor(hessian),
        linear=_tensor(effective_linear),
        constraint_matrix=_tensor(constraint_matrix),
        constraint_target=_tensor(constraint_target),
        constant=_tensor(0.0),
    )

    response = solve_linear_response(
        problem,
        stationarity_rhs=_tensor(control_matrix.T),
        constraint_rhs=torch.zeros((2, 3), dtype=DTYPE),
    )

    kkt = np.block(
        [
            [hessian, constraint_matrix.T],
            [constraint_matrix, np.zeros((2, 2))],
        ]
    )
    expected = np.linalg.solve(
        kkt,
        np.vstack((control_matrix.T, np.zeros((2, 3)))),
    )
    torch.testing.assert_close(
        response.state,
        _tensor(expected[:6]),
        atol=ATOL,
        rtol=0.0,
    )
    torch.testing.assert_close(
        response.multipliers,
        _tensor(expected[6:]),
        atol=ATOL,
        rtol=0.0,
    )

    epsilon = 1.0e-5
    finite_difference_columns = []
    for control_index in range(control.size):
        perturbation = np.zeros_like(control)
        perturbation[control_index] = epsilon
        states = []
        for sign in (-1.0, 1.0):
            perturbed_linear = base_linear - control_matrix.T @ (
                control + sign * perturbation
            )
            perturbed_problem = QuadraticProblem(
                hessian=_tensor(hessian),
                linear=_tensor(perturbed_linear),
                constraint_matrix=_tensor(constraint_matrix),
                constraint_target=_tensor(constraint_target),
                constant=_tensor(0.0),
            )
            states.append(solve_dense(perturbed_problem).state)
        finite_difference_columns.append((states[1] - states[0]) / (2.0 * epsilon))
    finite_difference = torch.stack(finite_difference_columns, dim=1)
    torch.testing.assert_close(
        response.state,
        finite_difference,
        atol=2.0e-9,
        rtol=1.0e-7,
    )


def test_constraint_target_response_uses_the_lower_kkt_rhs():
    hessian, linear, constraint_matrix, constraint_target = _random_problem(
        seed=271,
        num_variables=7,
        num_constraints=2,
    )
    problem = QuadraticProblem(
        hessian=_tensor(hessian),
        linear=_tensor(linear),
        constraint_matrix=_tensor(constraint_matrix),
        constraint_target=_tensor(constraint_target),
        constant=_tensor(0.0),
    )
    response = solve_linear_response(
        problem,
        stationarity_rhs=torch.zeros((7, 2), dtype=DTYPE),
        constraint_rhs=torch.eye(2, dtype=DTYPE),
    )

    kkt = np.block(
        [
            [hessian, constraint_matrix.T],
            [constraint_matrix, np.zeros((2, 2))],
        ]
    )
    expected = np.linalg.solve(
        kkt,
        np.vstack((np.zeros((7, 2)), np.eye(2))),
    )
    torch.testing.assert_close(
        response.state,
        _tensor(expected[:7]),
        atol=ATOL,
        rtol=0.0,
    )
    torch.testing.assert_close(
        response.multipliers,
        _tensor(expected[7:]),
        atol=ATOL,
        rtol=0.0,
    )

    epsilon = 1.0e-5
    finite_difference_columns = []
    for constraint_index in range(2):
        perturbation = np.zeros(2)
        perturbation[constraint_index] = epsilon
        states = []
        for sign in (-1.0, 1.0):
            perturbed_problem = QuadraticProblem(
                hessian=_tensor(hessian),
                linear=_tensor(linear),
                constraint_matrix=_tensor(constraint_matrix),
                constraint_target=_tensor(
                    constraint_target + sign * perturbation
                ),
                constant=_tensor(0.0),
            )
            states.append(solve_dense(perturbed_problem).state)
        finite_difference_columns.append((states[1] - states[0]) / (2.0 * epsilon))
    finite_difference = torch.stack(finite_difference_columns, dim=1)
    torch.testing.assert_close(
        response.state,
        finite_difference,
        atol=2.0e-9,
        rtol=1.0e-7,
    )


@pytest.mark.parametrize("num_components", [1, 3])
def test_same_dense_solver_handles_scalar_and_vector_internal_variables(num_components):
    num_atoms = 4
    num_variables = num_atoms * num_components
    rng = np.random.default_rng(307 + num_components)
    factor = rng.normal(size=(num_variables, num_variables))
    hessian = factor.T @ factor + np.eye(num_variables)
    linear = rng.normal(size=num_variables)

    # Atom-major flattening: x.reshape(num_atoms, num_components).  Each row
    # fixes the atom-sum of one Cartesian/internal component.
    constraints = np.zeros((num_components, num_variables), dtype=np.float64)
    for component in range(num_components):
        constraints[component, component::num_components] = 1.0
    targets = np.linspace(-0.3, 0.45, num_components)
    expected_state, _ = _numpy_kkt_oracle(hessian, linear, constraints, targets)
    problem = QuadraticProblem(
        hessian=_tensor(hessian),
        linear=_tensor(linear),
        constraint_matrix=_tensor(constraints),
        constraint_target=_tensor(targets),
        constant=_tensor(0.0),
    )

    result = solve_dense(problem)

    torch.testing.assert_close(result.state, _tensor(expected_state), atol=ATOL, rtol=0.0)
    state_by_atom = result.state.reshape(num_atoms, num_components)
    torch.testing.assert_close(
        state_by_atom.sum(dim=0),
        _tensor(targets),
        atol=ATOL,
        rtol=0.0,
    )
