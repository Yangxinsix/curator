"""Dense nonlinear KKT reference tests."""

import torch

from curator.variational import (
    NewtonKKTOptions,
    NonlinearEqualityProblem,
    QuadraticProblem,
    nonlinear_from_quadratic,
    solve_dense,
    solve_newton_kkt,
)


DTYPE = torch.float64


def test_quadratic_adapter_reaches_dense_kkt_solution_in_one_step():
    hessian = torch.tensor(
        [[3.0, 0.2, -0.1], [0.2, 2.4, 0.3], [-0.1, 0.3, 1.8]],
        dtype=DTYPE,
    )
    problem = QuadraticProblem(
        hessian=hessian,
        linear=torch.tensor([0.4, -0.7, 0.2], dtype=DTYPE),
        constraint_matrix=torch.ones((1, 3), dtype=DTYPE),
        constraint_target=torch.tensor([0.35], dtype=DTYPE),
    )
    exact = solve_dense(problem)
    result = solve_newton_kkt(
        nonlinear_from_quadratic(problem),
        torch.tensor([3.0, -2.0, 1.0], dtype=DTYPE),
    )

    assert result.status == "converged_minimum"
    assert result.iterations == 1
    assert result.reliable
    torch.testing.assert_close(result.state, exact.state, atol=2.0e-12, rtol=0.0)
    torch.testing.assert_close(
        result.multipliers, exact.multipliers, atol=2.0e-12, rtol=0.0
    )


def test_convex_quartic_with_total_constraint_converges_from_far_state():
    quadratic = torch.tensor([1.0, 1.5, 2.0], dtype=DTYPE)
    linear = torch.tensor([-0.4, 0.2, 0.8], dtype=DTYPE)
    total = 0.3
    problem = NonlinearEqualityProblem(
        n_state=3,
        n_constraints=1,
        energy_function=lambda state: (
            0.25 * state.pow(4).sum()
            + 0.5 * (quadratic * state.square()).sum()
            + linear @ state
        ),
        constraint_function=lambda state: torch.stack((state.sum() - total,)),
    )
    result = solve_newton_kkt(
        problem, torch.tensor([20.0, -10.0, 5.0], dtype=DTYPE)
    )

    assert result.reliable
    assert result.iterations > 1
    torch.testing.assert_close(
        result.state.sum(), torch.tensor(total, dtype=DTYPE), atol=2.0e-11, rtol=0.0
    )
    chemical_potential = result.state.pow(3) + quadratic * result.state + linear
    torch.testing.assert_close(
        chemical_potential,
        torch.full_like(chemical_potential, -result.multipliers[0]),
        atol=3.0e-9,
        rtol=0.0,
    )


def test_nonlinear_circle_constraint_matches_analytic_state_and_multiplier():
    target = torch.tensor([2.0, 1.0], dtype=DTYPE)
    problem = NonlinearEqualityProblem(
        n_state=2,
        n_constraints=1,
        energy_function=lambda state: 0.5 * (state - target).square().sum(),
        constraint_function=lambda state: torch.stack((state @ state - 1.0,)),
    )
    result = solve_newton_kkt(problem, torch.tensor([0.8, 0.6], dtype=DTYPE))

    expected_state = target / target.norm()
    expected_multiplier = (target.norm() - 1.0) / 2.0
    assert result.reliable
    torch.testing.assert_close(result.state, expected_state, atol=2.0e-11, rtol=0.0)
    torch.testing.assert_close(
        result.multipliers[0], expected_multiplier, atol=2.0e-11, rtol=0.0
    )
    torch.testing.assert_close(
        result.minimum_reduced_curvature,
        target.norm(),
        atol=2.0e-11,
        rtol=0.0,
    )


def test_constraint_hessian_is_used_in_the_newton_step():
    target = torch.tensor([2.0, 0.0], dtype=DTYPE)
    problem = NonlinearEqualityProblem(
        n_state=2,
        n_constraints=1,
        energy_function=lambda state: -(target @ state),
        constraint_function=lambda state: torch.stack((state @ state - 1.0,)),
    )
    result = solve_newton_kkt(
        problem,
        torch.tensor([0.8, 0.6], dtype=DTYPE),
        initial_multipliers=torch.tensor([1.0], dtype=DTYPE),
    )

    assert result.reliable
    torch.testing.assert_close(
        result.state, torch.tensor([1.0, 0.0], dtype=DTYPE), atol=2.0e-10, rtol=0.0
    )
    torch.testing.assert_close(
        result.multipliers, torch.tensor([1.0], dtype=DTYPE), atol=2.0e-10, rtol=0.0
    )
    torch.testing.assert_close(
        result.minimum_reduced_curvature,
        torch.tensor(2.0, dtype=DTYPE),
        atol=2.0e-10,
        rtol=0.0,
    )


def test_stationary_saddle_is_never_reported_as_reliable():
    problem = NonlinearEqualityProblem(
        n_state=2,
        n_constraints=0,
        energy_function=lambda state: -0.5 * state[0].square()
        + 0.5 * state[1].square(),
        constraint_function=lambda state: state.new_zeros((0,)),
    )
    result = solve_newton_kkt(problem, torch.zeros(2, dtype=DTYPE))

    assert result.converged
    assert not result.stable
    assert not result.reliable
    assert result.status == "unstable_stationary_point"


def test_rank_loss_and_iteration_limit_have_structured_failures():
    rank_lost = NonlinearEqualityProblem(
        n_state=2,
        n_constraints=1,
        energy_function=lambda state: 0.5 * state.square().sum(),
        constraint_function=lambda state: torch.stack((state @ state - 1.0,)),
    )
    rank_result = solve_newton_kkt(rank_lost, torch.zeros(2, dtype=DTYPE))
    assert rank_result.status == "rank_deficient_jacobian"
    assert not rank_result.reliable

    limited = NonlinearEqualityProblem(
        n_state=1,
        n_constraints=0,
        energy_function=lambda state: 0.25 * state[0].pow(4) + state[0],
        constraint_function=lambda state: state.new_zeros((0,)),
    )
    limited_result = solve_newton_kkt(
        limited,
        torch.tensor([2.0], dtype=DTYPE),
        options=NewtonKKTOptions(max_iterations=0),
    )
    assert limited_result.status == "max_iterations"
    assert not limited_result.reliable


def test_nonfinite_functional_is_a_structured_runtime_failure():
    problem = NonlinearEqualityProblem(
        n_state=1,
        n_constraints=0,
        energy_function=lambda state: state[0] * state.new_tensor(float("nan")),
        constraint_function=lambda state: state.new_zeros((0,)),
    )
    result = solve_newton_kkt(problem, torch.zeros(1, dtype=DTYPE))

    assert result.status == "nonfinite_residual"
    assert not result.reliable


def test_residual_line_search_backtracks_and_reports_rejected_failure():
    problem = NonlinearEqualityProblem(
        n_state=1,
        n_constraints=0,
        energy_function=lambda state: 0.25 * state[0].pow(4) + state[0],
        constraint_function=lambda state: state.new_zeros((0,)),
    )
    initial = torch.tensor([2.0], dtype=DTYPE)
    converged = solve_newton_kkt(problem, initial)
    failed = solve_newton_kkt(
        problem,
        initial,
        options=NewtonKKTOptions(max_backtracks=0),
    )

    assert converged.reliable
    assert converged.backtracks >= 1
    torch.testing.assert_close(
        converged.state, torch.tensor([-1.0], dtype=DTYPE), atol=2.0e-9, rtol=0.0
    )
    assert failed.status == "line_search_failed"
    assert failed.backtracks >= 1
    assert not failed.reliable


def test_large_hessian_scale_cannot_hide_a_unit_stationarity_residual():
    problem = NonlinearEqualityProblem(
        n_state=1,
        n_constraints=0,
        energy_function=lambda state: 0.5e20 * state[0].square() + state[0],
        constraint_function=lambda state: state.new_zeros((0,)),
    )
    result = solve_newton_kkt(
        problem,
        torch.zeros(1, dtype=DTYPE),
        options=NewtonKKTOptions(max_iterations=0),
    )

    assert result.residuals.stationarity_norm == 1.0
    assert result.status == "max_iterations"
    assert not result.reliable


def test_rank_deficient_jacobian_cannot_pass_from_an_exact_residual():
    problem = NonlinearEqualityProblem(
        n_state=2,
        n_constraints=2,
        energy_function=lambda state: 0.5 * state.square().sum(),
        constraint_function=lambda state: torch.stack((state[0], state[0])),
    )
    result = solve_newton_kkt(problem, torch.zeros(2, dtype=DTYPE))

    assert result.residuals.maximum_norm == 0.0
    assert result.status == "rank_deficient_jacobian"
    assert not result.converged
    assert not result.reliable
