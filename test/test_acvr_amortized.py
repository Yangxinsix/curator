"""Tests for the project/certify/correct/fallback quadratic state machine."""

import pytest
import torch

from curator.variational.amortized import solve_amortized
from curator.variational.backends.dense import solve_dense
from curator.variational.functional import QuadraticProblem


DTYPE = torch.float64
ATOL = 2.0e-11
RESIDUAL_TOLERANCE = 1.0e-10


def _problem():
    hessian = torch.tensor(
        [
            [3.0, 0.2, -0.1, 0.0],
            [0.2, 2.5, 0.3, 0.1],
            [-0.1, 0.3, 2.0, 0.2],
            [0.0, 0.1, 0.2, 1.75],
        ],
        dtype=DTYPE,
    )
    return QuadraticProblem(
        hessian=hessian,
        linear=torch.tensor([0.3, -0.5, 0.8, -0.2], dtype=DTYPE),
        constraint_matrix=torch.ones((1, 4), dtype=DTYPE),
        constraint_target=torch.tensor([0.65], dtype=DTYPE),
        constant=torch.tensor(0.4, dtype=DTYPE),
    )


def _assert_strictly_feasible(problem, result):
    torch.testing.assert_close(
        problem.constraint_matrix @ result.state,
        problem.constraint_target,
        atol=ATOL,
        rtol=0.0,
    )
    assert result.residuals.normalized_primal_norm <= 1000.0 * torch.finfo(DTYPE).eps


def _assert_matches_exact(problem, result):
    exact = solve_dense(problem)
    torch.testing.assert_close(result.state, exact.state, atol=ATOL, rtol=0.0)
    torch.testing.assert_close(
        result.multipliers,
        exact.multipliers,
        atol=ATOL,
        rtol=0.0,
    )
    torch.testing.assert_close(result.energy, exact.energy, atol=ATOL, rtol=0.0)
    assert result.residuals.maximum_normalized_norm <= RESIDUAL_TOLERANCE
    _assert_strictly_feasible(problem, result)


def test_projected_proposal_can_be_accepted_without_a_correction_step():
    problem = _problem()
    exact = solve_dense(problem)
    inverse_hessian_at = torch.linalg.solve(
        problem.hessian,
        problem.constraint_matrix.T,
    )
    # This differs from the optimum only along the H-normal to the feasible
    # surface, so an H-metric projection must recover the exact state.
    proposal = exact.state + 0.7 * inverse_hessian_at[:, 0]

    result = solve_amortized(
        problem,
        proposal,
        projection_metric=problem.hessian,
        residual_tolerance=RESIDUAL_TOLERANCE,
    )

    assert result.status == "accepted_projected"
    assert result.iterations == 0
    assert result.reliable
    assert not result.used_fallback
    assert result.failure_reason is None
    assert result.projection_residual.abs().max() <= ATOL
    _assert_matches_exact(problem, result)


def test_one_kkt_correction_reaches_the_exact_quadratic_solution():
    problem = _problem()
    exact = solve_dense(problem)
    tangent_error = torch.tensor([0.7, -0.7, 0.25, -0.25], dtype=DTYPE)
    proposal = exact.state + tangent_error

    result = solve_amortized(
        problem,
        proposal,
        residual_tolerance=RESIDUAL_TOLERANCE,
        max_correction_steps=3,
        fallback=False,
    )

    assert result.status == "corrected"
    assert result.iterations == 1
    assert result.reliable
    assert not result.used_fallback
    assert result.projected_residuals.normalized_stationarity_norm > RESIDUAL_TOLERANCE
    _assert_matches_exact(problem, result)


def test_zero_correction_steps_use_the_exact_fallback_when_enabled():
    problem = _problem()
    exact = solve_dense(problem)
    proposal = exact.state + torch.tensor([0.8, -0.8, 0.0, 0.0], dtype=DTYPE)

    result = solve_amortized(
        problem,
        proposal,
        residual_tolerance=RESIDUAL_TOLERANCE,
        max_correction_steps=0,
        fallback=True,
    )

    assert result.status == "fallback"
    assert result.iterations == 0
    assert result.reliable
    assert result.used_fallback
    assert "correction budget" in result.fallback_reason
    assert result.projected_residuals.normalized_stationarity_norm > RESIDUAL_TOLERANCE
    _assert_matches_exact(problem, result)


def test_zero_step_without_fallback_returns_an_explicit_unreliable_failure():
    problem = _problem()
    exact = solve_dense(problem)
    proposal = exact.state + torch.tensor([0.8, -0.8, 0.0, 0.0], dtype=DTYPE)

    result = solve_amortized(
        problem,
        proposal,
        residual_tolerance=RESIDUAL_TOLERANCE,
        max_correction_steps=0,
        fallback=False,
    )

    assert result.status == "failed"
    assert result.iterations == 0
    assert not result.reliable
    assert not result.used_fallback
    assert "fallback is disabled" in result.failure_reason
    assert result.residuals.normalized_stationarity_norm > RESIDUAL_TOLERANCE
    _assert_strictly_feasible(problem, result)


def test_hard_feasibility_is_independent_of_a_loose_stationarity_tolerance():
    problem = _problem()
    proposal = torch.tensor([50.0, -30.0, 20.0, 10.0], dtype=DTYPE)

    result = solve_amortized(
        problem,
        proposal,
        residual_tolerance=1.0e6,
        max_correction_steps=0,
        fallback=False,
    )

    assert result.status == "accepted_projected"
    assert result.reliable
    _assert_strictly_feasible(problem, result)


@pytest.mark.parametrize(
    "proposal, metric, message",
    [
        (
            torch.zeros(3, dtype=DTYPE),
            None,
            "proposal.*shape",
        ),
        (
            torch.zeros(4, dtype=DTYPE),
            torch.ones(3, dtype=DTYPE),
            "metric.*shape",
        ),
        (
            torch.zeros(4, dtype=DTYPE),
            torch.diag(torch.tensor([1.0, 1.0, -1.0, 1.0], dtype=DTYPE)),
            "positive definite",
        ),
        (
            torch.zeros(4, dtype=torch.float32),
            None,
            "same dtype",
        ),
    ],
    ids=["proposal-shape", "metric-shape", "metric-not-spd", "proposal-dtype"],
)
def test_invalid_proposal_or_projection_metric_is_rejected(proposal, metric, message):
    with pytest.raises(ValueError, match=message):
        solve_amortized(_problem(), proposal, projection_metric=metric)


@pytest.mark.parametrize("steps", [-1, 1.5, True])
def test_invalid_correction_step_count_is_rejected(steps):
    with pytest.raises(ValueError, match="non-negative integer"):
        solve_amortized(
            _problem(),
            torch.zeros(4, dtype=DTYPE),
            max_correction_steps=steps,
        )
