"""Dense float64-quality oracle for small equality-constrained problems."""

from typing import Optional

import torch

from ..functional import QuadraticProblem
from ..kkt import assemble_kkt, evaluate_kkt, kkt_right_hand_side
from ..result import VariationalResult
from ..validation import SingularKKTError, validate_problem


def _make_result(
    problem: QuadraticProblem,
    state: torch.Tensor,
    multipliers: torch.Tensor,
    curvature: torch.Tensor,
    residual_tolerance: Optional[float],
) -> VariationalResult:
    residuals = evaluate_kkt(problem, state, multipliers)
    if residual_tolerance is None:
        residual_tolerance = 1000.0 * torch.finfo(problem.hessian.dtype).eps
    stationarity_ok = residuals.stationarity_norm <= residual_tolerance * (
        1.0 + residuals.stationarity_scale
    )
    primal_ok = residuals.primal_norm <= residual_tolerance * (
        1.0 + residuals.primal_scale
    )
    curvature_scale = max(
        1.0,
        float(problem.hessian.detach().abs().max().item()),
    )
    stable = curvature > (
        100.0
        * torch.finfo(problem.hessian.dtype).eps
        * max(problem.hessian.shape)
        * curvature_scale
    )
    condition_number = torch.linalg.cond(assemble_kkt(problem).detach())
    return VariationalResult(
        state=state,
        multipliers=multipliers,
        energy=problem.energy(state),
        residuals=residuals,
        condition_number=condition_number,
        minimum_reduced_curvature=curvature,
        reliable=bool((stationarity_ok & primal_ok & stable).item()),
    )


def solve_dense(
    problem: QuadraticProblem,
    residual_tolerance: Optional[float] = None,
    validation_tolerance: Optional[float] = None,
    require_stable: bool = True,
) -> VariationalResult:
    """Solve a quadratic problem and return a residual-certified result."""

    curvature = validate_problem(
        problem,
        tolerance=validation_tolerance,
        require_stable=require_stable,
    )
    kkt = assemble_kkt(problem)
    try:
        solution = torch.linalg.solve(kkt, kkt_right_hand_side(problem))
    except RuntimeError as error:
        raise SingularKKTError("dense KKT solve failed") from error

    state = solution[: problem.n_state]
    multipliers = solution[problem.n_state :]
    return _make_result(
        problem,
        state,
        multipliers,
        curvature,
        residual_tolerance,
    )


def solve_schur(
    problem: QuadraticProblem,
    residual_tolerance: Optional[float] = None,
    validation_tolerance: Optional[float] = None,
    require_stable: bool = True,
) -> VariationalResult:
    """Solve through the Hessian Schur complement.

    This is an independent dense reference path.  Unlike ``solve_dense``, it
    additionally requires the full Hessian to be nonsingular; constrained
    problems whose constraints remove a Hessian zero mode must use the direct
    KKT solve.
    """

    curvature = validate_problem(
        problem,
        tolerance=validation_tolerance,
        require_stable=require_stable,
    )
    try:
        unconstrained = torch.linalg.solve(problem.hessian, -problem.linear)
        if problem.n_constraints:
            inverse_hessian_at = torch.linalg.solve(
                problem.hessian,
                problem.constraint_matrix.transpose(0, 1),
            )
            schur = problem.constraint_matrix @ inverse_hessian_at
            multipliers = torch.linalg.solve(
                schur,
                problem.constraint_matrix @ unconstrained
                - problem.constraint_target,
            )
            state = unconstrained - inverse_hessian_at @ multipliers
        else:
            state = unconstrained
            multipliers = state.new_zeros((0,))
    except RuntimeError as error:
        raise SingularKKTError("dense Schur solve failed") from error
    return _make_result(
        problem,
        state,
        multipliers,
        curvature,
        residual_tolerance,
    )
