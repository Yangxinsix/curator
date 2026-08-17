"""Residual-certified amortized solves for quadratic variational problems.

The state machine implemented here is deliberately small and explicit:

``proposal -> hard projection -> certification -> KKT correction -> fallback``.

Every returned state is feasible to a scale-aware, near-machine-precision
tolerance.  Stationarity and feasibility are certified independently, so a
loose stationarity tolerance can never turn an infeasible state into a reliable
one.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from .backends.dense import solve_dense
from .functional import QuadraticProblem
from .kkt import evaluate_kkt
from .projection import project_to_equalities
from .response import solve_linear_response
from .result import KKTResiduals
from .validation import SingularKKTError, validate_problem


@dataclass(frozen=True)
class AmortizedResult:
    """Observable outcome of the amortized certification state machine."""

    state: torch.Tensor
    multipliers: torch.Tensor
    energy: torch.Tensor
    residuals: KKTResiduals
    projected_residuals: KKTResiduals
    projection_residual: torch.Tensor
    status: str
    iterations: int
    reliable: bool
    used_fallback: bool
    fallback_reason: Optional[str]
    failure_reason: Optional[str]


def _validate_inputs(
    problem: QuadraticProblem,
    proposal: torch.Tensor,
    projection_metric: Optional[torch.Tensor],
    residual_tolerance: float,
    feasibility_tolerance: Optional[float],
    max_correction_steps: int,
) -> float:
    validate_problem(problem)
    if proposal.shape != (problem.n_state,):
        raise ValueError("proposal must have shape (n_state,)")
    if proposal.device != problem.hessian.device:
        raise ValueError("proposal and problem must be on the same device")
    if proposal.dtype != problem.hessian.dtype:
        raise ValueError("proposal and problem must use the same dtype")
    if projection_metric is not None:
        if projection_metric.device != problem.hessian.device:
            raise ValueError("projection metric and problem must be on the same device")
        if projection_metric.dtype != problem.hessian.dtype:
            raise ValueError("projection metric and problem must use the same dtype")
    if not isinstance(max_correction_steps, int) or isinstance(
        max_correction_steps, bool
    ):
        raise ValueError("max_correction_steps must be a non-negative integer")
    if max_correction_steps < 0:
        raise ValueError("max_correction_steps must be a non-negative integer")
    if residual_tolerance < 0.0 or not torch.isfinite(
        problem.hessian.new_tensor(residual_tolerance)
    ):
        raise ValueError("residual_tolerance must be finite and non-negative")
    if feasibility_tolerance is None:
        feasibility_tolerance = 1000.0 * torch.finfo(problem.hessian.dtype).eps
    if feasibility_tolerance < 0.0 or not torch.isfinite(
        problem.hessian.new_tensor(feasibility_tolerance)
    ):
        raise ValueError("feasibility_tolerance must be finite and non-negative")
    return float(feasibility_tolerance)


def _estimate_multipliers(
    problem: QuadraticProblem,
    state: torch.Tensor,
) -> torch.Tensor:
    """Least-squares multiplier for the physical KKT stationarity residual."""

    if problem.n_constraints == 0:
        return state.new_zeros((0,))
    constraints = problem.constraint_matrix
    gram = constraints @ constraints.transpose(0, 1)
    return torch.linalg.solve(gram, -(constraints @ problem.gradient(state)))


def _certified(
    residuals: KKTResiduals,
    residual_tolerance: float,
    feasibility_tolerance: float,
) -> bool:
    stationarity_ok = (
        residuals.normalized_stationarity_norm <= residual_tolerance
    )
    feasibility_ok = residuals.normalized_primal_norm <= feasibility_tolerance
    return bool((stationarity_ok & feasibility_ok).item())


def _strictly_feasible(
    residuals: KKTResiduals,
    feasibility_tolerance: float,
) -> bool:
    return bool(
        (residuals.normalized_primal_norm <= feasibility_tolerance).item()
    )


def _result(
    problem: QuadraticProblem,
    state: torch.Tensor,
    multipliers: torch.Tensor,
    residuals: KKTResiduals,
    projected_residuals: KKTResiduals,
    projection_residual: torch.Tensor,
    status: str,
    iterations: int,
    reliable: bool,
    used_fallback: bool,
    fallback_reason: Optional[str] = None,
    failure_reason: Optional[str] = None,
) -> AmortizedResult:
    return AmortizedResult(
        state=state,
        multipliers=multipliers,
        energy=problem.energy(state),
        residuals=residuals,
        projected_residuals=projected_residuals,
        projection_residual=projection_residual,
        status=status,
        iterations=iterations,
        reliable=reliable,
        used_fallback=used_fallback,
        fallback_reason=fallback_reason,
        failure_reason=failure_reason,
    )


def solve_amortized(
    problem: QuadraticProblem,
    proposal: torch.Tensor,
    projection_metric: Optional[torch.Tensor] = None,
    residual_tolerance: float = 1.0e-8,
    feasibility_tolerance: Optional[float] = None,
    max_correction_steps: int = 1,
    fallback: bool = True,
) -> AmortizedResult:
    """Project, certify, correct, and optionally fall back to an exact solve.

    ``iterations`` counts attempted KKT correction steps; an exact fallback is
    intentionally not counted as a correction.  For a quadratic problem, one
    successful KKT correction reaches the exact stationary point up to floating
    point roundoff.
    """

    feasibility_tolerance = _validate_inputs(
        problem,
        proposal,
        projection_metric,
        residual_tolerance,
        feasibility_tolerance,
        max_correction_steps,
    )
    projection = project_to_equalities(
        proposal,
        problem.constraint_matrix,
        problem.constraint_target,
        metric=projection_metric,
    )
    state = projection.state
    multipliers = _estimate_multipliers(problem, state)
    residuals = evaluate_kkt(problem, state, multipliers)
    projected_residuals = residuals
    if not _strictly_feasible(residuals, feasibility_tolerance):
        raise RuntimeError("hard projection did not produce a feasible state")
    if _certified(residuals, residual_tolerance, feasibility_tolerance):
        return _result(
            problem,
            state,
            multipliers,
            residuals,
            projected_residuals,
            projection.residual,
            status="accepted_projected",
            iterations=0,
            reliable=True,
            used_fallback=False,
        )

    iterations = 0
    correction_failure = None
    for _ in range(max_correction_steps):
        iterations += 1
        try:
            correction = solve_linear_response(
                problem,
                stationarity_rhs=-residuals.stationarity,
                constraint_rhs=-residuals.primal,
            )
        except SingularKKTError as error:
            correction_failure = str(error)
            break
        state = state + correction.state
        # Reprojection makes feasibility an invariant even for an ill-conditioned
        # KKT correction.  The physical multiplier is then recomputed rather than
        # confused with the metric-projection multiplier.
        state = project_to_equalities(
            state,
            problem.constraint_matrix,
            problem.constraint_target,
            metric=projection_metric,
        ).state
        multipliers = _estimate_multipliers(problem, state)
        residuals = evaluate_kkt(problem, state, multipliers)
        if not _strictly_feasible(residuals, feasibility_tolerance):
            correction_failure = "KKT correction did not preserve feasibility"
            break
        if _certified(residuals, residual_tolerance, feasibility_tolerance):
            return _result(
                problem,
                state,
                multipliers,
                residuals,
                projected_residuals,
                projection.residual,
                status="corrected",
                iterations=iterations,
                reliable=True,
                used_fallback=False,
            )

    if fallback:
        fallback_reason = correction_failure or (
            "projected proposal was not certified within the correction budget"
        )
        try:
            exact = solve_dense(problem, residual_tolerance=residual_tolerance)
        except RuntimeError as error:
            return _result(
                problem,
                state,
                multipliers,
                residuals,
                projected_residuals,
                projection.residual,
                status="failed",
                iterations=iterations,
                reliable=False,
                used_fallback=True,
                fallback_reason=fallback_reason,
                failure_reason="exact fallback failed: %s" % error,
            )
        exact_certified = _certified(
            exact.residuals,
            residual_tolerance,
            feasibility_tolerance,
        )
        if not _strictly_feasible(exact.residuals, feasibility_tolerance):
            return _result(
                problem,
                state,
                multipliers,
                residuals,
                projected_residuals,
                projection.residual,
                status="failed",
                iterations=iterations,
                reliable=False,
                used_fallback=True,
                fallback_reason=fallback_reason,
                failure_reason="exact fallback did not produce a feasible state",
            )
        return _result(
            problem,
            exact.state,
            exact.multipliers,
            exact.residuals,
            projected_residuals,
            projection.residual,
            status="fallback",
            iterations=iterations,
            reliable=exact_certified,
            used_fallback=True,
            fallback_reason=fallback_reason,
            failure_reason=None if exact_certified else "exact fallback was not certified",
        )

    reason = correction_failure or (
        "projected proposal was not certified and exact fallback is disabled"
    )
    return _result(
        problem,
        state,
        multipliers,
        residuals,
        projected_residuals,
        projection.residual,
        status="failed",
        iterations=iterations,
        reliable=False,
        used_fallback=False,
        failure_reason=reason,
    )
