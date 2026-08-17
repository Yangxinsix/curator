"""Matrix-free projected-CG solver for equality-constrained quadratics.

The solver uses the row-major ACVR convention ``A @ x = b``.  It never asks
the quadratic operator for a dense kernel: every Hessian action goes through
``operator.matvec``.  Equality matrices are expected to be small and dense.
"""

from dataclasses import dataclass
import math
from typing import Optional, Union

import torch

from ..operators import OperatorQuadraticProblem, VariationalOperator
from ..validation import SingularKKTError, validate_equalities


@dataclass(frozen=True)
class ProjectedCGResult:
    """One projected-CG solution and its convergence certificate."""

    state: torch.Tensor
    multipliers: torch.Tensor
    energy: torch.Tensor
    converged: bool
    reliable: bool
    reduced_stability_certified: bool
    iterations: int
    residual: torch.Tensor
    residual_norm: torch.Tensor
    primal_residual: torch.Tensor
    primal_residual_norm: torch.Tensor
    reason: str


def _infinity_norm(value: torch.Tensor) -> torch.Tensor:
    if value.numel() == 0:
        return value.new_zeros(())
    return value.abs().max()


def _checked_matvec(
    operator: VariationalOperator,
    state: torch.Tensor,
) -> torch.Tensor:
    value = operator.matvec(state)
    if value.shape != state.shape:
        raise ValueError("operator.matvec must preserve the state shape")
    if value.dtype != state.dtype or value.device != state.device:
        raise ValueError("operator.matvec must preserve state dtype and device")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError("operator.matvec returned non-finite values")
    return value


def solve_projected_cg(
    operator: VariationalOperator,
    linear: torch.Tensor,
    constraint_matrix: Optional[torch.Tensor] = None,
    constraint_target: Optional[torch.Tensor] = None,
    *,
    constant: Union[torch.Tensor, float] = 0.0,
    initial_state: Optional[torch.Tensor] = None,
    rtol: float = 1.0e-10,
    atol: Optional[float] = None,
    max_iter: Optional[int] = None,
    reduced_stability_certified: bool = False,
) -> ProjectedCGResult:
    r"""Minimize a quadratic with dense equality constraints using CG.

    The objective and constraints are

    .. math::

        \tfrac12 x^T Hx + g^T x + c, \qquad Ax=b.

    ``H`` is accessed only as ``operator.matvec(x)``.  The algorithm projects
    the initial state onto the feasible affine space, then applies CG to
    ``P H P`` using the full-space tangent projector

    .. math::

        P = I-A^T(AA^T)^{-1}A.

    Thus the full Hessian may be indefinite as long as it is positive definite
    on ``null(A)``. ``converged`` records a final, scale-aware stationarity
    certificate. ``reliable`` additionally requires feasibility and an
    explicit caller certificate that the reduced Hessian is positive definite.
    Matrix-free CG cannot discover an unexcited negative mode from
    stationarity alone.
    """

    if linear.ndim != 1 or not linear.is_floating_point():
        raise ValueError("linear must be a floating-point rank-1 tensor")
    if not bool(torch.isfinite(linear).all().item()):
        raise ValueError("linear must contain only finite values")
    n_state = linear.shape[0]
    if constraint_matrix is None:
        constraint_matrix = linear.new_zeros((0, n_state))
    if constraint_target is None:
        if constraint_matrix.shape[0] != 0:
            raise ValueError("constraint_target is required for nonempty constraints")
        constraint_target = linear.new_zeros((0,))
    if constraint_matrix.ndim != 2 or constraint_matrix.shape[1] != n_state:
        raise ValueError("constraint_matrix must have shape (n_constraints, n_state)")
    if constraint_matrix.dtype != linear.dtype or constraint_matrix.device != linear.device:
        raise ValueError("constraint_matrix must match linear dtype and device")
    if constraint_target.dtype != linear.dtype or constraint_target.device != linear.device:
        raise ValueError("constraint_target must match linear dtype and device")
    if not bool(torch.isfinite(constraint_matrix).all().item()) or not bool(
        torch.isfinite(constraint_target).all().item()
    ):
        raise ValueError("constraints must contain only finite values")
    validate_equalities(constraint_matrix, constraint_target)

    if initial_state is None:
        state = linear.new_zeros((n_state,))
    else:
        if initial_state.shape != (n_state,):
            raise ValueError("initial_state must have shape (n_state,)")
        if initial_state.dtype != linear.dtype or initial_state.device != linear.device:
            raise ValueError("initial_state must match linear dtype and device")
        state = initial_state.clone()
    if not bool(torch.isfinite(state).all().item()):
        raise ValueError("initial_state must contain only finite values")
    if not math.isfinite(rtol) or rtol < 0.0:
        raise ValueError("rtol must be finite and nonnegative")
    if atol is not None and (not math.isfinite(atol) or atol < 0.0):
        raise ValueError("atol must be finite and nonnegative")
    if max_iter is not None and (
        isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 0
    ):
        raise ValueError("max_iter must be a nonnegative integer")
    if not isinstance(reduced_stability_certified, bool):
        raise ValueError("reduced_stability_certified must be boolean")

    n_constraints = constraint_matrix.shape[0]
    if n_constraints:
        try:
            constraint_q, constraint_r = torch.linalg.qr(
                constraint_matrix.transpose(0, 1), mode="reduced"
            )
        except RuntimeError as error:
            raise SingularKKTError(
                "failed to factor the equality-constraint matrix"
            ) from error

        def project_tangent(value: torch.Tensor) -> torch.Tensor:
            return value - constraint_q @ (constraint_q.transpose(0, 1) @ value)

        state = state + constraint_q @ torch.linalg.solve(
            constraint_r.transpose(0, 1),
            constraint_target - constraint_matrix @ state,
        )
    else:
        def project_tangent(value: torch.Tensor) -> torch.Tensor:
            return value

    gradient = _checked_matvec(operator, state) + linear
    residual = -project_tangent(gradient)
    initial_residual_norm = torch.linalg.vector_norm(residual)
    if atol is None:
        atol = 1000.0 * torch.finfo(linear.dtype).eps * (
            1.0 + float(torch.linalg.vector_norm(linear.detach()).item())
        )
    iteration_threshold = linear.new_tensor(atol) + rtol * initial_residual_norm
    tangent_dimension = n_state - n_constraints
    if max_iter is None:
        max_iter = 10 * max(1, tangent_dimension)

    direction = residual.clone()
    residual_squared = torch.dot(residual, residual)
    converged = bool((initial_residual_norm <= iteration_threshold).item())
    reason = "converged" if converged else "maximum iterations reached"
    iterations = 0

    while not converged and iterations < max_iter:
        hessian_direction = project_tangent(
            _checked_matvec(operator, direction)
        )
        curvature = torch.dot(direction, hessian_direction)
        if not bool(torch.isfinite(curvature).item()) or bool((curvature <= 0.0).item()):
            reason = "non-positive projected curvature"
            break
        step = residual_squared / curvature
        state = state + step * direction
        next_residual = project_tangent(residual - step * hessian_direction)
        next_residual_squared = torch.dot(next_residual, next_residual)
        iterations += 1
        recursive_norm = torch.sqrt(torch.clamp_min(next_residual_squared, 0.0))
        if bool((recursive_norm <= iteration_threshold).item()):
            residual = next_residual
            residual_squared = next_residual_squared
            converged = True
            reason = "converged"
            break
        beta = next_residual_squared / residual_squared
        direction = project_tangent(next_residual + beta * direction)
        residual = next_residual
        residual_squared = next_residual_squared

    # Remove accumulated feasibility drift before constructing the KKT
    # certificate, then recompute the true (rather than recursive) residual.
    if n_constraints:
        state = state + constraint_q @ torch.linalg.solve(
            constraint_r.transpose(0, 1),
            constraint_target - constraint_matrix @ state,
        )
    hessian_state = _checked_matvec(operator, state)
    gradient = hessian_state + linear
    if n_constraints:
        multipliers = torch.linalg.solve(
            constraint_r,
            -(constraint_q.transpose(0, 1) @ gradient),
        )
    else:
        multipliers = linear.new_zeros((0,))
    residual = gradient + constraint_matrix.transpose(0, 1) @ multipliers
    primal_residual = constraint_matrix @ state - constraint_target
    residual_norm = torch.linalg.vector_norm(residual)
    primal_residual_norm = _infinity_norm(primal_residual)
    primal_scale = 1.0 + float(
        (
            _infinity_norm((constraint_matrix @ state).detach())
            + _infinity_norm(constraint_target.detach())
        ).item()
    )
    primal_threshold = 1000.0 * torch.finfo(linear.dtype).eps * primal_scale
    certification_rtol = max(rtol, 1000.0 * torch.finfo(linear.dtype).eps)
    stationarity_scale = (
        1.0
        + torch.linalg.vector_norm(hessian_state)
        + torch.linalg.vector_norm(linear)
    )
    certification_threshold = linear.new_tensor(atol) + (
        certification_rtol * stationarity_scale
    )
    actual_converged = bool((residual_norm <= certification_threshold).item())
    converged = converged and actual_converged
    if reason == "converged" and not actual_converged:
        reason = "recursive residual converged but true residual did not"
    reliable = (
        converged
        and bool((primal_residual_norm <= primal_threshold).item())
        and reduced_stability_certified
    )
    if converged and not reduced_stability_certified:
        reason = "stationary point found; reduced stability was not certified"

    energy = operator.energy(state) + linear @ state
    if isinstance(constant, torch.Tensor):
        if constant.numel() != 1:
            raise ValueError("constant must be scalar")
        if constant.dtype != linear.dtype or constant.device != linear.device:
            raise ValueError("constant must match linear dtype and device")
        energy = energy + constant.reshape(())
    else:
        energy = energy + linear.new_tensor(constant)
    if energy.numel() != 1:
        raise ValueError("operator.energy must return a scalar")
    if energy.dtype != linear.dtype or energy.device != linear.device:
        raise ValueError("operator.energy must preserve state dtype and device")
    if not bool(torch.isfinite(energy).item()):
        raise ValueError("operator.energy returned a non-finite value")

    return ProjectedCGResult(
        state=state,
        multipliers=multipliers,
        energy=energy.reshape(()),
        converged=converged,
        reliable=reliable,
        reduced_stability_certified=reduced_stability_certified,
        iterations=iterations,
        residual=residual,
        residual_norm=residual_norm,
        primal_residual=primal_residual,
        primal_residual_norm=primal_residual_norm,
        reason=reason,
    )


def solve_projected_cg_problem(
    problem: OperatorQuadraticProblem,
    **kwargs,
) -> ProjectedCGResult:
    """Solve a compiled operator problem without dropping boundary terms."""

    return solve_projected_cg(
        problem.operator,
        problem.linear,
        problem.constraint_matrix,
        problem.constraint_target,
        constant=problem.constant,
        **kwargs,
    )


__all__ = [
    "ProjectedCGResult",
    "solve_projected_cg",
    "solve_projected_cg_problem",
]
