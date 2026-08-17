"""Adjoint-differentiable dense KKT solve for quadratic reference problems."""

from typing import Optional, Tuple

import torch

from .functional import QuadraticProblem
from .kkt import assemble_kkt, evaluate_kkt, kkt_right_hand_side
from .result import VariationalResult
from .validation import SingularKKTError, validate_problem


class _ImplicitKKTSolve(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hessian: torch.Tensor,
        linear: torch.Tensor,
        constraint_matrix: torch.Tensor,
        constraint_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_state = hessian.shape[0]
        n_constraints = constraint_matrix.shape[0]
        kkt = torch.cat(
            (
                torch.cat((hessian, constraint_matrix.transpose(0, 1)), dim=1),
                torch.cat(
                    (
                        constraint_matrix,
                        hessian.new_zeros((n_constraints, n_constraints)),
                    ),
                    dim=1,
                ),
            ),
            dim=0,
        )
        right = torch.cat((-linear, constraint_target), dim=0)
        try:
            solution = torch.linalg.solve(kkt, right)
        except RuntimeError as error:
            raise SingularKKTError("implicit KKT solve failed") from error
        state = solution[:n_state]
        multipliers = solution[n_state:]
        ctx.save_for_backward(hessian, constraint_matrix, state, multipliers)
        return state, multipliers

    @staticmethod
    def backward(ctx, state_cotangent, multiplier_cotangent):
        hessian, constraints, state, multipliers = ctx.saved_tensors
        if state_cotangent is None:
            state_cotangent = torch.zeros_like(state)
        if multiplier_cotangent is None:
            multiplier_cotangent = torch.zeros_like(multipliers)
        n_constraints = constraints.shape[0]
        kkt = torch.cat(
            (
                torch.cat((hessian, constraints.transpose(0, 1)), dim=1),
                torch.cat(
                    (
                        constraints,
                        hessian.new_zeros((n_constraints, n_constraints)),
                    ),
                    dim=1,
                ),
            ),
            dim=0,
        )
        cotangent = torch.cat((state_cotangent, multiplier_cotangent), dim=0)
        adjoint = torch.linalg.solve(kkt.transpose(0, 1), cotangent)
        state_adjoint = adjoint[: state.numel()]
        multiplier_adjoint = adjoint[state.numel() :]
        hessian_gradient = -torch.outer(state_adjoint, state)
        linear_gradient = -state_adjoint
        constraint_gradient = -(
            torch.outer(multipliers, state_adjoint)
            + torch.outer(multiplier_adjoint, state)
        )
        target_gradient = multiplier_adjoint
        return (
            hessian_gradient,
            linear_gradient,
            constraint_gradient,
            target_gradient,
        )


def implicit_kkt_state(problem: QuadraticProblem) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return state and multipliers with a custom adjoint KKT backward."""

    validate_problem(problem)
    return _ImplicitKKTSolve.apply(
        problem.hessian,
        problem.linear,
        problem.constraint_matrix,
        problem.constraint_target,
    )


def solve_implicit(
    problem: QuadraticProblem,
    residual_tolerance: Optional[float] = None,
    validation_tolerance: Optional[float] = None,
    require_stable: bool = True,
) -> VariationalResult:
    """Solve a dense quadratic while differentiating by the adjoint KKT rule."""

    curvature = validate_problem(
        problem,
        tolerance=validation_tolerance,
        require_stable=require_stable,
    )
    state, multipliers = _ImplicitKKTSolve.apply(
        problem.hessian,
        problem.linear,
        problem.constraint_matrix,
        problem.constraint_target,
    )
    residuals = evaluate_kkt(problem, state, multipliers)
    if residual_tolerance is None:
        residual_tolerance = 1000.0 * torch.finfo(problem.hessian.dtype).eps
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
    reliable = bool(
        (
            residuals.normalized_stationarity_norm <= residual_tolerance
        ).item()
        and (residuals.normalized_primal_norm <= residual_tolerance).item()
        and stable.item()
    )
    return VariationalResult(
        state=state,
        multipliers=multipliers,
        energy=problem.energy(state),
        residuals=residuals,
        condition_number=torch.linalg.cond(assemble_kkt(problem).detach()),
        minimum_reduced_curvature=curvature,
        reliable=reliable,
    )


__all__ = ["implicit_kkt_state", "solve_implicit"]
