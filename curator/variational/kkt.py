"""KKT assembly and residual certification."""

import torch

from .functional import QuadraticProblem
from .result import KKTResiduals


def assemble_kkt(problem: QuadraticProblem) -> torch.Tensor:
    """Assemble ``[[H, A.T], [A, 0]]`` for the frozen sign convention."""

    hessian = problem.hessian
    constraints = problem.constraint_matrix
    n_constraints = problem.n_constraints
    return torch.cat(
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


def kkt_right_hand_side(problem: QuadraticProblem) -> torch.Tensor:
    return torch.cat((-problem.linear, problem.constraint_target), dim=0)


def _infinity_norm(vector: torch.Tensor) -> torch.Tensor:
    if vector.numel() == 0:
        return vector.new_zeros(())
    return vector.abs().max()


def evaluate_kkt(
    problem: QuadraticProblem,
    state: torch.Tensor,
    multipliers: torch.Tensor,
) -> KKTResiduals:
    """Evaluate stationarity and primal feasibility residuals."""

    if state.shape != (problem.n_state,):
        raise ValueError("state has the wrong shape")
    if multipliers.shape != (problem.n_constraints,):
        raise ValueError("multipliers have the wrong shape")
    stationarity = (
        problem.gradient(state)
        + problem.constraint_matrix.transpose(0, 1) @ multipliers
    )
    primal = problem.constraint_matrix @ state - problem.constraint_target
    hessian_term = problem.hessian @ state
    constraint_term = problem.constraint_matrix.transpose(0, 1) @ multipliers
    stationarity_scale = torch.stack(
        (
            _infinity_norm(hessian_term),
            _infinity_norm(problem.linear),
            _infinity_norm(constraint_term),
            problem.hessian.new_tensor(torch.finfo(problem.hessian.dtype).tiny),
        )
    ).max()
    constraint_value = problem.constraint_matrix @ state
    primal_scale = torch.stack(
        (
            _infinity_norm(constraint_value),
            _infinity_norm(problem.constraint_target),
            problem.hessian.new_tensor(torch.finfo(problem.hessian.dtype).tiny),
        )
    ).max()
    stationarity_norm = _infinity_norm(stationarity)
    primal_norm = _infinity_norm(primal)
    return KKTResiduals(
        stationarity=stationarity,
        primal=primal,
        stationarity_norm=stationarity_norm,
        primal_norm=primal_norm,
        stationarity_scale=stationarity_scale,
        primal_scale=primal_scale,
        normalized_stationarity_norm=stationarity_norm / (1.0 + stationarity_scale),
        normalized_primal_norm=primal_norm / (1.0 + primal_scale),
    )
