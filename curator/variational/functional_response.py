"""Optional linearized variational response using Hessian-vector products.

This solver is separate from proposal training. It solves the KKT linearization
at a supplied state; its interpretation as a relaxed physical response requires
a converged minimum and the correct perturbation derivatives on the right-hand
side. No dense state Hessian, inverse or null-space basis is constructed.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from .functionals import FunctionalProblem


@dataclass(frozen=True)
class FunctionalLinearResponse:
    state: torch.Tensor
    multipliers: torch.Tensor
    residual_norm: torch.Tensor
    iterations: int
    converged: bool
    status: str


def solve_functional_linear_response(
    problem: FunctionalProblem,
    state: torch.Tensor,
    stationarity_rhs: torch.Tensor,
    constraint_rhs: Optional[torch.Tensor] = None,
    *,
    tolerance: float = 1.0e-10,
    max_iterations: Optional[int] = None,
    create_graph: bool = False,
) -> FunctionalLinearResponse:
    """Solve ``H dx + A.T dlambda = rhs`` and ``A dx = constraint_rhs``.

    ``rhs`` is a vector or a matrix of right-hand sides. For a perturbation p,
    callers supply minus the partial p derivative of the stationary Lagrangian
    gradient, and minus the partial p derivative of the constraint residual.
    The equality-constrained Hessian must be positive definite. Projected CG
    reports failure on nonpositive curvature or exhausted iterations.

    ``create_graph`` differentiates the executed iterative operations, not the
    stopping decision; default evaluation returns detached numerical results.
    """

    problem._validate_state(state)
    if not isinstance(stationarity_rhs, torch.Tensor) or stationarity_rhs.ndim not in (1, 2):
        raise ValueError("stationarity_rhs must be a vector or matrix")
    vector_input = stationarity_rhs.ndim == 1
    rhs = stationarity_rhs[:, None] if vector_input else stationarity_rhs
    if rhs.shape[0] != problem.n_state or rhs.shape[1] < 1:
        raise ValueError("stationarity_rhs must have n_state rows and at least one column")
    if rhs.dtype != state.dtype or rhs.device != state.device or not torch.isfinite(rhs).all():
        raise ValueError("stationarity_rhs must be finite and share state dtype/device")
    if not torch.isfinite(torch.tensor(tolerance)) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    limit = max(20, 2 * problem.n_state) if max_iterations is None else max_iterations
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("max_iterations must be a positive integer")
    if constraint_rhs is None:
        equality_rhs = state.new_zeros((problem.n_constraints, rhs.shape[1]))
    else:
        equality_rhs = constraint_rhs[:, None] if vector_input and constraint_rhs.ndim == 1 else constraint_rhs
        if equality_rhs.shape != (problem.n_constraints, rhs.shape[1]):
            raise ValueError("constraint_rhs shape must match constraints and response columns")
        if equality_rhs.dtype != state.dtype or equality_rhs.device != state.device or not torch.isfinite(equality_rhs).all():
            raise ValueError("constraint_rhs must be finite and share state dtype/device")

    matrix = problem.constraint_matrix
    gram = matrix @ matrix.T if problem.n_constraints else None

    def tangent(value):
        if gram is None:
            return value
        return value - matrix.T @ torch.linalg.solve(gram, matrix @ value)

    def action(value):
        return problem.hvp(state, value, create_graph=create_graph)

    states, multipliers, residuals = [], [], []
    iterations, converged, status = 0, True, "converged"
    # Below floating-point resolution an absolute requested tolerance cannot be
    # meaningfully checked. Float64 retains the user's default 1e-10 tolerance.
    threshold_scale = max(float(tolerance), 10 * torch.finfo(state.dtype).eps)
    for column in range(rhs.shape[1]):
        forcing, equality = rhs[:, column], equality_rhs[:, column]
        particular = (matrix.T @ torch.linalg.solve(gram, equality)
                      if gram is not None else torch.zeros_like(state))
        residual = tangent(forcing - action(particular))
        direction = residual
        solution = torch.zeros_like(state)
        squared = residual @ residual
        threshold = threshold_scale * (1 + torch.linalg.vector_norm(forcing)
                                         + torch.linalg.vector_norm(equality))
        count = 0
        while float(squared.detach()) > float(threshold.detach().square()) and count < limit:
            product = tangent(action(direction))
            curvature = direction @ product
            if not torch.isfinite(curvature) or float(curvature.detach()) <= 0:
                status = "nonpositive_curvature"
                break
            alpha = squared / curvature
            solution = tangent(solution + alpha * direction)
            residual = tangent(residual - alpha * product)
            new_squared = residual @ residual
            direction = tangent(residual + (new_squared / squared) * direction)
            squared = new_squared
            count += 1
        response = particular + solution
        remaining = forcing - action(response)
        multiplier = (torch.linalg.solve(gram, matrix @ remaining)
                      if gram is not None else state.new_zeros(0))
        stationary = remaining - (matrix.T @ multiplier if gram is not None else 0)
        feasible = matrix @ response - equality if gram is not None else state.new_zeros(0)
        norm = torch.linalg.vector_norm(torch.cat((stationary, feasible)))
        passed = float(norm.detach()) <= float(threshold.detach())
        converged = converged and passed
        if not passed and status == "converged":
            status = "max_iterations"
        states.append(response)
        multipliers.append(multiplier)
        residuals.append(norm)
        iterations = max(iterations, count)
    values = (torch.stack(states, dim=1), torch.stack(multipliers, dim=1), torch.stack(residuals))
    if not create_graph:
        values = tuple(value.detach() for value in values)
    responses, lambdas, norms = values
    if vector_input:
        responses, lambdas, norms = responses[:, 0], lambdas[:, 0], norms[0]
    return FunctionalLinearResponse(responses, lambdas, norms, iterations, converged, status)
