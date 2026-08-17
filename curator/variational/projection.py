"""Hard projection of amortized predictions onto equality constraints."""

from typing import Optional

import torch

from .result import ProjectionResult
from .validation import SingularKKTError, validate_equalities


def _solve_metric(metric: Optional[torch.Tensor], right: torch.Tensor) -> torch.Tensor:
    if metric is None:
        return right
    if metric.ndim == 1:
        if metric.shape != (right.shape[0],):
            raise ValueError("diagonal metric has the wrong shape")
        if bool((metric <= 0.0).any().item()):
            raise ValueError("diagonal metric entries must be positive")
        return right / metric[:, None]
    if metric.ndim == 2 and metric.shape == (right.shape[0], right.shape[0]):
        scale = float(metric.detach().abs().max().item())
        tolerance = (
            100.0
            * torch.finfo(metric.dtype).eps
            * max(metric.shape)
            * max(scale, torch.finfo(metric.dtype).tiny)
        )
        if not torch.allclose(
            metric,
            metric.transpose(0, 1),
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("dense metric must be symmetric")
        try:
            cholesky = torch.linalg.cholesky(metric)
        except RuntimeError as error:
            raise ValueError("dense metric must be positive definite") from error
        return torch.cholesky_solve(right, cholesky)
    raise ValueError("metric must be None, a positive diagonal, or a square matrix")


def project_to_equalities(
    state: torch.Tensor,
    constraint_matrix: torch.Tensor,
    constraint_target: torch.Tensor,
    metric: Optional[torch.Tensor] = None,
) -> ProjectionResult:
    r"""Find the closest feasible state in the supplied positive metric.

    This solves

    .. math::

        \min_z \tfrac12(z-x)^T W(z-x),\qquad A z=b.
    """

    if state.ndim != 1:
        raise ValueError("state must be rank 1")
    if constraint_matrix.shape[1:] != (state.shape[0],):
        raise ValueError("constraint_matrix has the wrong shape")
    validate_equalities(constraint_matrix, constraint_target)
    if constraint_matrix.shape[0] == 0:
        return ProjectionResult(
            state=state,
            multipliers=state.new_zeros((0,)),
            residual=state.new_zeros((0,)),
        )

    inverse_metric_at = _solve_metric(metric, constraint_matrix.transpose(0, 1))
    schur = constraint_matrix @ inverse_metric_at
    violation = constraint_matrix @ state - constraint_target
    try:
        multipliers = torch.linalg.solve(schur, violation)
    except RuntimeError as error:
        raise SingularKKTError("constraint projection failed") from error
    projected = state - inverse_metric_at @ multipliers
    residual = constraint_matrix @ projected - constraint_target
    return ProjectionResult(
        state=projected,
        multipliers=multipliers,
        residual=residual,
    )
