"""Linearized response of equality-constrained quadratic optima."""

from typing import Optional

import torch

from .functional import QuadraticProblem
from .kkt import assemble_kkt
from .result import LinearResponse
from .validation import SingularKKTError, validate_problem


def solve_linear_response(
    problem: QuadraticProblem,
    stationarity_rhs: torch.Tensor,
    constraint_rhs: Optional[torch.Tensor] = None,
) -> LinearResponse:
    r"""Solve a linearized KKT equation for one or more perturbations.

    The caller supplies the right-hand side of

    .. math::

        \begin{bmatrix}H&A^T\\A&0\end{bmatrix}
        \begin{bmatrix}dx\\d\lambda\end{bmatrix}
        = \begin{bmatrix}r_x\\r_b\end{bmatrix}.

    For a control entering as ``-h.T @ B @ x``, ``r_x = B.T`` gives
    ``dx/dh``.  For an equality target, ``r_b = I`` gives ``dx/db``.
    """

    validate_problem(problem)
    if stationarity_rhs.ndim not in (1, 2):
        raise ValueError("stationarity_rhs must be a vector or matrix")
    if stationarity_rhs.shape[0] != problem.n_state:
        raise ValueError("stationarity_rhs has the wrong leading dimension")
    vector_rhs = stationarity_rhs.ndim == 1
    state_rhs = stationarity_rhs[:, None] if vector_rhs else stationarity_rhs
    n_rhs = state_rhs.shape[1]
    if constraint_rhs is None:
        lower_rhs = state_rhs.new_zeros((problem.n_constraints, n_rhs))
    else:
        lower_rhs = constraint_rhs[:, None] if constraint_rhs.ndim == 1 else constraint_rhs
        if lower_rhs.shape != (problem.n_constraints, n_rhs):
            raise ValueError("constraint_rhs has the wrong shape")

    rhs = torch.cat((state_rhs, lower_rhs), dim=0)
    kkt = assemble_kkt(problem)
    try:
        solution = torch.linalg.solve(kkt, rhs)
    except RuntimeError as error:
        raise SingularKKTError("linearized KKT solve failed") from error
    state = solution[: problem.n_state]
    multipliers = solution[problem.n_state :]
    residual = kkt @ solution - rhs
    stationarity_residual = residual[: problem.n_state]
    constraint_residual = residual[problem.n_state :]
    if vector_rhs:
        state = state[:, 0]
        multipliers = multipliers[:, 0]
        stationarity_residual = stationarity_residual[:, 0]
        constraint_residual = constraint_residual[:, 0]
    return LinearResponse(
        state=state,
        multipliers=multipliers,
        stationarity_residual=stationarity_residual,
        constraint_residual=constraint_residual,
    )
