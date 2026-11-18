"""Validation and diagnostics for compiled variational problems."""

from dataclasses import dataclass
from typing import Optional

import torch

from .functional import QuadraticProblem


class VariationalValidationError(ValueError):
    """Base class for invalid variational problem definitions."""


class InconsistentConstraintsError(VariationalValidationError):
    """The equality targets are outside the range of the constraint matrix."""


class RankDeficientConstraintsError(VariationalValidationError):
    """The equality system contains redundant rows."""


class UnstableVariationalProblemError(VariationalValidationError):
    """The objective is not strictly convex on the feasible tangent space."""


class SingularKKTError(RuntimeError):
    """The assembled KKT system cannot be solved uniquely."""


class ThermodynamicGaugeError(VariationalValidationError):
    """An intensive-control gauge was not resolved or acknowledged."""


@dataclass(frozen=True)
class EqualityValidation:
    rank: int
    augmented_rank: int
    n_constraints: int


def _rank_tolerance(tensor: torch.Tensor) -> float:
    """Scale-aware singular-value threshold used by every rank decision."""

    if tensor.numel() == 0:
        return 0.0
    largest = float(torch.linalg.svdvals(tensor.detach()).max().item())
    return float(
        100.0
        * torch.finfo(tensor.dtype).eps
        * max(tensor.shape or (1,))
        * largest
    )


def validate_equalities(
    matrix: torch.Tensor,
    target: torch.Tensor,
    tolerance: Optional[float] = None,
) -> EqualityValidation:
    """Reject inconsistent and redundant row-major equality systems."""

    if matrix.ndim != 2:
        raise ValueError("constraint matrix must be rank 2")
    if target.shape != (matrix.shape[0],):
        raise ValueError("constraint target has the wrong shape")
    if matrix.shape[0] == 0:
        return EqualityValidation(rank=0, augmented_rank=0, n_constraints=0)

    matrix_tolerance = _rank_tolerance(matrix) if tolerance is None else tolerance
    rank = int(torch.linalg.matrix_rank(matrix.detach(), tol=matrix_tolerance).item())
    augmented = torch.cat((matrix, target[:, None]), dim=1)
    augmented_tolerance = (
        _rank_tolerance(augmented) if tolerance is None else tolerance
    )
    augmented_rank = int(
        torch.linalg.matrix_rank(augmented.detach(), tol=augmented_tolerance).item()
    )
    if augmented_rank > rank:
        raise InconsistentConstraintsError(
            "equality targets are inconsistent with the constraint rows"
        )
    if rank < matrix.shape[0]:
        raise RankDeficientConstraintsError(
            "equality constraints contain redundant or gauge-equivalent rows"
        )
    return EqualityValidation(
        rank=rank,
        augmented_rank=augmented_rank,
        n_constraints=matrix.shape[0],
    )


def null_space(matrix: torch.Tensor, tolerance: Optional[float] = None) -> torch.Tensor:
    """Return an orthonormal basis for ``null(matrix)`` as matrix columns."""

    n_state = matrix.shape[1]
    if matrix.shape[0] == 0:
        return torch.eye(n_state, dtype=matrix.dtype, device=matrix.device)
    tolerance = _rank_tolerance(matrix) if tolerance is None else tolerance
    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=True)
    rank = int((singular_values > tolerance).sum().item())
    return vh[rank:].transpose(0, 1)


def thermodynamic_gauge_basis(
    constraint_matrix: torch.Tensor,
    control_matrix: torch.Tensor,
    tolerance: Optional[float] = None,
) -> torch.Tensor:
    r"""Return control directions that cannot drive feasible state changes.

    If ``Z`` spans ``null(A)``, a control displacement is a gauge direction
    when ``Z.T @ B.T @ delta_h = 0``. Compute the same kernel by removing the
    constraint row space from ``B.T``. Only thin state-by-constraint/control
    matrices are formed, never a full state-space projector or tangent basis.
    """

    if constraint_matrix.ndim != 2 or control_matrix.ndim != 2:
        raise ValueError("constraint and control matrices must be rank 2")
    if constraint_matrix.shape[1] != control_matrix.shape[1]:
        raise ValueError("constraint and control state dimensions must match")
    if control_matrix.shape[0] == 0:
        return control_matrix.new_zeros((0, 0))
    n_state = constraint_matrix.shape[1]
    n_controls = control_matrix.shape[0]
    control_transpose = control_matrix.transpose(0, 1)
    tangent_control = control_transpose
    if constraint_matrix.shape[0]:
        _, singular_values, rows = torch.linalg.svd(constraint_matrix, full_matrices=False)
        constraint_tolerance = (
            _rank_tolerance(constraint_matrix) if tolerance is None else tolerance
        )
        rank = int((singular_values > constraint_tolerance).sum().item())
        if rank == n_state:
            return torch.eye(n_controls, dtype=control_matrix.dtype, device=control_matrix.device)
        row_basis = rows[:rank]
        tangent_control = control_transpose - row_basis.transpose(0, 1) @ (
            row_basis @ control_transpose
        )
    # Full right singular vectors are only needed when there are more controls
    # than states; in that case the left factor is no larger than control space.
    _, singular_values, right = torch.linalg.svd(
        tangent_control, full_matrices=n_state < n_controls,
    )
    if tolerance is None:
        # Refer to the unprojected controls as well: an all-gauge control set
        # leaves only roundoff after row-space subtraction, not physical rank.
        scale = float(torch.linalg.svdvals(control_matrix.detach()).max().item())
        tolerance = 100.0 * torch.finfo(control_matrix.dtype).eps * max(n_state, n_controls) * scale
    rank = int((singular_values > tolerance).sum().item())
    return right[rank:].transpose(0, 1)


def minimum_reduced_curvature(
    problem: QuadraticProblem,
    tolerance: Optional[float] = None,
) -> torch.Tensor:
    """Return the smallest Hessian eigenvalue on feasible perturbations."""

    tangent = null_space(problem.constraint_matrix, tolerance=tolerance)
    if tangent.shape[1] == 0:
        return problem.hessian.new_tensor(float("inf"))
    reduced = tangent.transpose(0, 1) @ problem.hessian @ tangent
    reduced = 0.5 * (reduced + reduced.transpose(0, 1))
    return torch.linalg.eigvalsh(reduced).min()


def validate_problem(
    problem: QuadraticProblem,
    tolerance: Optional[float] = None,
    require_stable: bool = True,
) -> torch.Tensor:
    """Validate symmetry, equality consistency, and tangent-space stability."""

    scale = float(problem.hessian.detach().abs().max().item())
    symmetry_tolerance = (
        100.0
        * torch.finfo(problem.hessian.dtype).eps
        * max(problem.hessian.shape)
        * max(scale, torch.finfo(problem.hessian.dtype).tiny)
        if tolerance is None
        else tolerance
    )
    if not torch.allclose(
        problem.hessian,
        problem.hessian.transpose(0, 1),
        rtol=0.0,
        atol=symmetry_tolerance,
    ):
        raise VariationalValidationError("hessian must be symmetric")
    validate_equalities(
        problem.constraint_matrix,
        problem.constraint_target,
        tolerance=tolerance,
    )
    curvature = minimum_reduced_curvature(problem, tolerance=tolerance)
    curvature_scale = float(problem.hessian.detach().abs().max().item())
    stability_tolerance = (
        100.0
        * torch.finfo(problem.hessian.dtype).eps
        * max(problem.hessian.shape)
        * max(curvature_scale, torch.finfo(problem.hessian.dtype).tiny)
        if tolerance is None
        else tolerance
    )
    if require_stable and bool((curvature <= stability_tolerance).item()):
        raise UnstableVariationalProblemError(
            "objective is not strictly convex on the feasible tangent space"
        )
    return curvature
