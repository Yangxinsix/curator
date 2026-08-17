"""Atomic-charge functional and compatibility helpers."""

from dataclasses import dataclass
from typing import Optional

import torch

from ..compiler import CompiledBoundary
from ..functional import QuadraticProblem
from ..spec import BoundarySpec, ConstraintSpec, GroupSpec, VariableSpec


def charge_variable_spec(name: str = "atomic_charge") -> VariableSpec:
    """Return the built-in charge variable with Curator's physical units."""

    return VariableSpec(
        name=name,
        kind="atomic_charge",
        components=1,
        unit="e",
        conjugate_unit="V",
        component_names=("q",),
        semantics="variational_internal",
    )


def fixed_total_charge_boundary(
    total_charge: float,
    variable: Optional[VariableSpec] = None,
) -> BoundarySpec:
    """Build the canonical one-structure fixed-total-charge specification."""

    variable = charge_variable_spec() if variable is None else variable
    return BoundarySpec(
        variable=variable,
        groups=(GroupSpec.all(),),
        equalities=(
            ConstraintSpec.group_sum(
                name="total_charge",
                group="all",
                target=total_charge,
                unit="e",
            ),
        ),
    )


@dataclass(frozen=True)
class ChargeFunctional:
    r"""Quadratic QEq functional.

    .. math::

        F(q)=\chi^Tq+\tfrac12q^T[\operatorname{diag}(\eta)+K]q.
    """

    electronegativity: torch.Tensor
    hardness: torch.Tensor
    interaction_kernel: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.electronegativity.ndim != 1:
            raise ValueError("electronegativity must be rank 1")
        if self.hardness.shape != self.electronegativity.shape:
            raise ValueError("hardness must match electronegativity")
        if bool((self.hardness <= 0.0).any().item()):
            raise ValueError("hardness must be positive")
        if self.interaction_kernel is not None:
            n_state = self.electronegativity.numel()
            if self.interaction_kernel.shape != (n_state, n_state):
                raise ValueError("interaction kernel has the wrong shape")
            if not torch.allclose(
                self.interaction_kernel,
                self.interaction_kernel.transpose(0, 1),
            ):
                raise ValueError("interaction kernel must be symmetric")

    @property
    def hessian(self) -> torch.Tensor:
        hessian = torch.diag(self.hardness)
        if self.interaction_kernel is not None:
            hessian = hessian + self.interaction_kernel
        return hessian

    def energy(self, charge: torch.Tensor) -> torch.Tensor:
        return (
            self.electronegativity @ charge
            + 0.5 * charge @ self.hessian @ charge
        )

    def chemical_potential(self, charge: torch.Tensor) -> torch.Tensor:
        return self.hessian @ charge + self.electronegativity

    def problem(self, boundary: CompiledBoundary) -> QuadraticProblem:
        return boundary.to_problem(
            hessian=self.hessian,
            linear=self.electronegativity,
        )


def project_fixed_total_charge(
    raw_charge: torch.Tensor,
    hardness: torch.Tensor,
    image_idx: torch.Tensor,
    total_charge: torch.Tensor,
) -> torch.Tensor:
    """Optimized ragged projection in the onsite-hardness metric.

    This is the one-row-per-structure specialization of
    :func:`curator.variational.project_to_equalities`; it avoids launching one
    dense solve per structure on the molecular-dynamics path.
    """

    if raw_charge.ndim != 1 or hardness.shape != raw_charge.shape:
        raise ValueError("raw_charge and hardness must be matching vectors")
    if image_idx.shape != raw_charge.shape or image_idx.dtype != torch.long:
        raise ValueError("image_idx must be a long vector matching raw_charge")
    if total_charge.ndim != 1:
        raise ValueError("total_charge must be a vector")
    if bool((hardness <= 0.0).any().item()):
        raise ValueError("hardness must be positive")
    if raw_charge.numel() and (
        int(image_idx.min().item()) < 0
        or int(image_idx.max().item()) >= total_charge.numel()
    ):
        raise ValueError("image_idx is outside the total_charge batch")

    predicted_total = raw_charge.new_zeros(total_charge.shape)
    predicted_total.index_add_(0, image_idx, raw_charge)
    inverse_hardness = hardness.reciprocal()
    normalization = raw_charge.new_zeros(total_charge.shape)
    normalization.index_add_(0, image_idx, inverse_hardness)
    if bool((normalization <= 0.0).any().item()):
        raise ValueError("every structure must contain at least one entity")
    correction = inverse_hardness / normalization[image_idx]
    return raw_charge + correction * (
        total_charge - predicted_total
    )[image_idx]
