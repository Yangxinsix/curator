"""Structured results returned by the variational core."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KKTResiduals:
    """Residual vectors and norms for an equality-constrained optimum."""

    stationarity: torch.Tensor
    primal: torch.Tensor
    stationarity_norm: torch.Tensor
    primal_norm: torch.Tensor
    stationarity_scale: torch.Tensor
    primal_scale: torch.Tensor
    normalized_stationarity_norm: torch.Tensor
    normalized_primal_norm: torch.Tensor

    @property
    def maximum_norm(self) -> torch.Tensor:
        return torch.maximum(self.stationarity_norm, self.primal_norm)

    @property
    def maximum_normalized_norm(self) -> torch.Tensor:
        return torch.maximum(
            self.normalized_stationarity_norm,
            self.normalized_primal_norm,
        )


@dataclass(frozen=True)
class VariationalResult:
    """Exact or corrected state together with its numerical certificate."""

    state: torch.Tensor
    multipliers: torch.Tensor
    energy: torch.Tensor
    residuals: KKTResiduals
    condition_number: torch.Tensor
    minimum_reduced_curvature: torch.Tensor
    reliable: bool


@dataclass(frozen=True)
class ProjectionResult:
    """Result of a metric projection onto the equality constraint set."""

    state: torch.Tensor
    multipliers: torch.Tensor
    residual: torch.Tensor


@dataclass(frozen=True)
class LinearResponse:
    """Solution of one or more linearized KKT systems."""

    state: torch.Tensor
    multipliers: torch.Tensor
    stationarity_residual: torch.Tensor
    constraint_residual: torch.Tensor
