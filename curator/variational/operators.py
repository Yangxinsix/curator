"""Operator protocols shared by variable plugins and long-range backends."""

from dataclasses import dataclass
from typing import Protocol, Union

import torch


class VariationalOperator(Protocol):
    """Energy operator acting on a flattened internal state."""

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        ...

    def matvec(self, state: torch.Tensor) -> torch.Tensor:
        ...


class QuadraticOperator(VariationalOperator, Protocol):
    """An operator exposing its symmetric quadratic kernel."""

    def kernel(self) -> torch.Tensor:
        ...


class DenseQuadraticOperator:
    """Small dense operator used by reference solvers and tests."""

    def __init__(self, kernel: torch.Tensor):
        if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
            raise ValueError("quadratic kernel must be square")
        if not torch.allclose(kernel, kernel.transpose(0, 1)):
            raise ValueError("quadratic kernel must be symmetric")
        self._kernel = kernel

    def kernel(self) -> torch.Tensor:
        return self._kernel

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        return 0.5 * state @ self._kernel @ state

    def matvec(self, state: torch.Tensor) -> torch.Tensor:
        return self._kernel @ state


class BoundaryQuadraticOperator:
    """Add a low-rank Robin curvature to a matrix-free base operator."""

    def __init__(
        self,
        base: VariationalOperator,
        observable_matrix: torch.Tensor,
        stiffness: torch.Tensor,
    ) -> None:
        if observable_matrix.ndim != 2:
            raise ValueError("observable_matrix must be rank 2")
        if stiffness.shape != (observable_matrix.shape[0], observable_matrix.shape[0]):
            raise ValueError("stiffness shape must match observable rows")
        if not observable_matrix.is_floating_point():
            raise ValueError("boundary operator tensors must be floating point")
        if (
            stiffness.dtype != observable_matrix.dtype
            or stiffness.device != observable_matrix.device
        ):
            raise ValueError("boundary operator tensors must match dtype and device")
        if not bool(torch.isfinite(observable_matrix).all().item()) or not bool(
            torch.isfinite(stiffness).all().item()
        ):
            raise ValueError("boundary operator tensors must be finite")
        if not torch.allclose(stiffness, stiffness.transpose(0, 1)):
            raise ValueError("boundary stiffness must be symmetric")
        if bool((torch.linalg.eigvalsh(stiffness) < -1.0e-12).any().item()):
            raise ValueError("boundary stiffness must be positive semidefinite")
        self.base = base
        self.observable_matrix = observable_matrix
        self.stiffness = stiffness

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        observable = self.observable_matrix @ state
        return self.base.energy(state) + 0.5 * observable @ self.stiffness @ observable

    def matvec(self, state: torch.Tensor) -> torch.Tensor:
        observable = self.observable_matrix @ state
        return (
            self.base.matvec(state)
            + self.observable_matrix.transpose(0, 1) @ self.stiffness @ observable
        )


@dataclass(frozen=True)
class OperatorQuadraticProblem:
    """Matrix-free quadratic plus its fully compiled linear boundary terms."""

    operator: VariationalOperator
    linear: torch.Tensor
    constraint_matrix: torch.Tensor
    constraint_target: torch.Tensor
    constant: Union[torch.Tensor, float] = 0.0

    def __post_init__(self) -> None:
        if self.linear.ndim != 1 or not self.linear.is_floating_point():
            raise ValueError("linear must be a floating-point rank-1 tensor")
        if self.constraint_matrix.ndim != 2:
            raise ValueError("constraint_matrix must be rank 2")
        if self.constraint_target.shape != (self.constraint_matrix.shape[0],):
            raise ValueError("constraint_target must be a rank-1 row target")
        if self.constraint_matrix.shape[1] != self.linear.numel():
            raise ValueError("constraint shapes do not match the linear term")
        tensors = (self.constraint_matrix, self.constraint_target)
        if any(value.dtype != self.linear.dtype for value in tensors):
            raise ValueError("operator problem tensors must use the same dtype")
        if any(value.device != self.linear.device for value in tensors):
            raise ValueError("operator problem tensors must use the same device")
        if not isinstance(self.constant, torch.Tensor):
            object.__setattr__(self, "constant", self.linear.new_tensor(self.constant))
        if self.constant.numel() != 1:
            raise ValueError("constant must be scalar")
        if self.constant.dtype != self.linear.dtype or self.constant.device != self.linear.device:
            raise ValueError("constant must match the linear term")
        if not bool(torch.isfinite(self.linear).all().item()) or any(
            not bool(torch.isfinite(value).all().item()) for value in tensors
        ) or not bool(torch.isfinite(self.constant).all().item()):
            raise ValueError("operator problem tensors must be finite")


__all__ = [
    "BoundaryQuadraticOperator",
    "DenseQuadraticOperator",
    "OperatorQuadraticProblem",
    "QuadraticOperator",
    "VariationalOperator",
]
