"""Operator protocols shared by variable plugins and long-range backends."""

from typing import Protocol

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
