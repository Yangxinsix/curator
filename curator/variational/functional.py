"""Functionals used by the reference variational solvers."""

from dataclasses import dataclass
from typing import Union

import torch


@dataclass(frozen=True)
class QuadraticProblem:
    r"""A convex quadratic objective with row-major equality constraints.

    The problem is

    .. math::

        \min_x \tfrac12 x^T H x + g^T x + c,\qquad A x = b.

    ``constraint_matrix`` therefore has shape ``(n_constraints, n_state)``.
    Empty equality systems are represented by a ``(0, n_state)`` matrix.
    """

    hessian: torch.Tensor
    linear: torch.Tensor
    constraint_matrix: torch.Tensor
    constraint_target: torch.Tensor
    constant: Union[torch.Tensor, float] = 0.0

    def __post_init__(self) -> None:
        if self.hessian.ndim != 2 or self.hessian.shape[0] != self.hessian.shape[1]:
            raise ValueError("hessian must be a square rank-2 tensor")
        n_state = self.hessian.shape[0]
        if self.linear.shape != (n_state,):
            raise ValueError("linear must have shape (n_state,)")
        if self.constraint_matrix.ndim != 2:
            raise ValueError("constraint_matrix must be rank 2")
        if self.constraint_matrix.shape[1] != n_state:
            raise ValueError(
                "constraint_matrix must use row-major shape "
                "(n_constraints, n_state)"
            )
        if self.constraint_target.shape != (self.constraint_matrix.shape[0],):
            raise ValueError("constraint_target must have shape (n_constraints,)")
        if not isinstance(self.constant, torch.Tensor):
            object.__setattr__(
                self,
                "constant",
                self.hessian.new_tensor(self.constant),
            )
        if self.constant.numel() != 1:
            raise ValueError("constant must be scalar")
        tensors = (
            self.linear,
            self.constraint_matrix,
            self.constraint_target,
            self.constant,
        )
        if any(tensor.device != self.hessian.device for tensor in tensors):
            raise ValueError("all problem tensors must be on the same device")
        if any(tensor.dtype != self.hessian.dtype for tensor in tensors):
            raise ValueError("all problem tensors must use the same dtype")
        if not self.hessian.is_floating_point():
            raise ValueError("quadratic problems require a floating-point dtype")
        if not bool(torch.isfinite(self.hessian).all().item()) or any(
            not bool(torch.isfinite(tensor).all().item()) for tensor in tensors
        ):
            raise ValueError("quadratic problem tensors must be finite")

    @classmethod
    def unconstrained(
        cls,
        hessian: torch.Tensor,
        linear: torch.Tensor,
        constant: float = 0.0,
    ) -> "QuadraticProblem":
        n_state = hessian.shape[0]
        return cls(
            hessian=hessian,
            linear=linear,
            constraint_matrix=hessian.new_zeros((0, n_state)),
            constraint_target=hessian.new_zeros((0,)),
            constant=hessian.new_tensor(constant),
        )

    @property
    def n_state(self) -> int:
        return self.hessian.shape[0]

    @property
    def n_constraints(self) -> int:
        return self.constraint_matrix.shape[0]

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * state @ self.hessian @ state
            + self.linear @ state
            + self.constant.reshape(())
        )

    def gradient(self, state: torch.Tensor) -> torch.Tensor:
        return self.hessian @ state + self.linear
