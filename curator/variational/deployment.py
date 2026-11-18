"""Per-forward reuse of validated equality projections for deployment."""
from dataclasses import dataclass, field
from typing import Optional

import torch

from .functionals import FunctionalProblem


@dataclass(frozen=True)
class DeploymentFunctionalProblem(FunctionalProblem):
    """Same constrained energy, with one equality validation per live boundary.

    This object is ephemeral. It must not be cached across forwards: A and b may
    carry coordinate/control derivatives. Only the small A A^T system is cached;
    no state-space Hessian, inverse or projection matrix is formed.
    """

    projection_source: Optional["DeploymentFunctionalProblem"] = field(default=None, repr=False)
    _schur: Optional[torch.Tensor] = field(init=False, default=None, repr=False)

    def __post_init__(self):
        source = self.projection_source
        if source is None:
            super().__post_init__()
            if self.n_constraints:
                object.__setattr__(self, "_schur", self.constraint_matrix @ self.constraint_matrix.T)
        else:
            if (not isinstance(source, DeploymentFunctionalProblem)
                    or source.n_state != self.n_state
                    or source.constraint_matrix is not self.constraint_matrix
                    or source.constraint_target is not self.constraint_target):
                raise ValueError("projection reuse requires the same validated live boundary tensors")
            if not callable(self.energy_function):
                raise TypeError("energy_function must be callable")
            object.__setattr__(self, "_schur", source._schur)

    def with_energy(self, energy_function):
        return type(self)(energy_function, self.n_state, self.constraint_matrix,
                          self.constraint_target, projection_source=self)

    def _project(self, state, target):
        self._validate_state(state)
        if not self.n_constraints:
            return state
        matrix = self.constraint_matrix
        violation = matrix @ state - target
        # A single constraint is an exact scalar solve, regardless of row shape.
        multiplier = (violation / self._schur.reshape(())
                      if self.n_constraints == 1 else torch.linalg.solve(self._schur, violation))
        return state - matrix.T @ multiplier

    def project_state(self, state):
        return self._project(state, self.constraint_target)

    def project_tangent(self, state):
        target = None if self.constraint_target is None else torch.zeros_like(self.constraint_target)
        return self._project(state, target)

