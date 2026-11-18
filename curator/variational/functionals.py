"""Differentiable scalar functionals and equality residuals without dense Hessians.

These objects are short-lived adapters around an energy closure, not trainable
modules or checkpoint payloads. They do not choose which model parameters receive
loss gradients: the caller supplies the appropriate physical or detached context.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .projection import project_to_equalities
from .validation import validate_equalities


@dataclass(frozen=True)
class FunctionalStationarity:
    """First-order equality residuals; zero residual does not certify a minimum."""

    gradient: torch.Tensor
    stationarity: torch.Tensor
    feasibility: torch.Tensor
    multipliers: torch.Tensor

    @property
    def residual(self) -> torch.Tensor:
        return self.stationarity

    @property
    def primal(self) -> torch.Tensor:
        return self.feasibility


@dataclass(frozen=True)
class FunctionalProblem:
    """A scalar ``energy_function(state)`` with optional linear equalities.

    States are flat vectors. Constraints use ``A @ state = b`` and must have
    independent rows. Derivative methods preserve the graph of an input state
    that already requires gradients; ``create_graph=True`` makes their outputs
    differentiable for residual training. They form no full Hessian or null-space
    basis. ``hvp`` treats its direction as the cotangent of the state gradient.
    """

    energy_function: Callable[[torch.Tensor], torch.Tensor]
    n_state: int
    constraint_matrix: Optional[torch.Tensor] = None
    constraint_target: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if not callable(self.energy_function):
            raise TypeError("energy_function must be callable")
        if isinstance(self.n_state, bool) or not isinstance(self.n_state, int) or self.n_state <= 0:
            raise ValueError("n_state must be a positive integer")
        matrix, target = self.constraint_matrix, self.constraint_target
        if (matrix is None) != (target is None):
            raise ValueError("constraint_matrix and constraint_target must be supplied together")
        if matrix is None:
            return
        if not isinstance(matrix, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError("constraints must be tensors")
        if matrix.ndim != 2 or matrix.shape[1] != self.n_state:
            raise ValueError("constraint_matrix must have shape [n_constraints, n_state]")
        if target.shape != (matrix.shape[0],):
            raise ValueError("constraint_target must have shape [n_constraints]")
        if not matrix.is_floating_point() or not target.is_floating_point():
            raise TypeError("constraints must be floating-point tensors")
        if matrix.dtype != target.dtype or matrix.device != target.device:
            raise ValueError("constraint matrix and target must share dtype and device")
        if not bool(torch.isfinite(matrix).all()) or not bool(torch.isfinite(target).all()):
            raise ValueError("constraints must be finite")
        # Rank checks use the small row system, never an n_state-square basis.
        validate_equalities(matrix, target)

    @property
    def n_constraints(self) -> int:
        return 0 if self.constraint_matrix is None else self.constraint_matrix.shape[0]

    def _validate_state(self, state: torch.Tensor, name: str = "state") -> None:
        if not isinstance(state, torch.Tensor) or state.shape != (self.n_state,):
            raise ValueError("%s must be a vector of length n_state" % name)
        if not state.is_floating_point():
            raise TypeError("%s must be floating point" % name)
        if not bool(torch.isfinite(state).all()):
            raise ValueError("%s must be finite" % name)
        matrix = self.constraint_matrix
        if matrix is not None and (matrix.dtype != state.dtype or matrix.device != state.device):
            raise ValueError("state and constraints must share dtype and device")

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        self._validate_state(state)
        value = self.energy_function(state)
        if not isinstance(value, torch.Tensor) or value.ndim != 0:
            raise ValueError("energy_function must return a scalar tensor")
        if value.dtype != state.dtype or value.device != state.device:
            raise ValueError("energy and state must share dtype and device")
        if not bool(torch.isfinite(value)):
            raise ValueError("energy_function returned a non-finite value")
        return value

    def constraints(self, state: torch.Tensor) -> torch.Tensor:
        self._validate_state(state)
        if self.constraint_matrix is None:
            return state.new_zeros((0,))
        return self.constraint_matrix @ state - self.constraint_target

    def state_gradient(self, state: torch.Tensor, create_graph: bool = False,
                       *, energy: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Partial derivative; ``energy`` may reuse this same state's live graph."""

        self._validate_state(state)
        with torch.enable_grad():
            if energy is not None and not state.requires_grad:
                raise ValueError("reusing energy requires its original differentiable state")
            point = state if state.requires_grad else state.detach().requires_grad_(True)
            value = self.energy(point) if energy is None else energy
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                raise ValueError("cached energy must be a scalar tensor")
            if value.dtype != point.dtype or value.device != point.device:
                raise ValueError("cached energy must share state dtype and device")
            gradient = None
            if value.requires_grad:
                gradient = torch.autograd.grad(
                    value, point, create_graph=create_graph, retain_graph=True,
                    allow_unused=True,
                )[0]
            if gradient is None:
                gradient = torch.zeros_like(point)
            if create_graph:
                # Constant/affine energies still permit a zero second derivative.
                gradient = gradient + point * 0.0
            else:
                gradient = gradient.detach()
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError("state gradient is non-finite")
        return gradient

    def hvp(
        self, state: torch.Tensor, direction: torch.Tensor, create_graph: bool = False,
    ) -> torch.Tensor:
        """Compute ``Phi_xx(state) @ direction`` without materializing ``Phi_xx``."""

        self._validate_state(state)
        self._validate_state(direction, "direction")
        if direction.dtype != state.dtype or direction.device != state.device:
            raise ValueError("state and direction must share dtype and device")
        with torch.enable_grad():
            point = state if state.requires_grad else state.detach().requires_grad_(True)
            gradient = self.state_gradient(point, create_graph=True)
            result = torch.autograd.grad(
                gradient, point, grad_outputs=direction, create_graph=create_graph,
                retain_graph=True, allow_unused=True,
            )[0]
            if result is None:
                result = torch.zeros_like(point)
            if create_graph:
                result = result + point * 0.0 + direction * 0.0
            else:
                result = result.detach()
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Hessian vector product is non-finite")
        return result

    def project_state(self, state: torch.Tensor) -> torch.Tensor:
        """Euclidean projection, using only a constraint-sized linear solve."""

        self._validate_state(state)
        if not self.n_constraints:
            return state
        return project_to_equalities(
            state, self.constraint_matrix, self.constraint_target,
        ).state

    def stationarity(
        self, state: torch.Tensor, create_graph: bool = False,
        *, energy: Optional[torch.Tensor] = None,
    ) -> FunctionalStationarity:
        """Return tangent gradient and actual feasibility at ``state``.

        The state is not silently projected. Multipliers minimize the Euclidean
        stationarity residual, so ``stationarity = gradient + A.T @ lambda``.
        No division by a learned curvature or residual scale is performed here.
        """

        with torch.enable_grad():
            gradient = self.state_gradient(state, create_graph=create_graph, energy=energy)
            if self.n_constraints:
                tangent = project_to_equalities(
                    gradient, self.constraint_matrix, torch.zeros_like(self.constraint_target),
                )
                residual, multipliers = tangent.state, -tangent.multipliers
            else:
                residual, multipliers = gradient, state.new_zeros((0,))
            feasibility = self.constraints(state)
        if not create_graph:
            residual, feasibility, multipliers = (
                value.detach() for value in (residual, feasibility, multipliers)
            )
        return FunctionalStationarity(gradient, residual, feasibility, multipliers)


def solve_functional_reference(problem: FunctionalProblem, initial_state: torch.Tensor, options=None):
    """Explicit small-system evaluation using the existing dense Newton oracle.

    This routine constructs dense Hessians, returns detached numerical results,
    and may converge to a local stationary point or fail. Inspect ``converged``,
    ``stable`` and ``status``; it is not a differentiable layer or a global-minimum
    guarantee. Joint proposal training does not call this routine.
    """

    from .nonlinear import NonlinearEqualityProblem, solve_newton_kkt

    problem._validate_state(initial_state)
    reference = NonlinearEqualityProblem(
        n_state=problem.n_state,
        n_constraints=problem.n_constraints,
        # Keep the dense oracle well-defined for state-independent objectives.
        energy_function=lambda state: problem.energy(state) + 0.0 * state.square().sum(),
        constraint_function=problem.constraints,
    )
    with torch.enable_grad():
        return solve_newton_kkt(reference, initial_state, options=options)
