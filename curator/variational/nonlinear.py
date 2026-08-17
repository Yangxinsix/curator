"""Dense autograd oracle for nonlinear equality-constrained variational states."""

from dataclasses import dataclass
import math
from typing import Callable, Optional, Tuple

import torch

from .functional import QuadraticProblem
from .result import KKTResiduals
from .validation import null_space, validate_problem


@dataclass(frozen=True)
class NonlinearEqualityProblem:
    """A small reference problem ``min energy(x)`` subject to ``c(x)=0``."""

    n_state: int
    n_constraints: int
    energy_function: Callable[[torch.Tensor], torch.Tensor]
    constraint_function: Callable[[torch.Tensor], torch.Tensor]

    def __post_init__(self) -> None:
        if self.n_state <= 0 or self.n_constraints < 0:
            raise ValueError("nonlinear problem dimensions are invalid")

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        value = self.energy_function(state)
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise ValueError("nonlinear energy_function must return a scalar tensor")
        if value.dtype != state.dtype or value.device != state.device:
            raise ValueError("nonlinear energy must match state dtype and device")
        return value.reshape(())

    def constraints(self, state: torch.Tensor) -> torch.Tensor:
        value = self.constraint_function(state)
        if not isinstance(value, torch.Tensor) or value.shape != (self.n_constraints,):
            raise ValueError(
                "nonlinear constraint_function must return shape (n_constraints,)"
            )
        if value.dtype != state.dtype or value.device != state.device:
            raise ValueError("nonlinear constraints must match state dtype and device")
        return value


@dataclass(frozen=True)
class NewtonKKTOptions:
    max_iterations: int = 25
    stationarity_atol: float = 1.0e-10
    stationarity_rtol: float = 1.0e-8
    feasibility_atol: float = 1.0e-12
    feasibility_rtol: float = 1.0e-10
    armijo: float = 1.0e-4
    backtrack: float = 0.5
    max_backtracks: int = 20
    min_step: float = 1.0e-8

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations < 0
        ):
            raise ValueError("max_iterations must be a non-negative integer")
        if (
            isinstance(self.max_backtracks, bool)
            or not isinstance(self.max_backtracks, int)
            or self.max_backtracks < 0
        ):
            raise ValueError("max_backtracks must be a non-negative integer")
        tolerances = (
            self.stationarity_atol,
            self.stationarity_rtol,
            self.feasibility_atol,
            self.feasibility_rtol,
            self.min_step,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("Newton tolerances must be finite and non-negative")
        if not math.isfinite(self.armijo) or not 0.0 < self.armijo < 1.0:
            raise ValueError("armijo must lie strictly between zero and one")
        if not math.isfinite(self.backtrack) or not 0.0 < self.backtrack < 1.0:
            raise ValueError("backtrack must lie strictly between zero and one")


@dataclass(frozen=True)
class NewtonKKTResult:
    state: torch.Tensor
    multipliers: torch.Tensor
    energy: torch.Tensor
    residuals: KKTResiduals
    minimum_reduced_curvature: torch.Tensor
    converged: bool
    feasible: bool
    stable: Optional[bool]
    reliable: bool
    iterations: int
    backtracks: int
    status: str
    failure_reason: Optional[str]


def nonlinear_from_quadratic(problem: QuadraticProblem) -> NonlinearEqualityProblem:
    """Adapt a validated quadratic to the nonlinear reference interface."""

    validate_problem(problem)
    return NonlinearEqualityProblem(
        n_state=problem.n_state,
        n_constraints=problem.n_constraints,
        energy_function=problem.energy,
        constraint_function=lambda state: (
            problem.constraint_matrix @ state - problem.constraint_target
        ),
    )


def _derivatives(
    problem: NonlinearEqualityProblem,
    state: torch.Tensor,
    multipliers: torch.Tensor,
    need_hessian: bool,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
]:
    state = state.detach().requires_grad_(True)
    energy = problem.energy(state)
    constraints = problem.constraints(state)
    lagrangian = energy + multipliers.detach() @ constraints
    stationarity = torch.autograd.grad(
        lagrangian, state, create_graph=need_hessian
    )[0]
    if problem.n_constraints:
        jacobian = torch.autograd.functional.jacobian(
            problem.constraints, state, create_graph=False
        )
    else:
        jacobian = state.new_zeros((0, problem.n_state))
    hessian = None
    if need_hessian:
        hessian = torch.autograd.functional.hessian(
            lambda value: problem.energy(value)
            + multipliers.detach() @ problem.constraints(value),
            state,
            create_graph=False,
        )
        hessian = 0.5 * (hessian + hessian.transpose(0, 1))
    return energy.detach(), stationarity.detach(), constraints.detach(), (
        None if hessian is None else hessian.detach()
    ), jacobian.detach()


def _residuals(
    stationarity: torch.Tensor,
    constraints: torch.Tensor,
    hessian_scale: torch.Tensor,
    constraint_scale: torch.Tensor,
) -> KKTResiduals:
    stationarity_norm = (
        stationarity.abs().max() if stationarity.numel() else stationarity.new_zeros(())
    )
    primal_norm = (
        constraints.abs().max() if constraints.numel() else constraints.new_zeros(())
    )
    return KKTResiduals(
        stationarity=stationarity,
        primal=constraints,
        stationarity_norm=stationarity_norm,
        primal_norm=primal_norm,
        stationarity_scale=hessian_scale,
        primal_scale=constraint_scale,
        normalized_stationarity_norm=stationarity_norm / (1.0 + hessian_scale),
        normalized_primal_norm=primal_norm / (1.0 + constraint_scale),
    )


def _finite(*values: torch.Tensor) -> bool:
    return all(bool(torch.isfinite(value).all().item()) for value in values)


def solve_newton_kkt(
    problem: NonlinearEqualityProblem,
    initial_state: torch.Tensor,
    initial_multipliers: Optional[torch.Tensor] = None,
    options: Optional[NewtonKKTOptions] = None,
) -> NewtonKKTResult:
    """Find and classify a KKT root using dense Newton and residual line search."""

    options = NewtonKKTOptions() if options is None else options
    if initial_state.shape != (problem.n_state,) or not initial_state.is_floating_point():
        raise ValueError("initial_state must be a floating-point n_state vector")
    state = initial_state.detach().clone()
    multipliers = (
        state.new_zeros((problem.n_constraints,))
        if initial_multipliers is None
        else initial_multipliers.detach().clone()
    )
    if multipliers.shape != (problem.n_constraints,):
        raise ValueError("initial_multipliers has the wrong shape")
    if multipliers.dtype != state.dtype or multipliers.device != state.device:
        raise ValueError("initial multipliers must match state dtype and device")
    if not _finite(state, multipliers):
        raise ValueError("initial KKT state must be finite")

    total_backtracks = 0
    last_residuals = None
    last_energy = state.new_tensor(float("nan"))
    for iteration in range(options.max_iterations + 1):
        try:
            energy, stationarity, constraints, hessian, jacobian = _derivatives(
                problem, state, multipliers, need_hessian=True
            )
        except (RuntimeError, ValueError) as error:
            status = "nonfinite_residual"
            failure = str(error)
            break
        last_energy = energy
        if not _finite(energy, stationarity, constraints, hessian, jacobian):
            status = "nonfinite_residual"
            failure = "functional or derivative returned non-finite values"
            break
        constraint_term = (
            jacobian.transpose(0, 1) @ multipliers
            if problem.n_constraints
            else state.new_zeros(state.shape)
        )
        energy_gradient = stationarity - constraint_term
        stationarity_scale = torch.stack(
            (
                energy_gradient.abs().max(),
                constraint_term.abs().max(),
                state.new_tensor(torch.finfo(state.dtype).tiny),
            )
        ).max()
        constraint_scale = torch.stack(
            (
                constraints.abs().max() if constraints.numel() else state.new_zeros(()),
                state.new_tensor(torch.finfo(state.dtype).tiny),
            )
        ).max()
        last_residuals = _residuals(
            stationarity, constraints, stationarity_scale, constraint_scale
        )
        if problem.n_constraints:
            rank = int(torch.linalg.matrix_rank(jacobian).item())
            if rank < problem.n_constraints:
                status = "rank_deficient_jacobian"
                failure = "constraint Jacobian is rank deficient"
                break
        stationarity_ok = last_residuals.stationarity_norm <= (
            options.stationarity_atol
            + options.stationarity_rtol * last_residuals.stationarity_scale
        )
        feasible = bool(
            (
                last_residuals.primal_norm
                <= options.feasibility_atol
                + options.feasibility_rtol * last_residuals.primal_scale
            ).item()
        )
        if bool(stationarity_ok.item()) and feasible:
            tangent = null_space(jacobian)
            if tangent.shape[1]:
                reduced = tangent.transpose(0, 1) @ hessian @ tangent
                minimum_curvature = torch.linalg.eigvalsh(
                    0.5 * (reduced + reduced.transpose(0, 1))
                ).min()
            else:
                minimum_curvature = state.new_tensor(float("inf"))
            scale = max(1.0, float(hessian.abs().max().item()))
            stable = bool(
                (
                    minimum_curvature
                    > 100.0 * torch.finfo(state.dtype).eps * problem.n_state * scale
                ).item()
            )
            return NewtonKKTResult(
                state=state,
                multipliers=multipliers,
                energy=energy,
                residuals=last_residuals,
                minimum_reduced_curvature=minimum_curvature,
                converged=True,
                feasible=True,
                stable=stable,
                reliable=stable,
                iterations=iteration,
                backtracks=total_backtracks,
                status=(
                    "converged_minimum" if stable else "unstable_stationary_point"
                ),
                failure_reason=None if stable else "reduced Hessian is not positive",
            )
        if iteration == options.max_iterations:
            status = "max_iterations"
            failure = "maximum Newton iterations reached"
            break
        kkt = torch.cat(
            (
                torch.cat((hessian, jacobian.transpose(0, 1)), dim=1),
                torch.cat(
                    (
                        jacobian,
                        state.new_zeros(
                            (problem.n_constraints, problem.n_constraints)
                        ),
                    ),
                    dim=1,
                ),
            ),
            dim=0,
        )
        residual = torch.cat((stationarity, constraints), dim=0)
        try:
            step = torch.linalg.solve(kkt, -residual)
        except RuntimeError as error:
            status = "linear_solve_failed"
            failure = str(error)
            break
        state_step = step[: problem.n_state]
        multiplier_step = step[problem.n_state :]
        stationarity_denominator = 1.0 + last_residuals.stationarity_scale
        constraint_denominator = 1.0 + last_residuals.primal_scale
        merit = 0.5 * (
            (stationarity / stationarity_denominator).square().sum()
            + (constraints / constraint_denominator).square().sum()
        )
        accepted = False
        step_size = 1.0
        for backtrack_index in range(options.max_backtracks + 1):
            trial_state = state + step_size * state_step
            trial_multipliers = multipliers + step_size * multiplier_step
            try:
                _, trial_stationarity, trial_constraints, _, _ = _derivatives(
                    problem, trial_state, trial_multipliers, need_hessian=False
                )
            except (RuntimeError, ValueError):
                trial_stationarity = state.new_full(state.shape, float("nan"))
                trial_constraints = state.new_full(
                    (problem.n_constraints,), float("nan")
                )
            if _finite(trial_stationarity, trial_constraints):
                trial_merit = 0.5 * (
                    (trial_stationarity / stationarity_denominator).square().sum()
                    + (trial_constraints / constraint_denominator).square().sum()
                )
                if bool(
                    (
                        trial_merit
                        <= (1.0 - options.armijo * step_size) * merit
                    ).item()
                ):
                    accepted = True
                    state = trial_state
                    multipliers = trial_multipliers
                    break
            total_backtracks += 1
            step_size *= options.backtrack
            if step_size < options.min_step:
                break
        if not accepted:
            status = "line_search_failed"
            failure = "no residual-decreasing Newton step was accepted"
            break

    if last_residuals is None:
        empty_stationarity = state.new_full((problem.n_state,), float("nan"))
        empty_constraints = state.new_full((problem.n_constraints,), float("nan"))
        last_residuals = _residuals(
            empty_stationarity,
            empty_constraints,
            state.new_tensor(float("nan")),
            state.new_tensor(float("nan")),
        )
    feasible = bool(
        torch.isfinite(last_residuals.primal_norm).item()
        and (
            last_residuals.primal_norm
            <= options.feasibility_atol
            + options.feasibility_rtol * last_residuals.primal_scale
        ).item()
    )
    return NewtonKKTResult(
        state=state,
        multipliers=multipliers,
        energy=last_energy,
        residuals=last_residuals,
        minimum_reduced_curvature=state.new_tensor(float("nan")),
        converged=False,
        feasible=feasible,
        stable=None,
        reliable=False,
        iterations=min(iteration, options.max_iterations),
        backtracks=total_backtracks,
        status=status,
        failure_reason=failure,
    )


__all__ = [
    "NewtonKKTOptions",
    "NewtonKKTResult",
    "NonlinearEqualityProblem",
    "nonlinear_from_quadratic",
    "solve_newton_kkt",
]
