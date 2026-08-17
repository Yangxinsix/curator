"""Backend-independent teacher construction for amortized variational solves.

This module deliberately knows nothing about charge, Ewald, dipoles, or any
other physical plugin.  A plugin produces a :class:`QuadraticProblem` and a
boundary-conditioned proposal; this module projects the proposal, evaluates
its *pre-correction* KKT residual, and constructs an exact detached teacher.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from .backends.dense import solve_dense
from .functional import QuadraticProblem
from .kkt import evaluate_kkt
from .projection import project_to_equalities
from .validation import validate_problem


@dataclass(frozen=True)
class AmortizedTrainingBatch:
    """Ragged projected proposals and exact teachers for several conditions."""

    proposal: torch.Tensor
    projected_state: torch.Tensor
    teacher_state: torch.Tensor
    projected_multipliers: torch.Tensor
    teacher_multipliers: torch.Tensor
    normalized_projected_kkt: torch.Tensor
    state_ptr: torch.Tensor
    multiplier_ptr: torch.Tensor
    kkt_ptr: torch.Tensor
    projected_energy: torch.Tensor
    teacher_energy: torch.Tensor

    @property
    def n_conditions(self) -> int:
        return self.state_ptr.numel() - 1

    @property
    def correction(self) -> torch.Tensor:
        return self.teacher_state - self.projected_state

    @property
    def energy_gap(self) -> torch.Tensor:
        return self.projected_energy - self.teacher_energy


@dataclass(frozen=True)
class AmortizedTeacherBatch:
    """Exact detached states that may be cached while a functional is frozen."""

    state: torch.Tensor
    multipliers: torch.Tensor
    state_ptr: torch.Tensor
    multiplier_ptr: torch.Tensor
    energy: torch.Tensor
    problem_coefficients: Tuple[Tuple[torch.Tensor, ...], ...]

    @property
    def n_conditions(self) -> int:
        return self.state_ptr.numel() - 1


def _detached_problem(problem: QuadraticProblem) -> QuadraticProblem:
    return QuadraticProblem(
        hessian=problem.hessian.detach(),
        linear=problem.linear.detach(),
        constraint_matrix=problem.constraint_matrix.detach(),
        constraint_target=problem.constraint_target.detach(),
        constant=problem.constant.detach(),
    )


def estimate_kkt_multipliers(
    problem: QuadraticProblem,
    state: torch.Tensor,
) -> torch.Tensor:
    """Return the least-squares multiplier for stationarity at ``state``.

    For full-row-rank ``A``, this minimizes
    ``||gradient(state) + A.T @ multiplier||_2``.  It is used only to measure
    the projected proposal; projection multipliers have a different meaning.
    """

    if state.shape != (problem.n_state,):
        raise ValueError("state has the wrong shape")
    if problem.n_constraints == 0:
        return state.new_zeros((0,))
    constraints = problem.constraint_matrix
    gram = constraints @ constraints.transpose(0, 1)
    return torch.linalg.solve(
        gram,
        -(constraints @ problem.gradient(state)),
    )


def _normalized_kkt_vector(problem, state, multipliers):
    residuals = evaluate_kkt(problem, state, multipliers)
    # The scales are certification metadata, not trainable escape routes.  In
    # particular, a proposal must not reduce its loss by merely inflating the
    # normalization denominator.
    stationarity_scale = residuals.stationarity_scale.detach()
    primal_scale = residuals.primal_scale.detach()
    return torch.cat(
        (
            residuals.stationarity / (1.0 + stationarity_scale),
            residuals.primal / (1.0 + primal_scale),
        )
    )


def build_amortized_teacher_batch(
    problems: Sequence[QuadraticProblem],
) -> AmortizedTeacherBatch:
    """Solve and detach exact teachers for a frozen set of functionals.

    Cached teachers must be rebuilt whenever any physical functional
    coefficient or compiled boundary changes.  They are intended for the
    proposal-distillation stage, not for simultaneous functional fitting.
    """

    problems = tuple(problems)
    if not problems:
        raise ValueError("at least one quadratic problem is required")
    reference = problems[0].hessian
    states = []
    multipliers = []
    energies = []
    state_ptr = [0]
    multiplier_ptr = [0]
    coefficients = []
    with torch.no_grad():
        for problem in problems:
            if problem.hessian.dtype != reference.dtype or problem.hessian.device != reference.device:
                raise ValueError("all problems must share dtype and device")
            teacher = solve_dense(_detached_problem(problem))
            states.append(teacher.state.detach())
            multipliers.append(teacher.multipliers.detach())
            energies.append(teacher.energy.detach())
            coefficients.append(
                tuple(
                    tensor.detach().clone()
                    for tensor in (
                        problem.hessian,
                        problem.linear,
                        problem.constraint_matrix,
                        problem.constraint_target,
                        problem.constant,
                    )
                )
            )
            state_ptr.append(state_ptr[-1] + problem.n_state)
            multiplier_ptr.append(multiplier_ptr[-1] + problem.n_constraints)
    device = reference.device
    return AmortizedTeacherBatch(
        state=torch.cat(states),
        multipliers=torch.cat(multipliers),
        state_ptr=torch.tensor(state_ptr, dtype=torch.long, device=device),
        multiplier_ptr=torch.tensor(
            multiplier_ptr, dtype=torch.long, device=device
        ),
        energy=torch.stack(energies),
        problem_coefficients=tuple(coefficients),
    )


def prepare_amortized_training_batch(
    problems: Sequence[QuadraticProblem],
    proposal: torch.Tensor,
    *,
    projection_metrics: Optional[Sequence[Optional[torch.Tensor]]] = None,
    teacher: Optional[AmortizedTeacherBatch] = None,
    detach_functional: bool = True,
) -> AmortizedTrainingBatch:
    """Build differentiable pre-correction losses and detached exact teachers.

    Parameters
    ----------
    problems
        One already boundary-compiled quadratic problem per condition.  The
        physical plugin owns how each problem was constructed.
    proposal
        Flat concatenation of one raw proposal per condition.
    projection_metrics
        Optional positive projection metric per condition.  Metrics affect
        only hard feasibility and do not define the physical Hessian.
    detach_functional
        Detach ``H, g, A, b`` inside the amortization objective.  This defaults
        to true so a bad proposal is trained toward the functional rather than
        changing the functional to accommodate the proposal.  Physical
        energy/force losses should train the functional through a separate
        exact or implicit solve.
    teacher
        Optional cached exact teacher.  This is valid only while the physical
        functionals and boundaries are frozen.  Omitting it recomputes the
        moving teacher from the current problems on every call.
    """

    problems = tuple(problems)
    if not problems:
        raise ValueError("at least one quadratic problem is required")
    if proposal.ndim != 1:
        raise ValueError("proposal must be a flat rank-1 tensor")
    if projection_metrics is None:
        projection_metrics = (None,) * len(problems)
    else:
        projection_metrics = tuple(projection_metrics)
        if len(projection_metrics) != len(problems):
            raise ValueError("projection_metrics must have one entry per problem")

    expected = sum(problem.n_state for problem in problems)
    if proposal.numel() != expected:
        raise ValueError("proposal length does not match the ragged problems")

    reference = problems[0].hessian
    for problem in problems:
        if teacher is None:
            validate_problem(problem)
        if problem.hessian.dtype != reference.dtype or problem.hessian.device != reference.device:
            raise ValueError("all problems must share dtype and device")
    if proposal.dtype != reference.dtype or proposal.device != reference.device:
        raise ValueError("proposal must share the problem dtype and device")
    if teacher is not None:
        if not isinstance(teacher, AmortizedTeacherBatch):
            raise TypeError("teacher must be an AmortizedTeacherBatch")
        expected_state_ptr = [0]
        expected_multiplier_ptr = [0]
        for problem in problems:
            expected_state_ptr.append(expected_state_ptr[-1] + problem.n_state)
            expected_multiplier_ptr.append(
                expected_multiplier_ptr[-1] + problem.n_constraints
            )
        if teacher.state_ptr.tolist() != expected_state_ptr:
            raise ValueError("cached teacher state layout does not match problems")
        if teacher.multiplier_ptr.tolist() != expected_multiplier_ptr:
            raise ValueError("cached teacher multiplier layout does not match problems")
        teacher_tensors = (teacher.state, teacher.multipliers, teacher.energy)
        if any(tensor.dtype != reference.dtype or tensor.device != reference.device for tensor in teacher_tensors):
            raise ValueError("cached teacher must share problem dtype and device")
        if teacher.energy.shape != (len(problems),):
            raise ValueError("cached teacher energy must have one value per problem")
        if len(teacher.problem_coefficients) != len(problems):
            raise ValueError("cached teacher coefficient record does not match problems")
        for problem, cached in zip(problems, teacher.problem_coefficients):
            current = (
                problem.hessian,
                problem.linear,
                problem.constraint_matrix,
                problem.constraint_target,
                problem.constant,
            )
            if len(cached) != len(current) or any(
                not torch.equal(value.detach(), reference_value)
                for value, reference_value in zip(current, cached)
            ):
                raise ValueError(
                    "cached teacher is stale; rebuild it after functional or boundary changes"
                )

    projected_states = []
    teacher_states = []
    projected_multipliers = []
    teacher_multipliers = []
    normalized_kkt = []
    projected_energies = []
    teacher_energies = []
    state_ptr = [0]
    multiplier_ptr = [0]
    kkt_ptr = [0]
    offset = 0

    for condition, (problem, metric) in enumerate(zip(problems, projection_metrics)):
        objective_problem = _detached_problem(problem) if detach_functional else problem
        current = proposal[offset : offset + problem.n_state]
        current_metric = metric
        if current_metric is not None and detach_functional:
            current_metric = current_metric.detach()
        projection = project_to_equalities(
            current,
            objective_problem.constraint_matrix,
            objective_problem.constraint_target,
            metric=current_metric,
        )
        projected = projection.state
        multipliers = estimate_kkt_multipliers(objective_problem, projected)
        kkt = _normalized_kkt_vector(objective_problem, projected, multipliers)

        # The exact state is a moving teacher generated from the current
        # functional, but no gradient from the proposal objective is allowed to
        # update that functional or flow through the solve.
        if teacher is None:
            with torch.no_grad():
                exact = solve_dense(_detached_problem(problem))
            teacher_state = exact.state.detach()
            teacher_multiplier = exact.multipliers.detach()
            teacher_energy = exact.energy.detach()
        else:
            state_start = int(teacher.state_ptr[condition].item())
            state_stop = int(teacher.state_ptr[condition + 1].item())
            multiplier_start = int(teacher.multiplier_ptr[condition].item())
            multiplier_stop = int(teacher.multiplier_ptr[condition + 1].item())
            teacher_state = teacher.state[state_start:state_stop]
            teacher_multiplier = teacher.multipliers[
                multiplier_start:multiplier_stop
            ]
            teacher_energy = teacher.energy[condition]

        projected_states.append(projected)
        teacher_states.append(teacher_state)
        projected_multipliers.append(multipliers)
        teacher_multipliers.append(teacher_multiplier)
        normalized_kkt.append(kkt)
        projected_energies.append(objective_problem.energy(projected))
        teacher_energies.append(teacher_energy)
        offset += problem.n_state
        state_ptr.append(offset)
        multiplier_ptr.append(multiplier_ptr[-1] + problem.n_constraints)
        kkt_ptr.append(kkt_ptr[-1] + kkt.numel())

    device = proposal.device
    return AmortizedTrainingBatch(
        proposal=proposal,
        projected_state=torch.cat(projected_states),
        teacher_state=torch.cat(teacher_states),
        projected_multipliers=torch.cat(projected_multipliers),
        teacher_multipliers=torch.cat(teacher_multipliers),
        normalized_projected_kkt=torch.cat(normalized_kkt),
        state_ptr=torch.tensor(state_ptr, dtype=torch.long, device=device),
        multiplier_ptr=torch.tensor(multiplier_ptr, dtype=torch.long, device=device),
        kkt_ptr=torch.tensor(kkt_ptr, dtype=torch.long, device=device),
        projected_energy=torch.stack(projected_energies),
        teacher_energy=torch.stack(teacher_energies),
    )


__all__ = [
    "AmortizedTeacherBatch",
    "AmortizedTrainingBatch",
    "build_amortized_teacher_batch",
    "estimate_kkt_multipliers",
    "prepare_amortized_training_batch",
]
