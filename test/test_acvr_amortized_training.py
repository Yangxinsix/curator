"""End-to-end training tests for backend-independent amortized proposals."""

import pytest
import torch
from torch import nn

from curator.train import AmortizedTrainingModelOutput, AmortizedVariationalLoss
from curator.train.train import run_step
from curator.variational import (
    QuadraticProblem,
    build_amortized_teacher_batch,
    prepare_amortized_training_batch,
    solve_amortized,
)


DTYPE = torch.float64


def _problem(condition, *, requires_grad=False):
    # Four scalar state entries may equally be interpreted as two entities
    # carrying a two-component internal variable.  There is no electrostatic or
    # Ewald assumption anywhere in this family.
    hessian = torch.tensor(
        [
            [3.0, 0.2, 0.1, 0.0],
            [0.2, 2.5, 0.0, -0.1],
            [0.1, 0.0, 2.2, 0.3],
            [0.0, -0.1, 0.3, 2.8],
        ],
        dtype=DTYPE,
        requires_grad=requires_grad,
    )
    drive = torch.tensor(
        [
            [0.4, -0.2, 0.1],
            [-0.1, 0.3, 0.2],
            [0.2, 0.1, -0.3],
            [-0.3, 0.2, 0.4],
        ],
        dtype=DTYPE,
    )
    linear = torch.tensor([0.2, -0.1, 0.3, -0.2], dtype=DTYPE) + drive @ condition
    if requires_grad:
        linear = linear.detach().requires_grad_(True)
    constraints = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
        dtype=DTYPE,
        requires_grad=requires_grad,
    )
    target = torch.stack(
        (0.15 + 0.25 * condition[0], -0.1 + 0.2 * condition[1])
    )
    if requires_grad:
        target = target.detach().requires_grad_(True)
    return QuadraticProblem(
        hessian=hessian,
        linear=linear,
        constraint_matrix=constraints,
        constraint_target=target,
    )


class ConditionedProposalModel(nn.Module):
    def __init__(self, conditions):
        super().__init__()
        # A physics plugin owns this map.  The shared amortization code only
        # receives its flat proposal and never interprets these features.
        self.proposal = nn.Linear(3, 4, dtype=DTYPE)
        self.problems = tuple(_problem(condition) for condition in conditions)
        self.teacher = build_amortized_teacher_batch(self.problems)

    def forward(self, batch):
        conditions = batch["condition_parameters"]
        proposal = self.proposal(conditions).reshape(-1)
        return {
            "amortized_training": prepare_amortized_training_batch(
                self.problems,
                proposal,
                teacher=self.teacher,
            )
        }


def _training_batch():
    generator = torch.Generator().manual_seed(17)
    conditions = 1.5 * (
        2.0 * torch.rand((12, 3), generator=generator, dtype=DTYPE) - 1.0
    )
    return {
        "condition_parameters": conditions,
        "weight": torch.ones(3, dtype=DTYPE),
        "variational": {
            "condition_geometry_index": torch.arange(3).repeat_interleave(4),
            "condition_weight": torch.ones(12, dtype=DTYPE),
        },
    }


def test_curator_run_step_learns_a_zero_correction_solution_map():
    torch.manual_seed(4)
    batch = _training_batch()
    model = ConditionedProposalModel(batch["condition_parameters"])
    loss_fn = AmortizedVariationalLoss(
        state_weight=1.0,
        kkt_weight=0.25,
        state_scale=1.0,
    )
    output = AmortizedTrainingModelOutput(loss_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=7.0e-2)

    with torch.no_grad():
        initial = model(batch)["amortized_training"]
        initial_error = (initial.projected_state - initial.teacher_state).square().mean()

    for _ in range(100):
        run_step(
            model,
            batch,
            [output],
            rescale_layers=[],
            device="cpu",
            optimizer=optimizer,
            stage="train",
        )

    trained = model(batch)["amortized_training"]
    final_error = (trained.projected_state - trained.teacher_state).square().mean()
    assert final_error < initial_error * 2.0e-5
    assert final_error < 3.0e-6

    problems = tuple(_problem(condition) for condition in batch["condition_parameters"])
    for index, problem in enumerate(problems):
        start = int(trained.state_ptr[index])
        stop = int(trained.state_ptr[index + 1])
        result = solve_amortized(
            problem,
            trained.proposal[start:stop],
            residual_tolerance=6.0e-3,
            max_correction_steps=0,
            fallback=False,
        )
        assert result.status == "accepted_projected"
        assert result.iterations == 0


def test_proposal_loss_does_not_move_the_physical_functional_by_default():
    condition = torch.tensor([0.2, -0.4, 0.7], dtype=DTYPE)
    problem = _problem(condition, requires_grad=True)
    proposal = torch.zeros(4, dtype=DTYPE, requires_grad=True)
    training = prepare_amortized_training_batch((problem,), proposal)
    result = AmortizedVariationalLoss(state_weight=1.0, kkt_weight=1.0)(
        training
    )

    result["loss"].backward()
    assert proposal.grad is not None
    assert torch.linalg.vector_norm(proposal.grad) > 0
    assert problem.hessian.grad is None
    assert problem.linear.grad is None
    assert problem.constraint_matrix.grad is None
    assert problem.constraint_target.grad is None


def test_projected_kkt_is_measured_before_exact_teacher_correction():
    condition = torch.tensor([-0.3, 0.5, 0.1], dtype=DTYPE)
    problem = _problem(condition)
    proposal = torch.tensor([2.0, -1.0, 0.5, 0.2], dtype=DTYPE, requires_grad=True)
    training = prepare_amortized_training_batch((problem,), proposal)

    assert torch.linalg.vector_norm(training.normalized_projected_kkt) > 1.0e-3
    assert torch.linalg.vector_norm(training.correction) > 1.0e-3
    loss = training.normalized_projected_kkt.square().mean()
    loss.backward()
    assert proposal.grad is not None
    assert torch.linalg.vector_norm(proposal.grad) > 0


def test_frozen_functional_teacher_cache_matches_a_moving_teacher():
    conditions = (
        torch.tensor([0.1, 0.2, -0.3], dtype=DTYPE),
        torch.tensor([-0.4, 0.5, 0.2], dtype=DTYPE),
    )
    problems = tuple(_problem(condition) for condition in conditions)
    proposal = torch.arange(8, dtype=DTYPE) / 7.0
    cached_teacher = build_amortized_teacher_batch(problems)
    cached = prepare_amortized_training_batch(
        problems, proposal, teacher=cached_teacher
    )
    moving = prepare_amortized_training_batch(problems, proposal)

    torch.testing.assert_close(cached.projected_state, moving.projected_state)
    torch.testing.assert_close(cached.teacher_state, moving.teacher_state)
    torch.testing.assert_close(cached.teacher_multipliers, moving.teacher_multipliers)
    torch.testing.assert_close(cached.teacher_energy, moving.teacher_energy)

    changed = list(problems)
    changed[0] = QuadraticProblem(
        hessian=problems[0].hessian,
        linear=problems[0].linear + 0.01,
        constraint_matrix=problems[0].constraint_matrix,
        constraint_target=problems[0].constraint_target,
    )
    with pytest.raises(ValueError, match="cached teacher is stale"):
        prepare_amortized_training_batch(
            changed, proposal, teacher=cached_teacher
        )
