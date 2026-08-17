"""Deterministic analytic cases for testing boundary and solver semantics."""

from dataclasses import dataclass

import torch

from .functional import QuadraticProblem


@dataclass(frozen=True)
class AnalyticQuadraticCase:
    problem: QuadraticProblem
    exact_state: torch.Tensor
    exact_multipliers: torch.Tensor


def make_consistent_quadratic_case(
    n_state: int,
    n_constraints: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    condition_shift: float = 0.5,
) -> AnalyticQuadraticCase:
    """Construct an SPD problem whose primal and dual solutions are known."""

    if n_state <= 0:
        raise ValueError("n_state must be positive")
    if n_constraints < 0 or n_constraints > n_state:
        raise ValueError("n_constraints must be between zero and n_state")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random = torch.randn((n_state, n_state), generator=generator, dtype=dtype)
    hessian = random.transpose(0, 1) @ random
    hessian = hessian + condition_shift * torch.eye(n_state, dtype=dtype)
    exact_state = torch.randn((n_state,), generator=generator, dtype=dtype)
    if n_constraints:
        while True:
            constraints = torch.randn(
                (n_constraints, n_state), generator=generator, dtype=dtype
            )
            if int(torch.linalg.matrix_rank(constraints).item()) == n_constraints:
                break
    else:
        constraints = torch.zeros((0, n_state), dtype=dtype)
    exact_multipliers = torch.randn(
        (n_constraints,), generator=generator, dtype=dtype
    )
    target = constraints @ exact_state
    linear = -hessian @ exact_state - constraints.transpose(0, 1) @ exact_multipliers
    problem = QuadraticProblem(
        hessian=hessian,
        linear=linear,
        constraint_matrix=constraints,
        constraint_target=target,
        constant=hessian.new_zeros(()),
    )
    return AnalyticQuadraticCase(
        problem=problem,
        exact_state=exact_state,
        exact_multipliers=exact_multipliers,
    )
