"""Matrix-free projected-CG tests against the dense ACVR oracle."""

import pytest
import torch

from curator.variational import (
    BoundarySpec,
    ControlSpec,
    EnvironmentSpec,
    GroupSpec,
    LinearTermSpec,
    QuadraticProblem,
    VariableLayout,
    VariableSpec,
    compile_boundary,
    solve_dense,
)
from curator.variational.backends.iterative import (
    solve_projected_cg,
    solve_projected_cg_problem,
)


DTYPE = torch.float64
ATOL = 2.0e-10


class MatvecOnlyQuadratic:
    """Test operator whose dense kernel is deliberately not exposed."""

    def __init__(self, matrix):
        self.matrix = matrix
        self.matvec_calls = 0

    def kernel(self):
        raise AssertionError("the iterative solver must not request a dense kernel")

    def matvec(self, state):
        self.matvec_calls += 1
        return self.matrix @ state

    def energy(self, state):
        return 0.5 * state @ self.matvec(state)


def _empty_constraints(n_state):
    return (
        torch.empty((0, n_state), dtype=DTYPE),
        torch.empty((0,), dtype=DTYPE),
    )


def _spd_matrix(n_state, seed):
    generator = torch.Generator().manual_seed(seed)
    factor = torch.randn((n_state, n_state), dtype=DTYPE, generator=generator)
    return factor.transpose(0, 1) @ factor + 0.8 * torch.eye(n_state, dtype=DTYPE)


def test_unconstrained_matrix_free_cg_matches_dense_oracle():
    n_state = 9
    hessian = _spd_matrix(n_state, seed=11)
    linear = torch.linspace(-0.7, 0.9, n_state, dtype=DTYPE)
    constraints, target = _empty_constraints(n_state)
    dense = solve_dense(
        QuadraticProblem(hessian, linear, constraints, target, constant=0.35)
    )
    operator = MatvecOnlyQuadratic(hessian)

    result = solve_projected_cg(
        operator,
        linear,
        constant=0.35,
        rtol=1.0e-12,
        atol=1.0e-13,
        reduced_stability_certified=True,
    )

    assert result.converged
    assert result.reliable
    assert result.iterations > 0
    assert result.multipliers.shape == (0,)
    assert operator.matvec_calls > 0
    torch.testing.assert_close(result.state, dense.state, atol=ATOL, rtol=1.0e-10)
    torch.testing.assert_close(result.energy, dense.energy, atol=ATOL, rtol=1.0e-10)


def test_constrained_cg_projects_initial_state_and_matches_dense_oracle():
    n_state = 8
    hessian = _spd_matrix(n_state, seed=19)
    linear = torch.linspace(0.8, -0.5, n_state, dtype=DTYPE)
    constraints = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        dtype=DTYPE,
    )
    target = torch.tensor([0.4, -0.25], dtype=DTYPE)
    initial = torch.linspace(-2.0, 1.5, n_state, dtype=DTYPE)
    dense = solve_dense(QuadraticProblem(hessian, linear, constraints, target))

    result = solve_projected_cg(
        MatvecOnlyQuadratic(hessian),
        linear,
        constraints,
        target,
        initial_state=initial,
        rtol=1.0e-12,
        atol=1.0e-13,
        reduced_stability_certified=True,
    )

    assert result.converged
    assert result.reliable
    torch.testing.assert_close(result.state, dense.state, atol=ATOL, rtol=1.0e-10)
    torch.testing.assert_close(
        result.multipliers,
        dense.multipliers,
        atol=ATOL,
        rtol=1.0e-10,
    )
    torch.testing.assert_close(
        constraints @ result.state,
        target,
        atol=2.0e-13,
        rtol=0.0,
    )


def test_constraints_can_remove_a_global_negative_hessian_mode():
    hessian = torch.diag(torch.tensor([-4.0, 1.5, 3.0, 5.0], dtype=DTYPE))
    linear = torch.tensor([0.7, -0.2, 0.8, -1.1], dtype=DTYPE)
    constraints = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=DTYPE)
    target = torch.tensor([0.35], dtype=DTYPE)
    dense = solve_dense(QuadraticProblem(hessian, linear, constraints, target))

    result = solve_projected_cg(
        MatvecOnlyQuadratic(hessian),
        linear,
        constraints,
        target,
        initial_state=torch.tensor([-3.0, 2.0, -1.0, 0.5], dtype=DTYPE),
        rtol=1.0e-13,
        atol=1.0e-13,
        reduced_stability_certified=True,
    )

    assert result.converged
    assert result.reliable
    torch.testing.assert_close(result.state, dense.state, atol=2.0e-12, rtol=0.0)
    torch.testing.assert_close(
        result.multipliers,
        dense.multipliers,
        atol=2.0e-12,
        rtol=0.0,
    )


def test_exact_initial_state_converges_without_iterations():
    n_state = 6
    hessian = _spd_matrix(n_state, seed=41)
    linear = torch.linspace(-0.4, 0.6, n_state, dtype=DTYPE)
    constraints = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=DTYPE
    )
    target = torch.tensor([0.2], dtype=DTYPE)
    dense = solve_dense(QuadraticProblem(hessian, linear, constraints, target))

    result = solve_projected_cg(
        MatvecOnlyQuadratic(hessian),
        linear,
        constraints,
        target,
        initial_state=dense.state,
        rtol=1.0e-10,
        atol=1.0e-11,
        reduced_stability_certified=True,
    )

    assert result.converged
    assert result.reliable
    assert result.iterations == 0
    torch.testing.assert_close(result.state, dense.state, atol=2.0e-12, rtol=0.0)


def test_max_iter_failure_is_reported_and_never_marked_reliable():
    hessian = torch.diag(torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=DTYPE))
    linear = torch.ones(4, dtype=DTYPE)
    constraints, target = _empty_constraints(4)

    result = solve_projected_cg(
        MatvecOnlyQuadratic(hessian),
        linear,
        constraints,
        target,
        rtol=0.0,
        atol=1.0e-15,
        max_iter=1,
        reduced_stability_certified=True,
    )

    assert not result.converged
    assert not result.reliable
    assert result.iterations == 1
    assert result.residual_norm > 1.0e-8
    assert "maximum" in result.reason


def test_large_warm_start_cannot_relax_the_final_reliability_certificate():
    hessian = torch.diag(torch.tensor([1.0, 1.0e20], dtype=DTYPE))
    result = solve_projected_cg(
        MatvecOnlyQuadratic(hessian),
        torch.tensor([1.0, 0.0], dtype=DTYPE),
        initial_state=torch.tensor([0.0, 1.0], dtype=DTYPE),
        rtol=1.0e-10,
        reduced_stability_certified=True,
    )

    assert not result.converged
    assert not result.reliable
    assert result.residual_norm > 0.9


def test_stationarity_without_reduced_stability_is_not_reliable():
    hessian = torch.diag(torch.tensor([-1.0, 2.0], dtype=DTYPE))
    result = solve_projected_cg(
        MatvecOnlyQuadratic(hessian),
        torch.tensor([0.0, 2.0], dtype=DTYPE),
    )

    assert result.converged
    assert not result.reliable
    assert not result.reduced_stability_certified
    assert "stability" in result.reason


@pytest.mark.parametrize("max_iter", [1.5, True])
def test_invalid_iterative_controls_are_rejected(max_iter):
    with pytest.raises(ValueError, match="nonnegative integer"):
        solve_projected_cg(
            MatvecOnlyQuadratic(torch.eye(2, dtype=DTYPE)),
            torch.ones(2, dtype=DTYPE),
            max_iter=max_iter,
        )


def test_compiled_fixed_control_and_robin_terms_reach_matrix_free_solver():
    variable = VariableSpec(
        name="q",
        kind="atomic_charge",
        unit="e",
        conjugate_unit="V",
    )
    layout = VariableLayout(variable=variable, n_entities=3)
    boundary = BoundarySpec(
        variable=variable,
        groups=(GroupSpec.all("system"),),
        controls=(ControlSpec.group_potential("V", "system", 0.3, unit="V"),),
        environment=(
            EnvironmentSpec(
                name="reservoir",
                terms=(LinearTermSpec(group="system"),),
                target=0.4,
                stiffness=0.7,
                field=-0.1,
                observable_unit="e",
            ),
        ),
    )
    compiled = compile_boundary(boundary, layout)
    hessian = _spd_matrix(3, seed=82)
    linear = torch.tensor([0.2, -0.4, 0.6], dtype=DTYPE)
    dense = solve_dense(compiled.to_problem(hessian, linear, constant=0.25))
    operator_problem = compiled.to_operator_problem(
        MatvecOnlyQuadratic(hessian), linear, constant=0.25
    )

    iterative = solve_projected_cg_problem(
        operator_problem,
        rtol=1.0e-12,
        atol=1.0e-13,
        reduced_stability_certified=True,
    )

    assert iterative.reliable
    torch.testing.assert_close(iterative.state, dense.state, atol=ATOL, rtol=1.0e-10)
    torch.testing.assert_close(iterative.energy, dense.energy, atol=ATOL, rtol=1.0e-10)
