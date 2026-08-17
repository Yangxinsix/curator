"""Gradient oracles for the implicit equality-constrained KKT solve."""

import pytest
import torch

from curator.variational.functional import QuadraticProblem
from curator.variational.implicit import solve_implicit


DTYPE = torch.float64
ATOL = 3.0e-10


def _data():
    hessian = torch.tensor(
        [
            [3.2, 0.2, -0.1, 0.0, 0.1],
            [0.2, 2.8, 0.25, 0.1, 0.0],
            [-0.1, 0.25, 2.5, 0.2, -0.05],
            [0.0, 0.1, 0.2, 2.2, 0.15],
            [0.1, 0.0, -0.05, 0.15, 1.9],
        ],
        dtype=DTYPE,
    )
    constraint_matrix = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
        ],
        dtype=DTYPE,
    )
    control_matrix = torch.tensor(
        [
            [1.0, 0.0, -0.5, 0.0, 0.2],
            [0.0, 0.7, 0.0, -0.3, 0.4],
            [0.2, -0.1, 0.3, 0.0, -0.6],
        ],
        dtype=DTYPE,
    )
    base_linear = torch.tensor([0.3, -0.5, 0.8, -0.2, 0.15], dtype=DTYPE)
    constraint_target = torch.tensor([0.4, -0.25], dtype=DTYPE)
    control = torch.tensor([0.7, -0.35, 0.2], dtype=DTYPE)
    return (
        hessian,
        base_linear,
        constraint_target,
        control,
        constraint_matrix,
        control_matrix,
    )


def _symmetric(parameter):
    return 0.5 * (parameter + parameter.T)


def _problem(hessian_parameter, base_linear, target, control, constraints, controls):
    return QuadraticProblem(
        hessian=_symmetric(hessian_parameter),
        linear=base_linear - controls.T @ control,
        constraint_matrix=constraints,
        constraint_target=target,
        constant=hessian_parameter.new_tensor(0.37),
    )


def _torch_kkt_oracle(problem):
    """Pure PyTorch solve whose native autograd is the independent oracle."""

    num_constraints = problem.constraint_matrix.shape[0]
    if num_constraints == 0:
        return (
            torch.linalg.solve(problem.hessian, -problem.linear),
            problem.linear.new_zeros((0,)),
        )
    kkt = torch.cat(
        (
            torch.cat((problem.hessian, problem.constraint_matrix.T), dim=1),
            torch.cat(
                (
                    problem.constraint_matrix,
                    problem.hessian.new_zeros((num_constraints, num_constraints)),
                ),
                dim=1,
            ),
        ),
        dim=0,
    )
    rhs = torch.cat((-problem.linear, problem.constraint_target))
    solution = torch.linalg.solve(kkt, rhs)
    return solution[: problem.hessian.shape[0]], solution[problem.hessian.shape[0] :]


def _generic_loss(state, multipliers):
    state_target = state.new_tensor([0.1, -0.3, 0.2, 0.4, -0.15])
    state_weight = state.new_tensor([1.0, 0.7, 1.4, 0.8, 1.2])
    loss = 0.5 * (state_weight * (state - state_target).square()).sum()
    loss = loss + 0.13 * torch.sin(state).sum()
    if multipliers.numel():
        multiplier_target = multipliers.new_tensor([0.2, -0.1])
        loss = loss + 0.35 * (multipliers - multiplier_target).square().sum()
        loss = loss + 0.07 * torch.cos(multipliers).sum()
    return loss


def _leaf_copies():
    hessian, linear, target, control, constraints, controls = _data()
    return (
        hessian.clone().requires_grad_(),
        linear.clone().requires_grad_(),
        target.clone().requires_grad_(),
        control.clone().requires_grad_(),
        constraints,
        controls,
    )


def test_implicit_forward_and_arbitrary_loss_backward_match_native_autograd_oracle():
    (
        hessian_parameter,
        base_linear,
        target,
        control,
        constraints,
        controls,
    ) = _leaf_copies()
    problem = _problem(
        hessian_parameter,
        base_linear,
        target,
        control,
        constraints,
        controls,
    )
    result = solve_implicit(problem)
    loss = _generic_loss(result.state, result.multipliers)
    implicit_gradients = torch.autograd.grad(
        loss,
        (hessian_parameter, base_linear, target, control),
    )

    (
        oracle_hessian_parameter,
        oracle_base_linear,
        oracle_target,
        oracle_control,
        _,
        _,
    ) = _leaf_copies()
    oracle_problem = _problem(
        oracle_hessian_parameter,
        oracle_base_linear,
        oracle_target,
        oracle_control,
        constraints,
        controls,
    )
    oracle_state, oracle_multipliers = _torch_kkt_oracle(oracle_problem)
    oracle_loss = _generic_loss(oracle_state, oracle_multipliers)
    oracle_gradients = torch.autograd.grad(
        oracle_loss,
        (
            oracle_hessian_parameter,
            oracle_base_linear,
            oracle_target,
            oracle_control,
        ),
    )

    torch.testing.assert_close(result.state, oracle_state, atol=ATOL, rtol=0.0)
    torch.testing.assert_close(
        result.multipliers,
        oracle_multipliers,
        atol=ATOL,
        rtol=0.0,
    )
    for implicit, oracle in zip(implicit_gradients, oracle_gradients):
        torch.testing.assert_close(implicit, oracle, atol=2.0e-9, rtol=2.0e-8)

    grad_base_linear = implicit_gradients[1]
    grad_control = implicit_gradients[3]
    torch.testing.assert_close(
        grad_control,
        -(controls @ grad_base_linear),
        atol=ATOL,
        rtol=0.0,
    )


def test_minimized_energy_gradients_are_the_expected_conjugate_quantities():
    (
        hessian_parameter,
        base_linear,
        target,
        control,
        constraints,
        controls,
    ) = _leaf_copies()
    problem = _problem(
        hessian_parameter,
        base_linear,
        target,
        control,
        constraints,
        controls,
    )
    result = solve_implicit(problem)

    gradients = torch.autograd.grad(
        result.energy,
        (hessian_parameter, base_linear, target, control),
    )

    expected_hessian_gradient = 0.5 * torch.outer(result.state, result.state)
    torch.testing.assert_close(
        gradients[0],
        expected_hessian_gradient,
        atol=2.0e-9,
        rtol=2.0e-8,
    )
    torch.testing.assert_close(
        gradients[1],
        result.state,
        atol=2.0e-9,
        rtol=2.0e-8,
    )
    # L = Phi + lambda.T(Ax-b), hence d Phi*/db = -lambda.
    torch.testing.assert_close(
        gradients[2],
        -result.multipliers,
        atol=2.0e-9,
        rtol=2.0e-8,
    )
    # linear = g0 - B.T h, hence d Phi*/dh = -B x.
    torch.testing.assert_close(
        gradients[3],
        -(controls @ result.state),
        atol=2.0e-9,
        rtol=2.0e-8,
    )


def _resolved_loss(hessian_parameter, base_linear, target, control):
    _, _, _, _, constraints, controls = _data()
    problem = _problem(
        hessian_parameter,
        base_linear,
        target,
        control,
        constraints,
        controls,
    )
    state, multipliers = _torch_kkt_oracle(problem)
    return _generic_loss(state, multipliers)


@pytest.mark.parametrize("parameter_index", [0, 1, 2, 3], ids=["H", "g", "b", "control"])
def test_implicit_gradient_matches_central_difference_of_re_solved_problem(
    parameter_index,
):
    hessian, base_linear, target, control, _, _ = _data()
    parameters = [hessian, base_linear, target, control]
    differentiable = [value.clone().requires_grad_() for value in parameters]
    _, _, _, _, constraints, controls = _data()
    problem = _problem(*differentiable, constraints, controls)
    result = solve_implicit(problem)
    gradient = torch.autograd.grad(
        _generic_loss(result.state, result.multipliers),
        differentiable[parameter_index],
    )[0]

    generator = torch.Generator().manual_seed(1701 + parameter_index)
    direction = torch.randn(
        parameters[parameter_index].shape,
        dtype=DTYPE,
        generator=generator,
    )
    direction = direction / direction.norm()
    analytic_directional_derivative = (gradient * direction).sum()

    epsilon = 1.0e-6
    resolved_losses = []
    for sign in (-1.0, 1.0):
        perturbed = [value.clone() for value in parameters]
        perturbed[parameter_index] += sign * epsilon * direction
        resolved_losses.append(_resolved_loss(*perturbed))
    finite_difference = (resolved_losses[1] - resolved_losses[0]) / (2.0 * epsilon)

    torch.testing.assert_close(
        analytic_directional_derivative,
        finite_difference,
        atol=3.0e-7,
        rtol=3.0e-6,
    )


def test_unconstrained_empty_multiplier_backward_matches_native_autograd():
    hessian, linear, _, control, _, controls = _data()
    empty_constraints = torch.empty((0, 5), dtype=DTYPE)
    empty_target = torch.empty(0, dtype=DTYPE)
    hessian_parameter = hessian.clone().requires_grad_()
    base_linear = linear.clone().requires_grad_()
    effective_linear = base_linear - controls.T @ control
    problem = QuadraticProblem(
        hessian=_symmetric(hessian_parameter),
        linear=effective_linear,
        constraint_matrix=empty_constraints,
        constraint_target=empty_target,
        constant=torch.tensor(0.0, dtype=DTYPE),
    )
    result = solve_implicit(problem)
    loss = _generic_loss(result.state, result.multipliers)
    implicit_gradients = torch.autograd.grad(loss, (hessian_parameter, base_linear))

    oracle_hessian_parameter = hessian.clone().requires_grad_()
    oracle_base_linear = linear.clone().requires_grad_()
    oracle_problem = QuadraticProblem(
        hessian=_symmetric(oracle_hessian_parameter),
        linear=oracle_base_linear - controls.T @ control,
        constraint_matrix=empty_constraints,
        constraint_target=empty_target,
        constant=torch.tensor(0.0, dtype=DTYPE),
    )
    oracle_state, oracle_multipliers = _torch_kkt_oracle(oracle_problem)
    oracle_gradients = torch.autograd.grad(
        _generic_loss(oracle_state, oracle_multipliers),
        (oracle_hessian_parameter, oracle_base_linear),
    )

    assert result.multipliers.shape == (0,)
    for implicit, oracle in zip(implicit_gradients, oracle_gradients):
        torch.testing.assert_close(implicit, oracle, atol=2.0e-9, rtol=2.0e-8)


def test_constraint_matrix_adjoint_matches_native_autograd():
    hessian, linear, target, _, constraints, _ = _data()
    implicit_constraints = constraints.clone().requires_grad_()
    implicit_problem = QuadraticProblem(
        hessian=hessian,
        linear=linear,
        constraint_matrix=implicit_constraints,
        constraint_target=target,
    )
    implicit_result = solve_implicit(implicit_problem)
    implicit_gradient = torch.autograd.grad(
        _generic_loss(implicit_result.state, implicit_result.multipliers),
        implicit_constraints,
    )[0]

    oracle_constraints = constraints.clone().requires_grad_()
    oracle_problem = QuadraticProblem(
        hessian=hessian,
        linear=linear,
        constraint_matrix=oracle_constraints,
        constraint_target=target,
    )
    oracle_state, oracle_multipliers = _torch_kkt_oracle(oracle_problem)
    oracle_gradient = torch.autograd.grad(
        _generic_loss(oracle_state, oracle_multipliers), oracle_constraints
    )[0]

    torch.testing.assert_close(
        implicit_gradient, oracle_gradient, atol=2.0e-9, rtol=2.0e-8
    )


def test_unstable_stationary_point_can_be_returned_but_not_marked_reliable():
    problem = QuadraticProblem.unconstrained(
        torch.diag(torch.tensor([-1.0, 2.0], dtype=DTYPE)),
        torch.zeros(2, dtype=DTYPE),
    )
    result = solve_implicit(problem, require_stable=False)

    assert result.minimum_reduced_curvature < 0.0
    assert not result.reliable
