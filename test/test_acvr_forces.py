"""Finite-difference and response identities for ACVR force modes."""

import pytest
import torch

from curator.variational import (
    QuadraticProblem,
    differentiate_variational_energy,
    solve_implicit,
)


DTYPE = torch.float64


def _parametric_problem(coordinate):
    hessian = torch.stack(
        (
            torch.stack((2.0 + coordinate, 0.1 * coordinate)),
            torch.stack((0.1 * coordinate, 3.0 - coordinate)),
        )
    )
    linear = torch.stack((coordinate.square(), -0.4 * coordinate))
    constraints = torch.stack(
        (torch.stack((1.0 + 0.2 * coordinate, torch.ones_like(coordinate))),)
    )
    target = torch.stack((0.5 + 0.1 * coordinate,))
    return hessian, linear, constraints, target


def test_exact_envelope_full_and_adjoint_include_geometry_dependent_constraint():
    coordinate = torch.tensor(0.3, dtype=DTYPE, requires_grad=True)
    hessian, linear, constraints, target = _parametric_problem(coordinate)
    result = solve_implicit(
        QuadraticProblem(hessian, linear, constraints, target)
    )
    objective = lambda state: 0.5 * state @ hessian @ state + linear @ state
    constraint = lambda state: constraints @ state - target
    derivatives = [
        differentiate_variational_energy(
            objective_fn=objective,
            constraint_fn=constraint,
            coordinates={"positions": coordinate},
            state=result.state,
            multipliers=result.multipliers,
            mode=mode,
        )
        for mode in ("envelope", "full_differentiable", "adjoint_corrected")
    ]
    native = torch.autograd.grad(result.energy, coordinate)[0]

    for derivative in derivatives:
        torch.testing.assert_close(
            derivative.gradients["positions"], native, atol=3.0e-12, rtol=0.0
        )
    assert derivatives[0].kkt_certified
    assert not derivatives[0].conservative_for_reported_energy
    assert derivatives[1].conservative_for_reported_energy
    assert not derivatives[2].conservative_for_reported_energy

    delta = 1.0e-5
    energies = []
    for sign in (-1.0, 1.0):
        shifted = torch.tensor(0.3 + sign * delta, dtype=DTYPE)
        shifted_h, shifted_g, shifted_a, shifted_b = _parametric_problem(shifted)
        energies.append(
            solve_implicit(
                QuadraticProblem(shifted_h, shifted_g, shifted_a, shifted_b)
            ).energy
        )
    finite_difference = (energies[1] - energies[0]) / (2.0 * delta)
    torch.testing.assert_close(native, finite_difference, atol=2.0e-10, rtol=2.0e-9)


def test_full_minus_envelope_is_stationarity_times_state_response():
    coordinate = torch.tensor(0.7, dtype=DTYPE, requires_grad=True)
    hessian = torch.tensor([[2.0, 0.3], [0.3, 1.5]], dtype=DTYPE) + coordinate * torch.tensor(
        [[0.2, 0.0], [0.0, -0.1]], dtype=DTYPE
    )
    linear = torch.stack((0.4 * coordinate, -0.2 * coordinate))
    state = torch.stack((coordinate.square(), -coordinate.square()))
    constraints = torch.ones((1, 2), dtype=DTYPE)
    target = torch.zeros(1, dtype=DTYPE)
    chemical_potential = hessian @ state + linear
    multipliers = -chemical_potential.mean().reshape(1)
    objective = lambda charge: 0.5 * charge @ hessian @ charge + linear @ charge
    constraint = lambda charge: constraints @ charge - target

    envelope = differentiate_variational_energy(
        objective_fn=objective,
        constraint_fn=constraint,
        coordinates={"positions": coordinate},
        state=state,
        multipliers=multipliers,
        mode="envelope",
    )
    full = differentiate_variational_energy(
        objective_fn=objective,
        constraint_fn=constraint,
        coordinates={"positions": coordinate},
        state=state,
        multipliers=multipliers,
        mode="full_differentiable",
    )
    expected_correction = torch.autograd.grad(
        state,
        coordinate,
        grad_outputs=envelope.stationarity_residual,
        retain_graph=True,
    )[0]

    torch.testing.assert_close(
        full.response_correction["positions"],
        expected_correction,
        atol=2.0e-12,
        rtol=0.0,
    )
    assert not envelope.conservative_for_reported_energy
    assert not envelope.kkt_certified
    assert full.conservative_for_reported_energy


def _approximate_force_errors(epsilon):
    coordinate = torch.tensor(0.4, dtype=DTYPE, requires_grad=True)
    hessian = torch.stack(
        (
            torch.stack((2.0 + coordinate, 0.2 * coordinate)),
            torch.stack((0.2 * coordinate, 1.5 - 0.1 * coordinate)),
        )
    )
    linear = torch.stack((0.3 * coordinate.square(), -0.5 * coordinate))
    constraints = torch.ones((1, 2), dtype=DTYPE)
    target = torch.tensor([0.2], dtype=DTYPE)
    exact = solve_implicit(
        QuadraticProblem(hessian, linear, constraints, target)
    )
    approximate = exact.state.detach() + epsilon * torch.tensor(
        [1.0, -1.0], dtype=DTYPE
    )
    chemical_potential = hessian @ approximate + linear
    multipliers = -chemical_potential.mean().reshape(1)
    objective = lambda charge: 0.5 * charge @ hessian @ charge + linear @ charge
    constraint = lambda charge: constraints @ charge - target
    exact_derivative = differentiate_variational_energy(
        objective_fn=objective,
        constraint_fn=constraint,
        coordinates={"positions": coordinate},
        state=exact.state.detach(),
        multipliers=exact.multipliers.detach(),
        mode="envelope",
    ).gradients["positions"]
    envelope = differentiate_variational_energy(
        objective_fn=objective,
        constraint_fn=constraint,
        coordinates={"positions": coordinate},
        state=approximate,
        multipliers=multipliers,
        mode="envelope",
    ).gradients["positions"]
    adjoint_result = differentiate_variational_energy(
        objective_fn=objective,
        constraint_fn=constraint,
        coordinates={"positions": coordinate},
        state=approximate,
        multipliers=multipliers,
        mode="adjoint_corrected",
    )
    assert not adjoint_result.conservative_for_reported_energy
    adjoint = adjoint_result.gradients["positions"]
    return (envelope - exact_derivative).abs(), (adjoint - exact_derivative).abs()


def test_adjoint_correction_removes_first_order_state_error():
    envelope_large, adjoint_large = _approximate_force_errors(0.1)
    envelope_small, adjoint_small = _approximate_force_errors(0.05)

    torch.testing.assert_close(
        envelope_small / envelope_large,
        torch.tensor(0.5, dtype=DTYPE),
        atol=1.0e-2,
        rtol=0.0,
    )
    torch.testing.assert_close(
        adjoint_small / adjoint_large,
        torch.tensor(0.25, dtype=DTYPE),
        atol=2.0e-12,
        rtol=0.0,
    )


def test_qeq_envelope_contains_chi_hardness_and_kernel_derivatives():
    coordinate = torch.tensor(0.25, dtype=DTYPE, requires_grad=True)
    charge = torch.tensor([0.3, -0.2], dtype=DTYPE)
    chi = torch.stack((coordinate.square(), -0.4 * coordinate))
    hardness = torch.stack((2.0 + coordinate, 1.5 - 0.2 * coordinate))
    kernel = coordinate * torch.tensor([[0.3, -0.1], [-0.1, 0.2]], dtype=DTYPE)
    objective = lambda state: (
        chi @ state
        + 0.5 * (hardness * state.square()).sum()
        + 0.5 * state @ kernel @ state
    )
    derivative = differentiate_variational_energy(
        objective_fn=objective,
        constraint_fn=lambda state: state.new_zeros((0,)),
        coordinates={"positions": coordinate},
        state=charge,
        multipliers=torch.zeros(0, dtype=DTYPE),
        mode="envelope",
    )
    expected = torch.autograd.grad(objective(charge), coordinate)[0]

    torch.testing.assert_close(
        derivative.gradients["positions"], expected, atol=2.0e-12, rtol=0.0
    )


def _geometry_dependent_constraint_errors(epsilon):
    coordinate = torch.tensor(0.4, dtype=DTYPE, requires_grad=True)
    hessian = torch.tensor([[2.0, 0.3], [0.3, 1.5]], dtype=DTYPE)
    linear = torch.tensor([-0.4, 0.2], dtype=DTYPE)
    constraint_row = torch.stack(
        (1.0 + 0.2 * coordinate, 1.0 - 0.1 * coordinate)
    ).reshape(1, 2)
    target = torch.stack((0.3 + 0.15 * coordinate,))
    exact = solve_implicit(
        QuadraticProblem(hessian, linear, constraint_row, target)
    )
    tangent = torch.stack((constraint_row[0, 1], -constraint_row[0, 0]))
    approximate = exact.state.detach() + epsilon * tangent / tangent.norm()
    chemical_potential = hessian @ approximate + linear
    multipliers = -(
        constraint_row @ chemical_potential
    ) / (constraint_row @ constraint_row.transpose(0, 1))
    multipliers = multipliers.reshape(1)
    objective = lambda state: 0.5 * state @ hessian @ state + linear @ state
    constraint = lambda state: constraint_row @ state - target
    exact_gradient = differentiate_variational_energy(
        objective_fn=objective,
        constraint_fn=constraint,
        coordinates={"positions": coordinate},
        state=exact.state.detach(),
        multipliers=exact.multipliers.detach(),
        mode="envelope",
    ).gradients["positions"]
    envelope = differentiate_variational_energy(
        objective_fn=objective,
        constraint_fn=constraint,
        coordinates={"positions": coordinate},
        state=approximate,
        multipliers=multipliers,
        mode="envelope",
    )
    adjoint = differentiate_variational_energy(
        objective_fn=objective,
        constraint_fn=constraint,
        coordinates={"positions": coordinate},
        state=approximate,
        multipliers=multipliers,
        mode="adjoint_corrected",
    )
    assert adjoint.constraint_residual.abs().max() < 2.0e-15
    assert not adjoint.conservative_for_reported_energy
    return (
        (envelope.gradients["positions"] - exact_gradient).abs(),
        (adjoint.gradients["positions"] - exact_gradient).abs(),
    )


def test_adjoint_sign_and_geometry_dependent_constraint_correction():
    envelope_large, adjoint_large = _geometry_dependent_constraint_errors(0.1)
    envelope_small, adjoint_small = _geometry_dependent_constraint_errors(0.05)

    torch.testing.assert_close(
        envelope_small / envelope_large,
        torch.tensor(0.5, dtype=DTYPE),
        atol=2.0e-2,
        rtol=0.0,
    )
    torch.testing.assert_close(
        adjoint_small / adjoint_large,
        torch.tensor(0.25, dtype=DTYPE),
        atol=2.0e-11,
        rtol=0.0,
    )


def test_adjoint_rejects_second_derivatives_until_the_solve_is_differentiable():
    coordinate = torch.tensor(0.5, dtype=DTYPE, requires_grad=True)
    with pytest.raises(ValueError, match="first derivatives only"):
        differentiate_variational_energy(
            objective_fn=lambda state: 0.5
            * (1.0 + coordinate.square())
            * state[0].square(),
            constraint_fn=lambda state: state.new_zeros((0,)),
            coordinates={"positions": coordinate},
            state=torch.zeros(1, dtype=DTYPE),
            multipliers=torch.zeros(0, dtype=DTYPE),
            mode="adjoint_corrected",
            create_graph=True,
        )


def test_full_mode_returns_zero_for_a_detached_coordinate_independent_state():
    coordinate = torch.tensor(0.5, dtype=DTYPE, requires_grad=True)
    derivative = differentiate_variational_energy(
        objective_fn=lambda state: 0.5 * state.square().sum(),
        constraint_fn=lambda state: state.new_zeros((0,)),
        coordinates={"positions": coordinate},
        state=torch.tensor([0.3], dtype=DTYPE),
        multipliers=torch.zeros(0, dtype=DTYPE),
        mode="full_differentiable",
    )

    torch.testing.assert_close(
        derivative.gradients["positions"], torch.zeros_like(coordinate)
    )
    assert derivative.conservative_for_reported_energy
