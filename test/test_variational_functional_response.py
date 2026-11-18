"""Matrix-free response and a predicted-vs-relaxed phonon curvature counterexample."""

import pytest
import torch

from curator.variational.functionals import FunctionalProblem
from curator.variational.functional_response import solve_functional_linear_response


def test_nonlinear_constrained_response_uses_hvp_without_dense_hessian(monkeypatch):
    state = torch.tensor([1.0, 2.0], dtype=torch.float64)
    matrix = state.new_tensor([[1.0, 1.0]])
    problem = FunctionalProblem(lambda x: x.square().sum() + 0.25 * x.pow(4).sum(),
                                2, matrix, state.new_tensor([3.0]))

    def forbidden(*args, **kwargs):
        raise AssertionError("dense Hessian/nullspace must not be constructed")

    monkeypatch.setattr(torch.autograd.functional, "hessian", forbidden)
    monkeypatch.setattr(torch.linalg, "svd", forbidden)
    result = solve_functional_linear_response(
        problem, state, state.new_tensor([[1.0, 0.0], [0.0, 0.0]]),
        state.new_tensor([[0.0, 1.0]]),
    )
    assert result.converged and result.status == "converged"
    torch.testing.assert_close(result.state, state.new_tensor([[1 / 19, 14 / 19], [-1 / 19, 5 / 19]]))
    torch.testing.assert_close(result.multipliers, state.new_tensor([[14 / 19, -70 / 19]]))
    assert result.residual_norm.max() < 1e-12


def test_predicted_curvature_six_and_relaxed_curvature_four_despite_zero_residual():
    # Phi(u,x)=2u^2+(x-u)^2; x*=u, whereas the proposal always predicts x=0.
    u = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    x = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    energy = 2 * u.square() + (x[0] - u).square()
    gradient_x, = torch.autograd.grad(energy, x, create_graph=True, retain_graph=True)
    derivative_u, = torch.autograd.grad(energy, u, create_graph=True, retain_graph=True)
    clamped_curvature, = torch.autograd.grad(derivative_u, u, retain_graph=True)
    mixed, = torch.autograd.grad(gradient_x.sum(), u, retain_graph=True)
    assert gradient_x.item() == 0
    assert clamped_curvature.item() == 6
    problem = FunctionalProblem(lambda y: 2 * u.square() + (y[0] - u).square(), 1)
    response = solve_functional_linear_response(problem, x.detach(), -mixed.reshape(1))
    assert response.converged and response.state.item() == pytest.approx(1.0)
    relaxed_curvature = clamped_curvature + mixed * response.state[0]
    assert relaxed_curvature.item() == pytest.approx(4.0)


def test_response_matches_finite_perturbations_of_a_nonlinear_minimum():
    # E(x,p)=x^2/2+x^4/4-p*x; at x=1, p=2 and dx/dp=1/4.
    state = torch.ones(1, dtype=torch.float64)
    problem = FunctionalProblem(lambda x: 0.5 * x.square().sum() + 0.25 * x.pow(4).sum() - 2 * x.sum(), 1)
    response = solve_functional_linear_response(problem, state, torch.ones_like(state))

    def root(field):
        value = torch.ones_like(state)
        for _ in range(8):
            value = value - (value + value.pow(3) - field) / (1 + 3 * value.square())
        return value

    epsilon = 1e-5
    finite_difference = (root(2 + epsilon) - root(2 - epsilon)) / (2 * epsilon)
    torch.testing.assert_close(response.state, finite_difference, rtol=1e-9, atol=1e-10)


def test_nonpositive_curvature_is_reported_instead_of_claiming_convergence():
    state = torch.zeros(2, dtype=torch.float64)
    problem = FunctionalProblem(lambda x: -x.square().sum(), 2)
    result = solve_functional_linear_response(problem, state, torch.ones_like(state))
    assert not result.converged
    assert result.status == "nonpositive_curvature"


def test_optional_response_graph_reaches_curvature_parameters():
    curvature = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    state = torch.zeros(1, dtype=torch.float64)
    problem = FunctionalProblem(lambda x: 0.5 * curvature * x.square().sum(), 1)
    response = solve_functional_linear_response(problem, state, torch.ones_like(state), create_graph=True)
    derivative, = torch.autograd.grad(response.state.sum(), curvature)
    assert derivative.item() == pytest.approx(-0.25)
