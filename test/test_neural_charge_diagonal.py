"""Exact structured diagonal must match dense test-only Hessians and high-order AD."""
import pytest
import torch
from curator.layer._neural_charge_energy import NeuralChargeEnergy


@pytest.mark.parametrize('n_atoms', [1, 5])
@pytest.mark.parametrize('zero_state', [False, True])
def test_exact_diagonal_matches_dense_and_geometry_parameter_gradients(n_atoms, zero_state):
    torch.manual_seed(349)
    energy = NeuralChargeEnergy(scalar_channels=3, hidden_features=4).double()
    position = torch.randn(n_atoms, 3, dtype=torch.float64, requires_grad=True)
    context = {'scalar': torch.sin(position) + 0.1 * position.square()}
    state = torch.randn(n_atoms, 1, dtype=torch.float64, requires_grad=True)
    if zero_state:
        state = torch.zeros_like(state, requires_grad=True)
    reference = torch.autograd.functional.hessian(
        lambda charge: energy(context, charge), state, create_graph=True)
    reference = reference.reshape(n_atoms, n_atoms).diagonal().reshape_as(state)
    count = sum(p.numel() for p in energy.parameters())
    diagonal = energy.state_hessian_diagonal(context, state, create_graph=True)
    torch.testing.assert_close(diagonal, reference, rtol=1e-12, atol=1e-12)
    direction = torch.linspace(0.3, 1.2, n_atoms, dtype=torch.float64).reshape_as(state)
    targets = (position, state, *energy.parameters())
    actual = torch.autograd.grad((diagonal * direction).sum(), targets, retain_graph=True)
    expected = torch.autograd.grad((reference * direction).sum(), targets, retain_graph=True)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=2e-11, atol=2e-12)
    assert sum(p.numel() for p in energy.parameters()) == count


def test_production_diagonal_uses_no_dense_derivative_interface(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('dense derivative interface called')
    for name in ('hessian', 'jacobian'):
        monkeypatch.setattr(torch.autograd.functional, name, forbidden)
    for name in ('hessian', 'jacfwd', 'jacrev'):
        monkeypatch.setattr(torch.func, name, forbidden)
    torch.manual_seed(349)
    energy = NeuralChargeEnergy(scalar_channels=3, hidden_features=4).double()
    scalar = torch.randn(257, 3, dtype=torch.float64, requires_grad=True)
    state = torch.zeros(257, 1, dtype=torch.float64)
    with torch.no_grad():
        diagonal = energy.state_hessian_diagonal({'scalar': scalar}, state)
    assert diagonal.shape == state.shape and diagonal.requires_grad
    diagonal.sum().backward()
    assert scalar.grad is not None and torch.isfinite(scalar.grad).all()
    assert not energy.state_hessian_diagonal({'scalar': scalar}, state, create_graph=False).requires_grad
