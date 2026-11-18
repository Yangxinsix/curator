"""Exact structured derivatives preserve energy, forces, training derivatives."""
import pytest
import torch
from curator.layer._geometric_neural_charge_state_energy import GeometricNeuralChargeStateEnergy
from curator.layer._quadratic_taylor_energy import QuadraticTaylorEnergy
from curator.layer._neural_charge_energy import NeuralChargeEnergy

@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
@pytest.mark.parametrize('mode', ['full_hvp', 'diagonal_ewald'])
@pytest.mark.parametrize('periodic', [False, True])
def test_exact_energy_force_and_training_derivatives(dtype, mode, periodic):
    torch.manual_seed(736)
    base = GeometricNeuralChargeStateEnergy(5, hidden_features=7, pair_features=3).to(dtype)
    model = QuadraticTaylorEnergy(base, curvature_mode=mode)
    x = torch.randn(4, 3, dtype=dtype, requires_grad=True)
    scalar = torch.cat((x, x[:, :2].square()), dim=-1)
    anchor = (scalar[:, :1] + 0.3 * scalar[:, 1:2]).tanh()
    state = anchor + 0.2 * x[:, 2:3].sin()
    context = dict(scalar=scalar, positions=x, pbc=torch.full((3,), periodic),
                   cell=7 * torch.eye(3, dtype=dtype), expansion_state=anchor)
    context['ewald_kernel'] = torch.exp(-(x[:, None] - x[None, :]).square().sum(-1))
    params = tuple(base.parameters())
    results = []
    for backend in ('autograd', 'analytic'):
        model.derivative_backend = backend
        energy = model(context, state)
        force, = torch.autograd.grad(energy, x, create_graph=True, retain_graph=True)
        # Includes third mixed derivatives, as required by force training.
        loss = energy.square() * 0.03 + force.square().sum() * 0.07
        derivatives = torch.autograd.grad(loss, (x,) + params, allow_unused=True, retain_graph=True)
        results.append((energy, force, derivatives))
    tolerance = 2e-5 if dtype == torch.float32 else 3e-10
    for index in (0, 1):
        torch.testing.assert_close(results[0][index], results[1][index], atol=tolerance, rtol=tolerance)
    for old, new in zip(results[0][2], results[1][2]):
        if old is None or new is None:
            assert old is new
        else:
            torch.testing.assert_close(old, new, atol=tolerance, rtol=tolerance)

@pytest.mark.parametrize('values', [[-50., -2., 0., 5., 19., 20., 21., 40.]])
def test_softplus_threshold_and_third_derivative(values):
    x = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    value = torch.nn.functional.softplus(x)
    first = torch.autograd.grad(value.sum(), x, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), x, create_graph=True)[0]
    analytic_first, analytic_second = NeuralChargeEnergy._softplus_derivatives(x)
    for a, b in ((first, analytic_first), (second, analytic_second)):
        torch.testing.assert_close(a, b, atol=1e-15, rtol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(analytic_first.sum(), x, retain_graph=True)[0], second)
    torch.testing.assert_close(torch.autograd.grad(second.sum(), x)[0],
                               torch.autograd.grad(analytic_second.sum(), x)[0], atol=1e-15, rtol=1e-10)


def test_old_checkpoint_backend_and_no_new_parameters():
    model = QuadraticTaylorEnergy(NeuralChargeEnergy(3))
    state = torch.randn(3, 1)
    context = {'scalar': torch.randn(3, 3)}
    expected = model(context, state)
    keys = set(model.state_dict())
    del model.derivative_backend
    torch.testing.assert_close(expected, model(context, state))
    model.derivative_backend = 'analytic'
    assert set(model.state_dict()) == keys
    torch.testing.assert_close(expected, model(context, state))

@pytest.mark.parametrize('mode', ['full_hvp', 'diagonal_ewald'])
@pytest.mark.parametrize('with_anchor', [False, True])
def test_analytic_energy_only_honors_no_grad(mode, with_anchor):
    torch.manual_seed(443)
    base = GeometricNeuralChargeStateEnergy(3, hidden_features=4, pair_features=3).double()
    model = QuadraticTaylorEnergy(base, curvature_mode=mode, derivative_backend='analytic')
    positions = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    state = torch.randn(4, 1, dtype=torch.float64, requires_grad=True)
    context = dict(scalar=positions, positions=positions, pbc=torch.zeros(3, dtype=torch.bool),
                   cell=torch.eye(3, dtype=torch.float64), ewald_kernel=torch.eye(4, dtype=torch.float64))
    if with_anchor:
        context['expansion_state'] = positions[:, :1].sin()
    expected = model(context, state)
    assert expected.requires_grad
    with torch.no_grad():
        actual = model(context, state)
    assert not actual.requires_grad
    assert actual.grad_fn is None
    assert all(parameter.requires_grad for parameter in base.parameters())
    torch.testing.assert_close(expected, actual)
    model.derivative_backend = 'autograd'
    with torch.no_grad():
        reference = model(context, state)
    torch.testing.assert_close(actual, reference)
