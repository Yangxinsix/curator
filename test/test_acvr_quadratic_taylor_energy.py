"""Energy-first quadraticization without a dense-Hessian production path."""
import copy

import pytest
import torch
from torch import nn

from curator.layer._quadratic_taylor_energy import QuadraticTaylorEnergy
from test_acvr_joint_model import _batch, _model, _prediction


class AnalyticCoupledEnergy(nn.Module):
    """Nonquadratic analytic test fixture, not a production energy architecture."""
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(0.7, dtype=torch.float64))

    def forward(self, context, state):
        r = context['positions'][:, 0:1]
        t = self.theta
        return (t * context['positions'].square().sum()
                + (t * r.sin() * state).sum()
                + 0.5 * ((2 + t.square() * r.exp()) * state.square()).sum()
                + t * (r * state).sum().square()
                + 0.3 * t * state.pow(3).sum()
                + 0.2 * state.pow(4).sum())


def _context():
    return {'positions': torch.tensor([[0.2, -0.1, 0.3], [0.8, 0.1, -0.2],
                                       [-0.4, 0.5, 0.6]], dtype=torch.float64)}


def _closed(base, context, q, subtract):
    r, t = context['positions'][:, 0:1], base.theta
    return ((0 if subtract else t * context['positions'].square().sum())
            + (t * r.sin() * q).sum()
            + 0.5 * ((2 + t.square() * r.exp()) * q.square()).sum()
            + t * (r * q).sum().square())


@pytest.mark.parametrize('subtract', [False, True])
def test_fixed_zero_taylor_matches_dense_reference_and_is_quadratic(subtract):
    base = AnalyticCoupledEnergy()
    wrapper = QuadraticTaylorEnergy(base, subtract_u0=subtract)
    context = _context()
    q = torch.tensor([[0.3], [-0.7], [0.2]], dtype=torch.float64, requires_grad=True)
    zero = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    fn = lambda x: base(context, x[:, None])
    value0 = fn(zero)
    gradient = torch.autograd.grad(value0, zero, create_graph=True)[0]
    # A dense Hessian is used only as a small-dimensional test oracle.
    hessian = torch.autograd.functional.hessian(fn, zero, create_graph=True)
    expected = (0 if subtract else value0) + gradient @ q[:, 0] + 0.5 * q[:, 0] @ hessian @ q[:, 0]
    torch.testing.assert_close(wrapper(context, q), expected)
    torch.testing.assert_close(wrapper(context, q), _closed(base, context, q, subtract))
    actual_gradient = torch.autograd.grad(wrapper(context, q), q, create_graph=True)[0]
    torch.testing.assert_close(actual_gradient[:, 0], gradient + hessian @ q[:, 0])
    for current in [q.detach()[:, 0], q.detach()[:, 0] * 3 + 1]:
        actual_hessian = torch.autograd.functional.hessian(lambda x: wrapper(context, x[:, None]), current)
        torch.testing.assert_close(actual_hessian, hessian)
    assert not torch.allclose(torch.autograd.functional.hessian(fn, zero),
                              torch.autograd.functional.hessian(fn, q.detach()[:, 0]))


@pytest.mark.parametrize('subtract', [False, True])
def test_force_parameter_and_proposal_direction_gradients_without_dense_hessian(monkeypatch, subtract):
    def forbidden(*args, **kwargs):
        raise AssertionError('production quadratic Taylor path formed a dense Hessian')
    monkeypatch.setattr(torch.autograd.functional, 'hessian', forbidden)
    monkeypatch.setattr(torch.func, 'hessian', forbidden)
    base = AnalyticCoupledEnergy()
    wrapper = QuadraticTaylorEnergy(base, subtract_u0=subtract)
    position = _context()['positions'].clone().requires_grad_(True)
    # The proposal direction depends on geometry AND an independent parameter;
    # detaching it in HVP breaks both derivative comparisons below.
    proposal_weight = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    q = proposal_weight * position[:, 0:1].sin()
    actual = wrapper({'positions': position}, q)
    expected = _closed(base, {'positions': position}, q, subtract)
    variables = (position, base.theta, proposal_weight)
    ag = torch.autograd.grad(actual, variables, create_graph=True, retain_graph=True)
    eg = torch.autograd.grad(expected, variables, create_graph=True, retain_graph=True)
    for left, right in zip(ag, eg):
        torch.testing.assert_close(left, right)
    mixed_actual = torch.autograd.grad(ag[0].square().sum(), (base.theta, proposal_weight), retain_graph=True)
    mixed_expected = torch.autograd.grad(eg[0].square().sum(), (base.theta, proposal_weight))
    for left, right in zip(mixed_actual, mixed_expected):
        torch.testing.assert_close(left, right)
    direction = torch.tensor([[0.1, 0.2, -0.1], [-0.2, 0.1, 0.3], [0.1, -0.3, 0.2]], dtype=torch.float64)
    eps = 1e-5
    energies = []
    for sign in [-1, 1]:
        perturbed = position.detach() + sign * eps * direction
        state = proposal_weight * perturbed[:, 0:1].sin()
        energies.append(wrapper({'positions': perturbed}, state))
    torch.testing.assert_close((energies[1] - energies[0]) / (2 * eps),
                               (ag[0] * direction).sum(), rtol=2e-6, atol=1e-9)


@pytest.mark.parametrize('kind', ['constant', 'linear'])
def test_degenerate_curvature_and_no_grad_evaluation(kind):
    class Degenerate(nn.Module):
        def forward(self, context, state):
            return state.new_tensor(3.) + (state.sum() * 2 if kind == 'linear' else 0)
    q = torch.tensor([[0.2], [-0.1]], dtype=torch.float64, requires_grad=True)
    wrapper = QuadraticTaylorEnergy(Degenerate())
    with torch.no_grad():
        energy = wrapper({}, q)
    expected = 2 * q.sum() if kind == 'linear' else q.sum() * 0
    torch.testing.assert_close(energy, expected)
    torch.testing.assert_close(torch.autograd.grad(energy, q)[0],
                               torch.full_like(q, 2 if kind == 'linear' else 0))


def test_joint_acvr_force_and_residual_gradient_routing_and_checkpoint():
    model = _model(internal_energy=QuadraticTaylorEnergy(AnalyticCoupledEnergy())).double()
    pred = _prediction(model, _batch(charge=0.0))
    (pred['labels']['condition_forces']['values'].square().mean()
     + pred['labels']['condition_energy']['values'].square().mean()).backward()
    for group in model.parameter_groups():
        grads = [p.grad for p in group.params if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
    model.zero_grad(set_to_none=True)
    pred = _prediction(model, _batch(charge=0.0))
    pred['residuals']['stationarity']['values'].square().mean().backward()
    for group in model.parameter_groups():
        if group.name == 'proposal':
            assert any(p.grad is not None and p.grad.abs().max() > 0 for p in group.params)
        else:
            assert all(p.grad is None for p in group.params)
    rebuilt = _model(internal_energy=QuadraticTaylorEnergy(AnalyticCoupledEnergy())).double()
    rebuilt.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(_prediction(rebuilt, _batch(charge=0.0))['labels']['condition_energy']['values'],
                               _prediction(model, _batch(charge=0.0))['labels']['condition_energy']['values'])


class AnalyticDiagonalEnergy(AnalyticCoupledEnergy):
    """Exact diagonal interface fixture; introduces no new parameters."""
    def state_hessian_diagonal(self, context, state, *, create_graph=True):
        r, t = context['positions'][:, 0:1], self.theta
        result = 2 + t.square() * r.exp() + 2 * t * r.square() + 1.8 * t * state + 2.4 * state.square()
        return result if create_graph else result.detach()


@pytest.mark.parametrize('fast_interface', [False, True])
@pytest.mark.parametrize('subtract', [False, True])
def test_diagonal_ewald_uses_same_gradient_and_exact_diagonal_no_dense(monkeypatch, fast_interface, subtract):
    base = AnalyticDiagonalEnergy() if fast_interface else AnalyticCoupledEnergy()
    wrapper = QuadraticTaylorEnergy(base, subtract_u0=subtract, curvature_mode='diagonal_ewald')
    full = QuadraticTaylorEnergy(copy.deepcopy(base), subtract_u0=subtract)
    assert list(dict(wrapper.named_parameters())) == list(dict(full.named_parameters()))
    context = _context()
    context['positions'].requires_grad_(True)
    r = context['positions'][:, 0:1]
    # A geometry-dependent symmetric kernel stands in for J in this algebra test.
    kernel = (-(r - r.T).square()).exp()
    context['ewald_kernel'] = kernel
    q = torch.tensor([[0.3], [-0.7], [0.2]], dtype=torch.float64, requires_grad=True)
    zero = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    fn = lambda x: base(context, x[:, None])
    value0 = fn(zero)
    gradient = torch.autograd.grad(value0, zero, create_graph=True)[0]
    dense = torch.autograd.functional.hessian(fn, zero, create_graph=True)
    expected_curvature = torch.diag(dense.diag()) + kernel
    expected = (0 if subtract else value0) + gradient @ q[:, 0] + 0.5 * q[:, 0] @ expected_curvature @ q[:, 0]
    def forbidden(*args, **kwargs):
        raise AssertionError('diagonal runtime formed a full Hessian')
    monkeypatch.setattr(torch.autograd.functional, 'hessian', forbidden)
    monkeypatch.setattr(torch.func, 'hessian', forbidden)
    actual = wrapper(context, q)
    torch.testing.assert_close(actual, expected)
    av = torch.autograd.grad(actual, (q, context['positions'], base.theta), create_graph=True, retain_graph=True)
    ev = torch.autograd.grad(expected, (q, context['positions'], base.theta), create_graph=True, retain_graph=True)
    for left, right in zip(av, ev):
        torch.testing.assert_close(left, right)
    torch.testing.assert_close(av[0][:, 0], gradient + expected_curvature @ q[:, 0])
    # A directional second derivative checks the complete selected curvature
    # without creating a Hessian, at two independent current proposal states.
    direction = torch.tensor([[0.2], [-0.1], [0.5]], dtype=torch.float64)
    for proposal in [q, (q.detach() * 2 + 1).requires_grad_(True)]:
        g = torch.autograd.grad(wrapper(context, proposal), proposal, create_graph=True)[0]
        action = torch.autograd.grad(g, proposal, grad_outputs=direction, create_graph=True)[0]
        torch.testing.assert_close(action[:, 0], expected_curvature @ direction[:, 0])
    # Force-loss backward must retain mixed parameter derivatives of diag(H).
    af = torch.autograd.grad(av[1].square().sum(), base.theta, retain_graph=True)[0]
    ef = torch.autograd.grad(ev[1].square().sum(), base.theta)[0]
    torch.testing.assert_close(af, ef)


def test_full_mode_never_initializes_or_calls_ewald(monkeypatch):
    from curator.layer import _ewald
    from curator.variational.backends.ewald import EwaldChargeOperator
    def forbidden(*args, **kwargs):
        raise AssertionError('full HVP requested an Ewald kernel')
    monkeypatch.setattr(_ewald.EwaldSummation, '__init__', forbidden)
    monkeypatch.setattr(EwaldChargeOperator, 'geometry_kernels', forbidden)
    wrapper = QuadraticTaylorEnergy(AnalyticCoupledEnergy())
    assert wrapper.ewald is None
    assert wrapper.geometry_context({'n_atoms': torch.tensor([3])}) == [{}]
    value = wrapper(_context(), torch.ones(3, 1, dtype=torch.float64, requires_grad=True))
    value.backward()
    assert base_grad_is_finite(wrapper)


def base_grad_is_finite(wrapper):
    return all(p.grad is not None and torch.isfinite(p.grad).all() for p in wrapper.base_energy.parameters())


def test_diagonal_mode_constructs_existing_ewald_context_and_conservative_force():
    from curator.data import properties
    base = AnalyticDiagonalEnergy()
    wrapper = QuadraticTaylorEnergy(base, curvature_mode='diagonal_ewald')
    context = _context()
    position = context['positions'].clone().requires_grad_(True)
    q = torch.tensor([[0.3], [-0.1], [-0.2]], dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64) * 8
    def value(pos):
        data = {properties.positions: pos, properties.n_atoms: torch.tensor([3]),
                properties.cell: cell[None], properties.pbc: torch.ones(1, 3, dtype=torch.bool)}
        extra = wrapper.geometry_context(data)[0]
        return wrapper(dict(positions=pos, **extra), q)
    energy = value(position)
    derivative = torch.autograd.grad(energy, position, create_graph=True)[0]
    direction = torch.tensor([[0.1, 0.2, -0.1], [-0.2, 0.1, 0.3], [0.1, -0.3, 0.2]], dtype=torch.float64)
    eps = 1e-5
    finite = (value(position.detach() + eps * direction) - value(position.detach() - eps * direction)) / (2 * eps)
    torch.testing.assert_close(finite, (derivative * direction).sum(), rtol=2e-6, atol=1e-8)
    derivative.square().sum().backward()
    assert torch.isfinite(base.theta.grad)


class ContextAnchorEnergy(AnalyticCoupledEnergy):
    """Fixture where context itself shares q0, to distinguish partial from total AD."""
    def forward(self, context, state):
        return super().forward(context, state) + 0.6 * context['expansion_state'].sum() * state.sum()

    def initial_state(self, context):
        return self.theta * context['positions'][:, :1].sin()


def _moving_closed(base, context, state, subtract, mode):
    r, t, q0 = context['positions'][:, :1], base.theta, context['expansion_state']
    delta = state - q0
    gradient = (t * r.sin() + (2 + t.square() * r.exp()) * q0
                + 2 * t * r * (r * q0).sum() + 0.9 * t * q0.square() + 0.8 * q0.pow(3)
                + 0.6 * q0.sum())
    local_diagonal = 2 + t.square() * r.exp() + 1.8 * t * q0 + 2.4 * q0.square()
    energy = base(context, q0) - (base(context, torch.zeros_like(q0)) if subtract else 0)
    energy = energy + (gradient * delta).sum()
    if mode == 'full_hvp':
        energy = energy + 0.5 * (local_diagonal * delta.square()).sum() + t * (r * delta).sum().square()
    else:
        diagonal = local_diagonal + 2 * t * r.square()
        energy = energy + 0.5 * (diagonal * delta.square()).sum()
        energy = energy + 0.5 * state[:, 0] @ context['ewald_kernel'] @ state[:, 0]
    return energy


@pytest.mark.parametrize('mode', ['full_hvp', 'diagonal_ewald'])
@pytest.mark.parametrize('subtract', [False, True])
def test_moving_anchor_preserves_partial_state_and_total_geometry_parameter_gradients(monkeypatch, mode, subtract):
    base = ContextAnchorEnergy()
    wrapper = QuadraticTaylorEnergy(base, subtract_u0=subtract, curvature_mode=mode)
    position = _context()['positions'].clone().requires_grad_(True)
    anchor_weight = torch.tensor(0.35, dtype=torch.float64, requires_grad=True)
    correction_weight = torch.tensor(-0.21, dtype=torch.float64, requires_grad=True)
    def build(pos):
        anchor = anchor_weight * pos[:, :1].tanh() + 0.2 * base.theta
        context = {'positions': pos, 'expansion_state': anchor,
                   'ewald_kernel': (-(pos[:, :1] - pos[:, :1].T).square()).exp()}
        return context, anchor + correction_weight * pos[:, :1].sin()
    context, state = build(position)
    def forbidden(*args, **kwargs):
        raise AssertionError('moving-anchor production path formed a dense Hessian')
    monkeypatch.setattr(torch.autograd.functional, 'hessian', forbidden)
    monkeypatch.setattr(torch.func, 'hessian', forbidden)
    actual = wrapper(context, state)
    expected = _moving_closed(base, context, state, subtract, mode)
    torch.testing.assert_close(actual, expected)
    variables = (position, base.theta, anchor_weight, correction_weight)
    ag = torch.autograd.grad(actual, variables, create_graph=True, retain_graph=True)
    eg = torch.autograd.grad(expected, variables, create_graph=True, retain_graph=True)
    for left, right in zip(ag, eg):
        torch.testing.assert_close(left, right, rtol=1e-9, atol=1e-10)
    actual_force_loss_grad = torch.autograd.grad(ag[0].square().sum(), variables[1:], retain_graph=True)
    expected_force_loss_grad = torch.autograd.grad(eg[0].square().sum(), variables[1:])
    for left, right in zip(actual_force_loss_grad, expected_force_loss_grad):
        torch.testing.assert_close(left, right, rtol=1e-8, atol=1e-9)
    direction = torch.tensor([[0.1, 0.2, -0.1], [-0.2, 0.1, 0.3], [0.1, -0.3, 0.2]], dtype=torch.float64)
    eps = 1e-5
    values = []
    for sign in [-1, 1]:
        perturbed_context, perturbed_state = build(position.detach() + sign * eps * direction)
        values.append(wrapper(perturbed_context, perturbed_state))
    torch.testing.assert_close((values[1] - values[0]) / (2 * eps), (ag[0] * direction).sum(), rtol=2e-6, atol=1e-9)
    # A separate q0 leaf verifies that its total derivative is retained even
    # when the current physical state is an independent variable.
    anchor_leaf = context['expansion_state'].detach().requires_grad_(True)
    independent_context = dict(context, expansion_state=anchor_leaf)
    fixed_state = state.detach()
    a = wrapper(independent_context, fixed_state)
    e = _moving_closed(base, independent_context, fixed_state, subtract, mode)
    torch.testing.assert_close(torch.autograd.grad(a, anchor_leaf)[0], torch.autograd.grad(e, anchor_leaf)[0])


def test_nonzero_anchor_ewald_acts_on_actual_charge_and_contains_jq0():
    base = ContextAnchorEnergy()
    wrapper = QuadraticTaylorEnergy(base, subtract_u0=False, curvature_mode='diagonal_ewald')
    context = _context()
    q0 = torch.tensor([[0.3], [-0.1], [0.2]], dtype=torch.float64)
    kernel = torch.tensor([[2., 0.1, -0.2], [0.1, 1.5, 0.3], [-0.2, 0.3, 1.8]], dtype=torch.float64)
    context.update(expansion_state=q0, ewald_kernel=kernel)
    q = q0.clone().requires_grad_(True)
    value = wrapper(context, q)
    torch.testing.assert_close(value, base(context, q0) + 0.5 * q0[:, 0] @ kernel @ q0[:, 0])
    # Use an independent partial variable to keep context q0 fixed.
    partial = q0.clone().requires_grad_(True)
    partial_gradient = torch.autograd.grad(base(context, partial), partial)[0]
    torch.testing.assert_close(torch.autograd.grad(value, q)[0], partial_gradient + (kernel @ q0[:, 0])[:, None])


@pytest.mark.parametrize('mode', ['full_hvp', 'diagonal_ewald'])
def test_nonzero_context_anchor_is_strictly_quadratic_in_current_state(mode):
    base = ContextAnchorEnergy()
    wrapper = QuadraticTaylorEnergy(base, curvature_mode=mode)
    context = _context()
    context.update(expansion_state=torch.tensor([[0.3], [-0.1], [0.2]], dtype=torch.float64),
                   ewald_kernel=torch.eye(3, dtype=torch.float64))
    fn = lambda q: wrapper(context, q[:, None])
    first = torch.autograd.functional.hessian(fn, torch.tensor([0.2, 0.4, -0.3], dtype=torch.float64))
    second = torch.autograd.functional.hessian(fn, torch.tensor([-0.8, 1.2, 0.6], dtype=torch.float64))
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(wrapper.initial_state(context), base.initial_state(context))
    fallback = QuadraticTaylorEnergy(AnalyticCoupledEnergy())
    torch.testing.assert_close(fallback.initial_state(dict(context, origin=torch.ones(3, 1))), torch.zeros(3, 1))
