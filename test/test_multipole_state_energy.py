"""Symmetry, derivatives and residual routing for multipole state modules."""
from types import SimpleNamespace

import pytest
import torch
from e3nn import o3
from torch.func import functional_call

from curator.layer._global_state_blocks import STATE_IRREPS, prepare_state_geometry
from curator.layer._multipole_state_energy import NeuralMultipoleStateEnergy
from curator.layer._multipole_state_proposal import MultipoleStateProposal


@pytest.fixture(autouse=True)
def double_precision():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(41)
    yield
    torch.set_default_dtype(old)


def example():
    spec = SimpleNamespace(irreps=o3.Irreps("3x0e + 2x1o + 1x2e"))
    context = {
        "features": (0.2 * torch.randn(3, spec.irreps.dim)).requires_grad_(),
        "positions": torch.tensor([[0., 0., 0.], [1.2, .3, -.2], [-.4, 1.1, .5]], requires_grad=True),
        "pbc": torch.zeros(3, dtype=torch.bool),
        "origin": 0.1 * torch.randn(3, 4),
        "network_field": 0.2 * torch.randn(3, 4),
    }
    kwargs = dict(hidden_irreps="8x0e + 4x1o + 2x2e", num_layers=2, num_heads=2, radial_features=4)
    energy = NeuralMultipoleStateEnergy(**kwargs).bind(spec)
    proposal = MultipoleStateProposal(**kwargs).bind(spec)
    state = (0.15 * torch.randn(3, 4)).requires_grad_()
    return spec, context, energy, proposal, state


@pytest.mark.parametrize("reflection", [False, True])
def test_o3_energy_invariance_and_proposal_covariance(reflection):
    spec, context, energy, proposal, state = example()
    rotation = o3.rand_matrix()
    if reflection:
        rotation = -rotation
    feature_rotation = spec.irreps.D_from_matrix(rotation)
    state_rotation = STATE_IRREPS.D_from_matrix(rotation)
    transformed = {
        **context,
        "positions": context["positions"] @ rotation.T,
        "features": context["features"] @ feature_rotation.T,
        "origin": context["origin"] @ state_rotation.T,
        "network_field": context["network_field"] @ state_rotation.T,
    }
    torch.testing.assert_close(energy(transformed, state @ state_rotation.T), energy(context, state),
                               rtol=2e-6, atol=2e-9)
    torch.testing.assert_close(proposal(transformed), proposal(context) @ state_rotation.T,
                               rtol=2e-6, atol=2e-8)


def test_translation_permutation_and_boundary_independent_internal_energy():
    _, context, energy, proposal, state = example()
    permutation = torch.tensor([2, 0, 1])
    reordered = {key: value[permutation] if key != "pbc" else value
                 for key, value in context.items()}
    reordered["positions"] = reordered["positions"] + torch.tensor([8., -2., 5.])
    torch.testing.assert_close(energy(reordered, state[permutation]), energy(context, state))
    torch.testing.assert_close(proposal(reordered), proposal(context)[permutation])
    changed = {**context, "origin": 10 * context["origin"], "network_field": 3 * context["network_field"]}
    torch.testing.assert_close(energy(changed, state), energy(context, state), atol=0, rtol=0)
    assert not torch.allclose(proposal(changed), proposal(context))


def test_hvp_matches_gradient_difference_and_force_matches_energy_difference():
    _, context, energy, _, state = example()
    cached = prepare_state_geometry(context)
    value = energy(cached, state)
    gradient, = torch.autograd.grad(value, state, create_graph=True)
    direction = torch.randn_like(state)
    hvp, = torch.autograd.grad((gradient * direction).sum(), state, create_graph=True)
    epsilon = 1e-5
    plus = (state.detach() + epsilon * direction).requires_grad_()
    minus = (state.detach() - epsilon * direction).requires_grad_()
    gp, = torch.autograd.grad(energy(cached, plus), plus)
    gm, = torch.autograd.grad(energy(cached, minus), minus)
    torch.testing.assert_close(hvp, (gp - gm) / (2 * epsilon), rtol=3e-5, atol=1e-8)
    position_gradient, = torch.autograd.grad(value, context["positions"], retain_graph=True)
    displacement = torch.randn_like(context["positions"])
    ep = energy({**context, "positions": context["positions"] + epsilon * displacement}, state)
    em = energy({**context, "positions": context["positions"] - epsilon * displacement}, state)
    torch.testing.assert_close((position_gradient * displacement).sum(), (ep - em) / (2 * epsilon),
                               rtol=3e-5, atol=1e-8)
    third, = torch.autograd.grad(hvp.square().sum(), state)
    assert torch.isfinite(third).all()
    assert float(third.abs().max()) > 0

def test_residual_routes_to_proposal_without_energy_or_backbone_gradients():
    _, context, energy, proposal, _ = example()
    live_context = prepare_state_geometry(context)
    frozen_context = {key: value.detach() for key, value in live_context.items()}
    state = proposal(frozen_context)
    frozen_parameters = {name: value.detach() for name, value in energy.named_parameters()}
    frozen_buffers = {name: value.detach() for name, value in energy.named_buffers()}
    value = functional_call(energy, (frozen_parameters, frozen_buffers), (frozen_context, state))
    gradient, = torch.autograd.grad(value, state, create_graph=True)
    # The charge tangent removes only its spatial mean; dipoles are unconstrained.
    charge = gradient[:, :1] - gradient[:, :1].mean()
    loss = charge.square().sum() + gradient[:, 1:].square().sum()
    loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in proposal.parameters())
    assert all(p.grad is None for p in energy.parameters())
    assert context["features"].grad is None
    assert context["positions"].grad is None


def test_pbc_rejected_and_checkpoint_roundtrip():
    spec, context, energy, proposal, state = example()
    periodic = {**context, "pbc": torch.ones(3, dtype=torch.bool)}
    with pytest.raises(ValueError, match="nonperiodic"):
        energy(periodic, state)
    with pytest.raises(ValueError, match="nonperiodic"):
        proposal(periodic)
    kwargs = dict(hidden_irreps="8x0e + 4x1o + 2x2e", num_layers=2, num_heads=2, radial_features=4)
    other = NeuralMultipoleStateEnergy(**kwargs).bind(spec)
    other.load_state_dict(energy.state_dict())
    torch.testing.assert_close(other(context, state), energy(context, state), atol=0, rtol=0)
    assert all(torch.isfinite(p).all() for p in energy.parameters())
    assert all(torch.isfinite(p).all() for p in proposal.parameters())
