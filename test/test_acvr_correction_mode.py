"""Initial-charge plus correction integration, including physical gradient ownership."""
import copy
import io

import pytest
import torch

from curator.data import properties
from curator.layer._neural_charge_state_energy import NeuralChargeStateEnergy
from curator.layer._quadratic_taylor_energy import QuadraticTaylorEnergy
from curator.layer._variational_proposal import BoundaryConditionedProposal
from test_acvr_joint_model import _batch, _model, _prediction


def _periodic(positions=None, charge=0.4):
    batch = _batch(positions, charge=charge)
    batch[properties.cell] = torch.eye(3, dtype=torch.float64)[None] * 8
    batch[properties.pbc] = torch.ones(1, 3, dtype=torch.bool)
    return batch


def _correction(mode, backbone='painn'):
    model = _model(backbone=backbone,
                   internal_energy=QuadraticTaylorEnergy(NeuralChargeStateEnergy(hidden_features=4), curvature_mode=mode),
                   proposal=BoundaryConditionedProposal(communication='mp', hidden_features=4, num_layers=2,
                                                         num_heads=2, state_scale=0.05, use_physics_features=False),
                   state_unit='e', conjugate_unit='V', proposal_mode='correction').double()
    readout = model.output_modules[0].internal_energy.base_energy.q0_readout
    with torch.no_grad():
        readout.weight.copy_(torch.linspace(0.05, 0.15, readout.weight.numel(), dtype=torch.float64).reshape_as(readout.weight))
    return model


@pytest.mark.parametrize('mode', ['full_hvp', 'diagonal_ewald'])
@pytest.mark.parametrize('backbone', ['painn', 'mace'])
def test_raw_anchor_feasible_seed_projected_sum_and_shared_residual_state(mode, backbone):
    model = _correction(mode, backbone)
    head = model.output_modules[0]
    contexts = []
    hook = head.proposal.register_forward_pre_hook(lambda module, args: contexts.append(dict(args[0])))
    result = _prediction(model, _periodic())
    hook.remove()
    labels = result['labels']
    initial = labels['initial_variable_value']['values']
    delta = labels['proposal_delta']['values']
    final = labels['variable_value']['values']
    assert initial.abs().max() > 1e-6
    assert (initial.sum() - 0.4).abs() > 1e-6
    expected = initial + delta
    expected = expected + (0.4 - expected.sum()) / initial.numel()
    torch.testing.assert_close(final, expected)
    torch.testing.assert_close(final.sum(), final.new_tensor(0.4))
    assert len(contexts) == 2
    for context in contexts:
        torch.testing.assert_close(context['expansion_state'].reshape(-1), initial)
        torch.testing.assert_close(context['initial_state'].reshape(-1), initial)
        torch.testing.assert_close(context['origin'].reshape(-1), initial + (0.4 - initial.sum()) / initial.numel())
    assert contexts[0]['initial_state'].requires_grad
    assert not contexts[1]['initial_state'].requires_grad
    assert not contexts[1]['expansion_state'].requires_grad
    assert not contexts[1]['scalar'].requires_grad
    assert result['metadata']['proposal_mode'] == 'initial_state_plus_correction'
    assert result['metadata']['initial_variable_value'] == 'raw_geometry_state'
    assert result['metadata']['proposal_delta'] == 'unprojected_correction'


@pytest.mark.parametrize('mode', ['full_hvp', 'diagonal_ewald'])
@pytest.mark.parametrize('supervised', ['condition_energy', 'condition_forces'])
def test_energy_or_force_trains_initial_charge_but_stationarity_only_trains_proposal(mode, supervised):
    model = _correction(mode)
    head = model.output_modules[0]
    initial_parameter = head.internal_energy.base_energy.q0_readout.weight
    result = _prediction(model, _periodic())
    result['labels'][supervised]['values'].square().mean().backward()
    assert initial_parameter.grad is not None
    assert torch.isfinite(initial_parameter.grad).all()
    assert initial_parameter.grad.abs().max() > 1e-12
    for group in model.parameter_groups():
        gradients = [p.grad for p in group.params if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients), group.name
    model.zero_grad(set_to_none=True)
    before = {name: p.detach().clone() for name,p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    result = _prediction(model, _periodic())
    result['residuals']['stationarity']['values'].square().mean().backward()
    for group in model.parameter_groups():
        if group.name == 'proposal':
            assert any(p.grad is not None and p.grad.abs().max() > 0 for p in group.params)
        else:
            assert all(p.grad is None for p in group.params), group.name
    assert initial_parameter.grad is None
    optimizer.step()
    changed = [name for name,p in model.named_parameters() if not torch.equal(p,before[name])]
    assert changed and all(name.startswith('output_modules.0.proposal.') for name in changed)


@pytest.mark.parametrize('mode', ['full_hvp', 'diagonal_ewald'])
def test_complete_correction_force_matches_coordinate_finite_difference_and_reloads(mode):
    model = _correction(mode).eval()
    positions = _periodic()[properties.positions]
    direction = torch.tensor([[0.1, 0.2, -0.1], [-0.2, 0.1, 0.3], [0.1, -0.3, 0.2]], dtype=torch.float64)
    eps = 1e-5
    predictions = [_prediction(model, _periodic(positions + sign * eps * direction)) for sign in [-1, 0, 1]]
    numeric = (predictions[2]['labels']['condition_energy']['values'] - predictions[0]['labels']['condition_energy']['values']).sum() / (2 * eps)
    force = predictions[1]['labels']['condition_forces']['values'].reshape(-1, 3)
    torch.testing.assert_close(numeric, -(force * direction).sum(), rtol=3e-5, atol=1e-8)
    rebuilt = _correction(mode).eval()
    rebuilt.load_state_dict(copy.deepcopy(model.state_dict()))
    restored = _prediction(rebuilt, _periodic())
    for name in ['condition_energy', 'condition_forces', 'variable_value', 'initial_variable_value', 'proposal_delta']:
        torch.testing.assert_close(restored['labels'][name]['values'], predictions[1]['labels'][name]['values'])


def test_legacy_absolute_checkpoint_without_mode_restores_unchanged():
    model = _model().eval()
    expected = _prediction(model, _batch(charge=0.4))
    assert 'initial_variable_value' not in expected['labels']
    legacy = copy.deepcopy(model)
    del legacy.output_modules[0].proposal_mode
    payload = io.BytesIO()
    torch.save(legacy, payload)
    payload.seek(0)
    loaded = torch.load(payload, weights_only=False)
    assert loaded.output_modules[0].proposal_mode == 'absolute'
    actual = _prediction(loaded, _batch(charge=0.4))
    assert set(actual['labels']) == set(expected['labels'])
    assert actual['metadata'] == expected['metadata']
    for name in expected['labels']:
        torch.testing.assert_close(actual['labels'][name]['values'], expected['labels'][name]['values'])
    rebuilt = _model().eval()
    rebuilt.load_state_dict(copy.deepcopy(loaded.state_dict()))
    torch.testing.assert_close(_prediction(rebuilt, _batch(charge=0.4))['labels']['condition_energy']['values'],
                               expected['labels']['condition_energy']['values'])
