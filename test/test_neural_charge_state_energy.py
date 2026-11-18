import io
import torch
from curator.layer._neural_charge_energy import NeuralChargeEnergy
from curator.layer._neural_charge_state_energy import NeuralChargeStateEnergy


def invariant_context(positions):
    norm2 = positions.square().sum(-1, keepdim=True)
    return {'scalar': torch.cat([norm2, torch.exp(-norm2), norm2.square()], dim=-1)}


def test_initial_state_is_permutation_equivariant_and_rotation_invariant():
    torch.manual_seed(123)
    model = NeuralChargeStateEnergy(3, hidden_features=4).double()
    assert torch.count_nonzero(model.q0_readout.weight) == 0
    with torch.no_grad():
        model.q0_readout.weight.copy_(torch.tensor([[0.4, -0.2, 0.1]], dtype=torch.float64))
    positions = torch.randn(7, 3, dtype=torch.float64)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    permutation = torch.tensor([3, 1, 5, 0, 6, 2, 4])
    q0 = model.initial_state(invariant_context(positions))
    assert q0.shape == (7, 1)
    torch.testing.assert_close(model.initial_state(invariant_context(positions @ rotation)), q0)
    torch.testing.assert_close(model.initial_state(invariant_context(positions[permutation])), q0[permutation])


def test_readout_has_energy_and_force_gradients_without_charge_targets():
    torch.manual_seed(123)
    model = NeuralChargeStateEnergy(3, hidden_features=4).double()
    positions = (0.3 * torch.randn(5, 3, dtype=torch.float64)).requires_grad_(True)
    context = invariant_context(positions)
    initial = model.initial_state(context)
    assert torch.count_nonzero(initial) == 0
    # Fixed-total-charge projection, as applied by the outer ACVR head.
    state = initial - initial.mean()
    energy = model(context, state)
    energy_gradient, = torch.autograd.grad(energy, model.q0_readout.weight, retain_graph=True)
    forces, = torch.autograd.grad(-energy, positions, create_graph=True)
    # A nonzero physical force target tests learnability from zero initialization.
    target = torch.linspace(-0.2, 0.3, forces.numel(), dtype=torch.float64).reshape_as(forces)
    force_gradient, = torch.autograd.grad((forces - target).square().sum(), model.q0_readout.weight)
    assert torch.isfinite(energy_gradient).all() and energy_gradient.norm() > 0
    assert torch.isfinite(force_gradient).all() and force_gradient.norm() > 0


def test_shared_scalar_energy_and_checkpoint_contract_are_unchanged():
    torch.manual_seed(123)
    base = NeuralChargeEnergy(3, hidden_features=4).double()
    model = NeuralChargeStateEnergy(3, hidden_features=4).double()
    state_dict = {k: v for k, v in base.state_dict().items() if k != '_extra_state'}
    result = model.load_state_dict(state_dict, strict=False)
    assert set(result.missing_keys) == {'_extra_state', 'q0_readout.weight'}
    assert not result.unexpected_keys
    context = {'scalar': torch.randn(5, 3, dtype=torch.float64)}
    state = torch.randn(5, 1, dtype=torch.float64)
    torch.testing.assert_close(model(context, state), base(context, state), rtol=0, atol=0)
    torch.testing.assert_close(model.state_hessian_diagonal(context, state),
                               base.state_hessian_diagonal(context, state), rtol=0, atol=0)
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in base.parameters()) == 3
    stream = io.BytesIO()
    torch.save(model, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    torch.testing.assert_close(restored.initial_state(context), model.initial_state(context))
