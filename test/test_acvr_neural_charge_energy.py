"""Functional derivatives, boundary independence, and joint-force integration."""
import copy

import pytest
import torch

from curator.layer._neural_charge_energy import NeuralChargeEnergy
from curator.variational.functionals import FunctionalProblem
from test_acvr_joint_model import _batch, _model, _prediction


def test_energy_has_learned_nonquadratic_global_curvature_and_correct_hvp():
    torch.manual_seed(51)
    energy = NeuralChargeEnergy(4, hidden_features=8, curvature_floor=2).double()
    context = {"scalar": torch.randn(5, 4, dtype=torch.float64)}
    q = torch.linspace(-0.7, 0.8, 5, dtype=torch.float64)
    fn = lambda x: energy(context, x[:, None])
    h0 = torch.autograd.functional.hessian(fn, q)
    h1 = torch.autograd.functional.hessian(fn, q + 1.3)
    assert (h0 - torch.diag(h0.diag())).abs().max() > 1e-6
    assert (h1 - h0).abs().max() > 1e-4
    assert torch.linalg.eigvalsh(h0).min() >= 2 - 1e-10
    direction = torch.randn_like(q)
    problem = FunctionalProblem(fn, q.numel())
    torch.testing.assert_close(problem.hvp(q, direction), h0 @ direction)
    eps = 1e-5
    torch.testing.assert_close(problem.state_gradient(q) @ direction,
                              (fn(q + eps * direction) - fn(q - eps * direction)) / (2 * eps))
    changed = dict(context, field=torch.randn(5, 1), origin=torch.randn(5, 1))
    torch.testing.assert_close(energy(context, q[:, None]), energy(changed, q[:, None]))
    permutation = torch.tensor([4, 1, 3, 0, 2])
    torch.testing.assert_close(fn(q), energy({"scalar": context['scalar'][permutation]}, q[permutation, None]))


@pytest.mark.parametrize('backbone', ['painn', 'mace'])
def test_neural_energy_joint_force_and_residual_routing(backbone):
    model = _model(backbone=backbone, internal_energy=NeuralChargeEnergy(hidden_features=8)).double()
    batch = _batch(charge=0.0)
    pred = _prediction(model, batch)
    loss = sum(pred['labels'][k]['values'].square().mean() for k in ['condition_energy', 'condition_forces'])
    loss.backward()
    for group in model.parameter_groups():
        grads = [p.grad for p in group.params if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), group.name
    model.zero_grad(set_to_none=True)
    pred = _prediction(model, _batch(charge=0.0))
    pred['residuals']['stationarity']['values'].square().mean().backward()
    for group in model.parameter_groups():
        if group.name != 'proposal':
            assert all(p.grad is None for p in group.params)
    model.eval()
    pos = _batch()['positions']
    direction = torch.randn_like(pos)
    eps = 1e-5
    outputs = [_prediction(model, _batch(pos + sign * eps * direction, charge=0.0)) for sign in [-1, 0, 1]]
    diff = (outputs[2]['labels']['condition_energy']['values'] - outputs[0]['labels']['condition_energy']['values']) / (2 * eps)
    force = outputs[1]['labels']['condition_forces']['values'].reshape(-1, 3)
    torch.testing.assert_close(diff.sum(), -(force * direction).sum(), rtol=3e-5, atol=1e-7)
    rebuilt = _model(backbone=backbone, internal_energy=NeuralChargeEnergy(hidden_features=8)).double()
    rebuilt.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(_prediction(rebuilt, _batch(charge=0.0))['labels']['condition_energy']['values'],
                               _prediction(model, _batch(charge=0.0))['labels']['condition_energy']['values'])
