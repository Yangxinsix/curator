"""Quadraticity, state derivatives, periodic geometry, and real ACVR routing."""
import copy

import pytest
import torch

from curator.layer._quadratic_charge_energy import QuadraticChargeEnergy
from curator.layer._ewald import EwaldSummation
from curator.variational.functionals import FunctionalProblem
from test_acvr_joint_model import _batch, _model, _prediction


def _context():
    torch.manual_seed(71)
    return {"scalar": torch.randn(5, 4, dtype=torch.float64),
            "positions": torch.randn(5, 3, dtype=torch.float64),
            "cell": torch.tensor([[7., 0., 0.], [0.4, 6., 0.], [0.2, 0.3, 8.]], dtype=torch.float64),
            "pbc": torch.ones(3, dtype=torch.bool)}


def test_shared_local_initialization():
    torch.manual_seed(34)
    a = QuadraticChargeEnergy(4, coupling="ewald").double()
    torch.manual_seed(34)
    b = QuadraticChargeEnergy(4, coupling="learned").double()
    for name, value in a.local.state_dict().items():
        torch.testing.assert_close(value, b.local.state_dict()[name], rtol=0, atol=0)
    assert a.get_extra_state()["coupling"] == "ewald"
    assert b.get_extra_state()["coupling"] == "learned"


@pytest.mark.parametrize("coupling", ["learned", "ewald"])
def test_strict_quadratic_and_autograd_hvp(coupling):
    context = _context()
    energy = QuadraticChargeEnergy(4, coupling=coupling).double()
    if coupling == "ewald":
        # This derivative unit test uses a supplied symmetric kernel. The next
        # integration tests construct Curator's real periodic Ewald kernels.
        factor = torch.randn(5, 3, dtype=torch.float64)
        context["ewald_kernel"] = factor @ factor.T
    q = torch.randn(5, dtype=torch.float64)
    fn = lambda x: energy(context, x[:, None])
    h0 = torch.autograd.functional.hessian(fn, q)
    h1 = torch.autograd.functional.hessian(fn, 3 * q + 2)
    torch.testing.assert_close(h0, h1, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(h0, h0.T)
    assert torch.linalg.eigvalsh(h0).min() >= energy.hardness_floor
    assert (h0 - torch.diag(h0.diag())).abs().max() > 1e-7
    problem = FunctionalProblem(fn, q.numel())
    v = torch.randn_like(q)
    torch.testing.assert_close(problem.hvp(q, v), h0 @ v)
    torch.testing.assert_close(problem.state_gradient(q), problem.state_gradient(torch.zeros_like(q)) + h0 @ q)
    changed = dict(context, field=torch.randn(5, 1), origin=torch.randn(5, 1))
    torch.testing.assert_close(fn(q), energy(changed, q[:, None]))


def test_learned_periodicity_rotation_translation_permutation_and_distance_dependence():
    c = _context()
    energy = QuadraticChargeEnergy(4).double()
    q = torch.randn(5, 1, dtype=torch.float64)
    reference = energy(c, q)
    rotation = torch.tensor([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]], dtype=torch.float64)
    transformed = dict(c, positions=c["positions"] @ rotation.T + 3.2, cell=c["cell"] @ rotation.T)
    torch.testing.assert_close(energy(transformed, q), reference)
    shifts = torch.tensor([[1., 0., -2.], [0., 3., 1.], [-1., 0., 0.], [2., 2., 2.], [0., 0., 0.]], dtype=torch.float64)
    torch.testing.assert_close(energy(dict(c, positions=c["positions"] + shifts @ c["cell"]), q), reference)
    p = torch.tensor([4, 1, 3, 0, 2])
    torch.testing.assert_close(energy(dict(c, positions=c["positions"][p], scalar=c["scalar"][p]), q[p]), reference)
    moved = c["positions"].clone()
    moved[0, 0] += 0.7
    assert abs(float(energy(dict(c, positions=moved), q) - reference)) > 1e-7


def _periodic_batch(positions=None):
    b = _batch(positions, charge=0.0)
    b["cell"] = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 8
    b["pbc"] = torch.ones(1, 3, dtype=torch.bool)
    return b


@pytest.mark.parametrize("coupling", ["learned", "ewald"])
def test_joint_force_and_residual_routing(coupling):
    energy = QuadraticChargeEnergy(coupling=coupling, hidden_features=8, hardness_floor=12)
    model = _model(internal_energy=energy, state_unit="e", conjugate_unit="V").double()
    pred = _prediction(model, _periodic_batch())
    loss = sum(pred["labels"][k]["values"].square().mean() for k in ["condition_energy", "condition_forces"])
    loss.backward()
    for group in model.parameter_groups():
        grads = [p.grad for p in group.params if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), group.name
    model.zero_grad(set_to_none=True)
    pred = _prediction(model, _periodic_batch())
    pred["residuals"]["stationarity"]["values"].square().mean().backward()
    for group in model.parameter_groups():
        if group.name != "proposal":
            assert all(p.grad is None for p in group.params), group.name
        else:
            assert any(p.grad is not None and p.grad.abs().max() > 0 for p in group.params)
    model.eval()
    pos = _periodic_batch()["positions"]
    direction = torch.randn_like(pos)
    eps = 1e-5
    outputs = [_prediction(model, _periodic_batch(pos + s * eps * direction)) for s in [-1, 0, 1]]
    derivative = (outputs[2]["labels"]["condition_energy"]["values"] - outputs[0]["labels"]["condition_energy"]["values"]) / (2 * eps)
    force = outputs[1]["labels"]["condition_forces"]["values"].reshape(-1, 3)
    torch.testing.assert_close(derivative.sum(), -(force * direction).sum(), rtol=3e-5, atol=1e-7)
    rebuilt = _model(internal_energy=QuadraticChargeEnergy(coupling=coupling, hidden_features=8, hardness_floor=12), state_unit="e", conjugate_unit="V").double()
    rebuilt.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(_prediction(rebuilt, _periodic_batch())["labels"]["condition_energy"]["values"],
                               _prediction(model, _periodic_batch())["labels"]["condition_energy"]["values"])
