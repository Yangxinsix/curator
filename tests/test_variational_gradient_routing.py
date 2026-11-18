"""Replay equivalence, accumulation scaling and disconnected-gradient semantics."""
import copy
from functools import partial

import pytest
import torch
from torch import nn

from curator.data import properties
from curator.train.variational_losses import VariationalJointLoss, VariationalModelOutput
from curator.train.variational_optimization import VariationalGradientTask
from curator.variational.functionals import FunctionalProblem


def test_cached_stationarity_reuses_nonleaf_state_graph():
    calls = []
    parameter = torch.tensor([.4, -.2, .8], dtype=torch.float64, requires_grad=True)
    state = parameter.square() + .1

    def energy(q):
        calls.append(1)
        return (.5 * q.square() + .3 * q.pow(4)).sum()

    problem = FunctionalProblem(energy, 3, torch.ones(1, 3, dtype=torch.float64),
                                torch.ones(1, dtype=torch.float64))
    value = problem.energy(state)
    cached = problem.stationarity(state, create_graph=True, energy=value)
    assert len(calls) == 1
    expected = problem.stationarity(state, create_graph=True)
    assert len(calls) == 2
    torch.testing.assert_close(cached.stationarity, expected.stationarity, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(cached.stationarity.square().sum(), parameter,
                                     retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.stationarity.square().sum(), parameter)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-13, atol=1e-13)
    with pytest.raises(ValueError, match="original differentiable state"):
        problem.state_gradient(state.detach(), energy=value)


class ToyHead(nn.Module):
    joint_variational_output = True
    residual_gradient_policy = "proposal_only"

    def __init__(self, mode):
        super().__init__()
        self.residual_evaluation = mode
        self.proposal = nn.Linear(1, 1, bias=False).double()
        self.proposal.weight.data.fill_(.3)
        self.proposal.register_parameter("unused", nn.Parameter(torch.tensor(.9, dtype=torch.float64)))
        self.hardness = nn.Parameter(torch.tensor(1.4, dtype=torch.float64))


class ToyModel(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.geometry = nn.Parameter(torch.tensor(.7, dtype=torch.float64))
        self.head = ToyHead(mode)

    def forward(self, batch):
        r = batch[properties.positions].detach().clone().requires_grad_(True)
        h = self.geometry * r[:, 0].square()
        n = h.numel()
        matrix, target = h.new_ones(1, n), h.new_ones(1)

        def energy(q):
            return h.square().sum() + (.5 * self.head.hardness * (1 + h.square()) * q.square()
                                       + h.sin() * q).sum()

        problem = FunctionalProblem(energy, n, matrix, target)
        q = problem.project_state(self.head.proposal(h[:, None]).flatten())
        value = problem.energy(q)
        forces = -torch.autograd.grad(value, r, create_graph=True, retain_graph=True)[0]
        if self.head.residual_evaluation == "replay":
            hd = h.detach()
            replay = FunctionalProblem(lambda z: (.5 * self.head.hardness.detach()
                * (1 + hd.square()) * z.square() + hd.sin() * z).sum(), n, matrix, target)
            qr = replay.project_state(self.head.proposal(hd[:, None]).flatten())
            residual = replay.stationarity(qr, create_graph=True).stationarity
        else:
            residual = problem.stationarity(q, create_graph=True, energy=value).stationarity

        def label(x, unit, shape):
            return {"values": x.flatten(), "ptr": torch.tensor([0, x.numel()]),
                    "shape": (shape,), "unit": unit}

        return {"variational_prediction": {"value_space": "physical", "labels": {
            "condition_energy": label(value, "eV", (1,)),
            "condition_forces": label(forces, "eV/Angstrom", (n, 3))},
            "residuals": {"stationarity": {"values": residual,
                "ptr": torch.tensor([0, n]), "source": "variational_energy_gradient",
                "gradient_policy": "proposal_only", "normalization": {"scale": 1.},
                "requires_parameter_routing": self.head.residual_evaluation == "reuse"}}}}


def batch():
    def target(size, unit, shape):
        return {"values": torch.zeros(size, dtype=torch.float64),
                "ptr": torch.tensor([0, size]), "shape": (shape,), "unit": unit,
                "present": torch.ones(1, dtype=torch.bool), "metadata": ({},)}

    return {properties.n_atoms: torch.tensor([3]),
            properties.positions: torch.tensor([[.3, 0., 0.], [1.2, .4, .3], [-.7, .2, .4]],
                                              dtype=torch.float64),
            "weight": torch.tensor([1.7], dtype=torch.float64),
            "variational": {"condition_weight": torch.tensor([.8], dtype=torch.float64),
                "condition_geometry_index": torch.tensor([0]), "condition_ptr": torch.tensor([0, 1]),
                "boundary_spec": ({"compile_status": "compiled"},),
                "labels": {"condition_energy": target(1, "eV", (1,)),
                           "condition_forces": target(9, "eV/Angstrom", (3, 3))}}}


def objective(residual_only=False, weight=2.3):
    return VariationalModelOutput(VariationalJointLoss(
        supervised_weights={"condition_energy": 0. if residual_only else 1.7,
                            "condition_forces": 0. if residual_only else 2.1},
        supervised_scales={"condition_energy": .9, "condition_forces": 1.2},
        residual_weights={"stationarity": 3.1}, energy_per_atom=True), loss_weight=weight)


@pytest.mark.parametrize("residual_only", [False, True])
@pytest.mark.parametrize("weight", [0., 2.3])
def test_routed_gradients_and_adamw_match_replay(residual_only, weight):
    replay = ToyModel("replay")
    reused = copy.deepcopy(replay)
    reused.head.residual_evaluation = "reuse"
    old_output, new_output = objective(residual_only, weight), objective(residual_only, weight)
    task = VariationalGradientTask(model=reused, outputs=[new_output],
                                   optimizer=partial(torch.optim.AdamW, lr=.01))
    old_optim = torch.optim.AdamW(replay.parameters(), lr=.01, weight_decay=.1)
    new_optim = torch.optim.AdamW(reused.parameters(), lr=.01, weight_decay=.1)
    # Two microbatches in a length-three window test the native incomplete-window
    # scale as well as accumulation into already-existing proposal gradients.
    for _ in range(2):
        data = batch()
        old_pred, new_pred = replay(data), reused(data)
        for name in ("condition_energy", "condition_forces"):
            torch.testing.assert_close(old_pred["variational_prediction"]["labels"][name]["values"],
                                       new_pred["variational_prediction"]["labels"][name]["values"])
        torch.testing.assert_close(old_pred["variational_prediction"]["residuals"]["stationarity"]["values"],
                                   new_pred["variational_prediction"]["residuals"]["stationarity"]["values"])
        loss_old = old_output.calculate_loss(old_pred, data, return_num_obs=False, subset="train")
        losses, _ = task.loss_fn(new_pred, data, "train")
        torch.testing.assert_close(loss_old, losses["train_total_loss"], rtol=0, atol=0)
        (loss_old / 3).backward()
        task.backward(losses["train_total_loss"] / 3)
    for (name, old), (_, new) in zip(replay.named_parameters(), reused.named_parameters()):
        assert (old.grad is None) == (new.grad is None), name
        if old.grad is not None:
            torch.testing.assert_close(old.grad, new.grad, rtol=1e-12, atol=1e-12, msg=name)
        if residual_only and name in ("geometry", "head.hardness"):
            assert new.grad is None
    assert reused.head.proposal.unused.grad is None
    torch.nn.utils.clip_grad_norm_(replay.parameters(), .1)
    torch.nn.utils.clip_grad_norm_(reused.parameters(), .1)
    old_optim.step()
    new_optim.step()
    for old, new in zip(replay.parameters(), reused.parameters()):
        torch.testing.assert_close(old, new, rtol=1e-12, atol=1e-12)


def test_ordinary_backward_is_rejected_for_live_proposal_only_residuals():
    model, data = ToyModel("reuse"), batch()
    with pytest.raises(RuntimeError, match="VariationalGradientTask"):
        objective().calculate_loss(model(data), data, return_num_obs=False)


@pytest.mark.parametrize("world_size,precision", [(2, "32-true"), (1, "16-mixed"), (1, "bf16-mixed")])
def test_unverified_distributed_and_amp_modes_fail_explicitly(world_size, precision):
    from types import SimpleNamespace
    task = VariationalGradientTask(model=ToyModel("reuse"), outputs=[objective()],
                                   optimizer=partial(torch.optim.Adam, lr=.01))
    task._trainer = SimpleNamespace(world_size=world_size, precision=precision)
    with pytest.raises(ValueError, match="AMP and distributed"):
        task.on_train_start()


def test_all_terms_disabled_preserve_none_and_skip_adamw_decay():
    model = ToyModel("reuse")
    output = objective(residual_only=True)
    output.loss_fn.residual_weights["stationarity"] = 0.
    task = VariationalGradientTask(model=model, outputs=[output],
                                   optimizer=partial(torch.optim.AdamW, lr=.01))
    data = batch()
    losses, _ = task.loss_fn(model(data), data, "train")
    assert losses["train_total_loss"].item() == 0
    task.backward(losses["train_total_loss"])
    assert all(parameter.grad is None for parameter in model.parameters())
    original = [parameter.detach().clone() for parameter in model.parameters()]
    torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1).step()
    for parameter, expected in zip(model.parameters(), original):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)


def test_parameter_alias_into_backbone_is_rejected():
    model = ToyModel("reuse")
    model.shared_alias = model.head.proposal.weight
    task = VariationalGradientTask(model=model, outputs=[objective()],
                                   optimizer=partial(torch.optim.AdamW, lr=.01))
    with pytest.raises(ValueError, match="separate proposal and physical parameters"):
        task._proposal_parameters()
