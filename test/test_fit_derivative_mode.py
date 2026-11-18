"""Fit must restore derivative graphs even when a checkpoint was saved in eval mode."""
import io
from functools import partial
import torch
from torch import nn
from curator.model.lit_module import LitNNP


class DerivativeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.curvature = nn.Parameter(torch.tensor(2.0))
        self.frozen = nn.Parameter(torch.tensor(3.0), requires_grad=False)

    def forward(self, x):
        energy = 0.5 * self.curvature * x.square() + self.frozen * x
        derivative, = torch.autograd.grad(energy.sum(), x, create_graph=self.training)
        return derivative


def test_fit_restores_eval_checkpoint_derivative_graph_without_unfreezing_teacher():
    stream = io.BytesIO()
    torch.save(DerivativeModel().eval(), stream)
    stream.seek(0)
    model = torch.load(stream, weights_only=False)
    assert not model.training
    assert not model(torch.tensor([1.0], requires_grad=True)).requires_grad
    teacher = nn.Linear(1, 1).eval()
    teacher.requires_grad_(False)
    teacher.name = 'frozen_teacher'
    teacher.loss_weight = 0.0
    task = LitNNP(model=model, outputs=[teacher], optimizer=partial(torch.optim.Adam, lr=1e-3))
    task.optimizers = lambda: None
    task._write_log_only = lambda text: None
    frozen_before = model.frozen.detach().clone()
    task.on_train_start()
    assert model.training
    assert not teacher.training
    assert not model.frozen.requires_grad
    derivative = model(torch.tensor([1.0], requires_grad=True))
    assert derivative.requires_grad
    derivative.square().sum().backward()
    assert model.curvature.grad is not None and model.curvature.grad.abs().item() > 0
    assert model.frozen.grad is None
    torch.testing.assert_close(model.frozen, frozen_before)
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
