"""Loaded inference checkpoints must enter Curator's normalized training path."""
from functools import partial

import pytest
import torch
from pytorch_lightning import Callback, Trainer
from torch import nn
from torch.utils.data import DataLoader

from curator.data import properties as p
from curator.data.properties import HeadConfig
from curator.layer import GlobalRescaleShift
from curator.model.base import NeuralNetworkPotential
from curator.model.lit_module import LitNNP
from curator.train.model_output import ModelOutput


class ConstantEnergy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.frozen = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.independent_mode = nn.Identity()

    def forward(self, data):
        data[p.energy] = (self.weight + self.frozen).reshape(1)
        return data


class CheckTrainingMode(Callback):
    def __init__(self, keep_submodule_eval):
        self.keep_submodule_eval = keep_submodule_eval
        self.train_batches = 0
        self.validation_batches = 0

    def on_fit_start(self, trainer, task):
        if self.keep_submodule_eval:
            # User callbacks may intentionally freeze a module's execution mode.
            task.model.representation.independent_mode.eval()

    def on_train_batch_start(self, trainer, task, batch, batch_idx):
        self.train_batches += 1
        assert task.training and task.model.training
        assert task.model.output_modules[0].training
        assert not task.model.representation.frozen.requires_grad
        assert task.model.representation.independent_mode.training != self.keep_submodule_eval

    def on_before_backward(self, trainer, task, loss):
        # Raw prediction=3, normalized target=(13-5)/2=4.
        torch.testing.assert_close(loss, torch.tensor(1.0))

    def on_after_backward(self, trainer, task):
        torch.testing.assert_close(task.model.representation.weight.grad, torch.tensor(-2.0))
        assert task.model.representation.frozen.grad is None

    def on_validation_batch_start(self, trainer, task, batch, batch_idx, dataloader_idx=0):
        self.validation_batches += 1
        assert not task.training and not task.model.training
        assert not task.model.output_modules[0].training


@pytest.mark.parametrize("sanity_steps", [0, 1])
@pytest.mark.parametrize("keep_submodule_eval", [False, True])
def test_fit_restores_saved_eval_model_without_unfreezing_parameters(
    tmp_path, sanity_steps, keep_submodule_eval
):
    scale = GlobalRescaleShift([HeadConfig(
        key=p.energy, scale_by=2.0, shift_by=5.0, atomwise_normalization=False
    )])
    model = NeuralNetworkPotential(ConstantEnergy(), output_modules=[scale], model_outputs=[p.energy])
    model._initialized = True
    checkpoint = tmp_path / "inference.model"
    torch.save(model.eval(), checkpoint)
    restored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert not restored.training and not restored.output_modules[0].training
    task = LitNNP(restored, [ModelOutput(p.energy, nn.MSELoss(), per_atom_loss=True)],
                  partial(torch.optim.SGD, lr=0.1))
    data = {
        p.energy: torch.tensor([13.0]), p.n_atoms: torch.tensor([1]),
        p.Z: torch.tensor([1]), p.image_idx: torch.tensor([0]),
    }
    loader = DataLoader([data], batch_size=None)
    check = CheckTrainingMode(keep_submodule_eval)
    trainer = Trainer(
        default_root_dir=tmp_path, accelerator="cpu", devices=1, max_steps=1,
        num_sanity_val_steps=sanity_steps, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, callbacks=[check],
    )
    trainer.fit(task, loader, loader)
    assert check.train_batches == 1
    assert check.validation_batches == sanity_steps + 1
    torch.testing.assert_close(restored.representation.weight, torch.tensor(2.2))
    torch.testing.assert_close(restored.representation.frozen, torch.tensor(1.0))
    assert not restored.representation.frozen.requires_grad
    assert restored.training and restored.output_modules[0].training
    assert restored.representation.independent_mode.training != keep_submodule_eval
