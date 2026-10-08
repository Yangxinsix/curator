"""Exercise Curator scheduling through real, small CPU Lightning runs."""
from functools import partial

import pytest
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader

from curator.model.base import NeuralNetworkPotential
from curator.model.lit_module import LitNNP
from curator.train.callbacks import ValidationScheduler


class TinyTask(LitNNP):
    def __init__(self, interval="auto", plateau=True, missing_metric=False, multiple_loaders=False):
        scheduler = (partial(torch.optim.lr_scheduler.ReduceLROnPlateau, patience=0, factor=0.5)
                     if plateau else partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.5))
        super().__init__(NeuralNetworkPotential(nn.Linear(1, 1)), [],
                         partial(torch.optim.SGD, lr=0.1), scheduler=scheduler,
                         scheduler_monitor="val_loss", scheduler_interval=interval,
                         save_entire_model=False)
        self.save_configuration(OmegaConf.create({"data": {}, "model": {}}))
        self.validation_steps = []
        self.missing_metric = missing_metric
        self.multiple_loaders = multiple_loaders

    def setup(self, stage=None):
        pass

    def training_step(self, batch, batch_idx):
        return self.model.representation(batch).square().mean()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # A constant held-out metric makes the expected LR trajectory exact.
        self.log("other" if self.missing_metric else "val_loss", 1.0,
                 on_step=False, on_epoch=True, add_dataloader_idx=self.multiple_loaders)

    def on_validation_epoch_end(self):
        if self.multiple_loaders:
            self.log("val_loss", 1.0, on_step=False, on_epoch=True)
        if not self.trainer.sanity_checking:
            self.validation_steps.append(self.global_step)


def loader(batches=1):
    return DataLoader(torch.ones(batches, 1), batch_size=1)


def trainer(path, steps=6, callbacks=None, **kwargs):
    return Trainer(default_root_dir=path, accelerator="cpu", devices=1, max_epochs=-1,
                   max_steps=steps, val_check_interval=2, check_val_every_n_epoch=None,
                   num_sanity_val_steps=2, logger=False, enable_progress_bar=False,
                   enable_model_summary=False, enable_checkpointing=bool(callbacks),
                   callbacks=callbacks or [], **kwargs)


def validation_scheduler(fit):
    return next(cb for cb in fit.callbacks if isinstance(cb, ValidationScheduler))


@pytest.mark.parametrize("batches", [1, 3])
def test_validation_clock_crosses_epochs_without_using_stale_metrics(tmp_path, batches):
    task = TinyTask()
    fit = trainer(tmp_path)
    fit.fit(task, loader(batches), loader())
    callback = validation_scheduler(fit)
    assert task.validation_steps == [2, 4, 6]
    assert callback.scheduler.last_epoch == 3
    assert callback.last_validation_step == 6
    assert fit.optimizers[0].param_groups[0]["lr"] == pytest.approx(0.025)
    assert fit.lr_scheduler_configs == []  # Only the validation callback owns stepping.
    fit.validate(task, loader(), verbose=False)
    assert callback.scheduler.last_epoch == 3


def test_scheduler_state_is_saved_before_checkpoint_and_restored(tmp_path):
    save = ModelCheckpoint(dirpath=tmp_path / "first", save_last=True, save_top_k=-1,
                           save_on_train_epoch_end=False)
    first = trainer(tmp_path / "first", steps=4, callbacks=[save])
    first.fit(TinyTask(), loader(), loader())
    checkpoint = torch.load(save.last_model_path, weights_only=False, map_location="cpu")
    callback = validation_scheduler(first)
    state = checkpoint["callbacks"][callback.state_key]
    assert state["last_validation_step"] == 4
    assert state["scheduler"]["last_epoch"] == 2
    assert checkpoint["optimizer_states"][0]["param_groups"][0]["lr"] == pytest.approx(0.05)

    resumed_task = TinyTask()
    resumed = trainer(tmp_path / "resumed")
    resumed.fit(resumed_task, loader(), loader(), ckpt_path=save.last_model_path)
    assert validation_scheduler(resumed).scheduler.last_epoch == 3
    assert resumed.optimizers[0].param_groups[0]["lr"] == pytest.approx(0.025)


def test_standard_early_stopping_counts_the_same_validations(tmp_path):
    stop = EarlyStopping("val_loss", patience=2, check_on_train_epoch_end=False)
    task = TinyTask()
    fit = trainer(tmp_path, steps=20, callbacks=[stop])
    fit.fit(task, loader(), loader())
    assert fit.global_step == 6
    assert task.validation_steps == [2, 4, 6]
    assert stop.wait_count == 2
    assert validation_scheduler(fit).scheduler.last_epoch == 3


def test_multiple_validation_loaders_advance_only_after_aggregation(tmp_path):
    task = TinyTask(multiple_loaders=True)
    fit = trainer(tmp_path)
    fit.fit(task, loader(), [loader(), loader(2)])
    assert validation_scheduler(fit).scheduler.last_epoch == 3


def test_step_scheduler_follows_optimizer_updates_with_accumulation(tmp_path):
    fit = trainer(tmp_path, steps=3, accumulate_grad_batches=2)
    fit.fit(TinyTask(interval="step", plateau=False), loader(2), loader())
    assert fit.lr_scheduler_configs[0].scheduler.last_epoch == 3
    assert fit.optimizers[0].param_groups[0]["lr"] == pytest.approx(0.0125)
    assert validation_scheduler(fit).scheduler is None


def test_plateau_counts_validations_between_accumulated_optimizer_updates(tmp_path):
    stop = EarlyStopping("val_loss", patience=10, check_on_train_epoch_end=False)
    task = TinyTask()
    fit = trainer(tmp_path, steps=2, callbacks=[stop], accumulate_grad_batches=4)
    fit.fit(task, loader(8), loader())
    assert task.validation_steps == [0, 1, 1, 2]
    assert validation_scheduler(fit).scheduler.last_epoch == 4
    assert stop.wait_count == 3


def test_missing_validation_metric_fails_explicitly(tmp_path):
    fit = trainer(tmp_path)
    with pytest.raises(RuntimeError, match="Validation scheduler metric 'val_loss' is missing"):
        fit.fit(TinyTask(missing_metric=True), loader(), loader())


def test_legacy_lightning_scheduler_state_migrates():
    task = TinyTask()
    task.configure_optimizers()
    old = task._validation_scheduler
    old.step(1.0)
    old.step(1.0)
    saved = old.state_dict()
    callback = ValidationScheduler("val_loss")
    callback.on_load_checkpoint(None, task, {"lr_schedulers": [saved]})
    task.configure_optimizers()
    callback.on_fit_start(None, task)
    assert callback.scheduler.last_epoch == 2
    assert callback.scheduler.best == 1.0


def test_state_can_be_restored_after_on_fit_start():
    task = TinyTask()
    task.configure_optimizers()
    callback = ValidationScheduler("val_loss")
    callback.on_fit_start(None, task)
    saved = dict(callback.scheduler.state_dict(), last_epoch=7, best=0.25)
    callback.load_state_dict({"scheduler": saved, "last_validation_step": 14})
    assert callback.scheduler.last_epoch == 7
    assert callback.scheduler.best == 0.25
    assert callback._pending_state is None


def test_validation_interval_rejects_non_metric_scheduler():
    with pytest.raises(ValueError, match="validation scheduling requires"):
        TinyTask(interval="validation", plateau=False).configure_optimizers()
