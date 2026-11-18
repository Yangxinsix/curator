"""Real CPU Lightning regressions for scheduling, stopping, and full resume.

Two training batches per epoch make callback ordering failures reproducible without
ACVR data or GPUs. These assert behavior across Lightning hooks, not private helpers.
"""
from pathlib import Path

import pytest
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from curator.train.convergence import LearningRateGatedEarlyStopping
from curator.train.resume_callbacks import ResumeCheckpoint, ResumeTrainingState


class TinyScheduledModel(LightningModule):
    def __init__(self, *, scheduler=True, scores=(1.0,), log_score=True,
                 physical_values=(1.0,), log_physical=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.with_scheduler = scheduler
        self.scores = scores
        self.log_score = log_score
        self.physical_values = physical_values
        self.log_physical = log_physical

    def training_step(self, batch, batch_idx):
        return self.weight.square()

    def validation_step(self, batch, batch_idx):
        if self.log_score:
            value = self.scores[min(self.current_epoch, len(self.scores) - 1)]
            self.log('val_loss', torch.tensor(value, dtype=torch.float64), on_epoch=True)
        if self.log_physical:
            value = self.physical_values[min(self.current_epoch, len(self.physical_values) - 1)]
            self.log('val_energy', torch.tensor(value, dtype=torch.float64), on_epoch=True)

    def test_step(self, batch, batch_idx):
        self.log('test_loss', self.weight.square())

    def predict_step(self, batch, batch_idx):
        return self.weight.detach().clone()

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1, amsgrad=True)
        if not self.with_scheduler:
            return optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=0, cooldown=0,
            min_lr=0.025, threshold=0.005, threshold_mode='rel')
        return {'optimizer': optimizer, 'lr_scheduler': {
            'scheduler': scheduler, 'monitor': 'val_loss', 'interval': 'epoch'}}

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.ones(2)), batch_size=1, num_workers=0)

    def val_dataloader(self):
        return DataLoader(TensorDataset(torch.ones(1)), batch_size=1, num_workers=0)


class EpochTrace(Callback):
    def __init__(self):
        self.rows = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.rows.append((int(trainer.current_epoch), int(trainer.global_step),
                          trainer.optimizers[0].param_groups[0]['lr']))


class StopFromAnotherCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            trainer.should_stop = True


class CrashAtEpochStart(Callback):
    def __init__(self, epoch):
        self.epoch = epoch

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.epoch:
            raise RuntimeError('intentional crash after resume checkpoint')


def trainer(tmp_path, callbacks=(), epochs=6, max_steps=-1):
    return Trainer(default_root_dir=tmp_path, accelerator='cpu', devices=1,
                   max_epochs=epochs, max_steps=max_steps, callbacks=list(callbacks),
                   logger=False, enable_checkpointing=any(isinstance(c, ModelCheckpoint) for c in callbacks),
                   enable_progress_bar=False, enable_model_summary=False,
                   num_sanity_val_steps=0, deterministic=True)


def stopper(**kwargs):
    defaults = dict(monitor='val_loss', physical_metrics=['val_energy'],
                    min_lr=0.025, min_lr_checks=2, plateau_window=1,
                    patience=1, check_on_train_epoch_end=False)
    defaults.update(kwargs)
    return LearningRateGatedEarlyStopping(**defaults)


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_native_scheduler_controls_lr_and_gate_waits_for_floor(tmp_path):
    trace = EpochTrace()
    early = stopper()
    fit = trainer(tmp_path, [trace, early], epochs=15)
    fit.fit(TinyScheduledModel())
    assert trace.rows[:4] == [(0, 0, 0.1), (1, 2, 0.1), (2, 4, 0.05), (3, 6, 0.025)]
    assert len(trace.rows) >= 5, 'Must observe at least two validations at the LR floor'
    assert len(trace.rows) < 15, 'A stationary finite objective must eventually early-stop'
    assert fit.should_stop
    assert fit.optimizers[0].param_groups[0]['lr'] == 0.025
    assert isinstance(fit.lr_scheduler_configs[0].scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_improving_physical_metric_prevents_plateau_stop(tmp_path):
    fit = trainer(tmp_path, [stopper()], epochs=9)
    fit.fit(TinyScheduledModel(physical_values=tuple(0.8 ** i for i in range(9))))
    assert fit.global_step == 18
    assert not fit.should_stop


def test_zero_metrics_can_converge(tmp_path):
    fit = trainer(tmp_path, [stopper()], epochs=15)
    fit.fit(TinyScheduledModel(scores=(0.0,), physical_values=(0.0,)))
    assert fit.should_stop
    assert fit.global_step < 30


def test_closed_gate_never_cancels_another_callbacks_stop(tmp_path):
    fit = trainer(tmp_path, [StopFromAnotherCallback(), stopper()], epochs=10)
    fit.fit(TinyScheduledModel())
    assert fit.global_step == 2
    assert fit.should_stop
    assert fit.optimizers[0].param_groups[0]['lr'] == 0.1


def test_closed_gate_does_not_hide_missing_objective(tmp_path):
    fit = trainer(tmp_path, [stopper(strict=True)], epochs=2)
    with pytest.raises((RuntimeError, ValueError), match='val_loss'):
        fit.fit(TinyScheduledModel(log_score=False))


def test_closed_gate_does_not_hide_missing_physical_metric(tmp_path):
    fit = trainer(tmp_path, [stopper(strict=True)], epochs=2)
    with pytest.raises((RuntimeError, ValueError), match='val_energy'):
        fit.fit(TinyScheduledModel(log_physical=False))


def test_closed_gate_does_not_mask_nonfinite_objective(tmp_path):
    fit = trainer(tmp_path, [stopper(check_finite=True)], epochs=10)
    fit.fit(TinyScheduledModel(scores=(float('nan'),)))
    assert fit.should_stop
    assert fit.global_step == 2


def test_closed_gate_does_not_mask_nonfinite_physical_metric(tmp_path):
    fit = trainer(tmp_path, [stopper(check_finite=True)], epochs=10)
    with pytest.raises(ValueError, match='val_energy.*finite'):
        fit.fit(TinyScheduledModel(physical_values=(float('nan'),)))
    assert fit.global_step == 2


def run_full(tmp_path, epochs=6):
    model = TinyScheduledModel()
    trace = EpochTrace()
    fit = trainer(tmp_path, [trace], epochs=epochs, max_steps=2 * epochs)
    fit.fit(model)
    return fit, model, trace


def test_post_scheduler_final_checkpoint_resumes_exact_optimizer_and_lr(tmp_path):
    full, full_model, full_trace = run_full(tmp_path / 'full')
    first_dir = tmp_path / 'partial'
    first_model = TinyScheduledModel()
    first = trainer(first_dir, [ResumeCheckpoint(dirpath=first_dir)], epochs=3, max_steps=6)
    first.fit(first_model)
    path = first_dir / 'last.ckpt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['curator_checkpoint_phase'] == 'resume_after_scheduler'
    assert saved['optimizer_states'][0]['param_groups'][0]['lr'] == 0.025
    assert saved['lr_schedulers'][0]['last_epoch'] == 3
    restored_model, restored_trace = TinyScheduledModel(), EpochTrace()
    restored = trainer(tmp_path / 'restored', [ResumeTrainingState(), restored_trace], epochs=6, max_steps=12)
    restored.fit(restored_model, ckpt_path=path)
    assert restored_trace.rows == full_trace.rows[3:]
    assert_tree_equal(restored_model.state_dict(), full_model.state_dict())
    assert_tree_equal(restored.optimizers[0].state_dict(), full.optimizers[0].state_dict())
    assert_tree_equal(restored.lr_scheduler_configs[0].scheduler.state_dict(),
                      full.lr_scheduler_configs[0].scheduler.state_dict())


def test_epoch_start_checkpoint_resumes_without_replaying_or_skipping_epoch(tmp_path):
    full, full_model, full_trace = run_full(tmp_path / 'full')
    crashed_dir = tmp_path / 'crashed'
    fit = trainer(crashed_dir, [ResumeCheckpoint(dirpath=crashed_dir), CrashAtEpochStart(3)],
                  epochs=6, max_steps=12)
    with pytest.raises(RuntimeError, match='intentional crash'):
        fit.fit(TinyScheduledModel())
    path = crashed_dir / 'last.ckpt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['global_step'] == 6
    assert saved['curator_checkpoint_phase'] == 'resume_after_scheduler'
    restored_model, restored_trace = TinyScheduledModel(), EpochTrace()
    restored = trainer(tmp_path / 'restored', [ResumeTrainingState(), restored_trace], epochs=6, max_steps=12)
    restored.fit(restored_model, ckpt_path=path)
    assert restored_trace.rows == full_trace.rows[3:]
    assert_tree_equal(restored_model.state_dict(), full_model.state_dict())
    assert_tree_equal(restored.optimizers[0].state_dict(), full.optimizers[0].state_dict())
    assert_tree_equal(restored.lr_scheduler_configs[0].scheduler.state_dict(),
                      full.lr_scheduler_configs[0].scheduler.state_dict())


def test_legacy_checkpoint_without_scheduler_keeps_adam_and_initializes_scheduler(tmp_path):
    old_dir = tmp_path / 'old'
    old = trainer(old_dir, [ResumeCheckpoint(dirpath=old_dir)], epochs=2, max_steps=4)
    old.fit(TinyScheduledModel(scheduler=False))
    path = old_dir / 'last.ckpt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    # Simulate the actual legacy serialization protocol, which had no phase tag.
    saved.pop('curator_checkpoint_phase')
    saved['callbacks']['ACVRConvergenceController.v1'] = {'bad_checks': 5, 'reductions': 2}
    torch.save(saved, path)
    audit = tmp_path / 'resume.json'
    trace = EpochTrace()
    resumed = trainer(tmp_path / 'new', [ResumeTrainingState(output_path=audit), trace], epochs=3, max_steps=6)
    resumed.fit(TinyScheduledModel(), ckpt_path=path)
    assert trace.rows == [(2, 4, 0.1)]
    assert resumed.global_step == 6
    assert resumed.lr_scheduler_configs[0].scheduler.last_epoch == 1
    assert int(next(iter(resumed.optimizers[0].state.values()))['step']) == 6
    assert audit.is_file()


def test_pre_scheduler_best_checkpoint_cannot_silently_resume(tmp_path):
    best_dir = tmp_path / 'best'
    best = ModelCheckpoint(dirpath=best_dir, monitor='val_loss', save_top_k=-1, save_last=False)
    fit = trainer(best_dir, [ResumeCheckpoint(dirpath=tmp_path / 'resume'), best], epochs=2)
    fit.fit(TinyScheduledModel())
    candidates = list(best_dir.glob('*.ckpt'))
    assert candidates
    path = max(candidates, key=lambda p: torch.load(p, weights_only=False)['global_step'])
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['curator_checkpoint_phase'] == 'evaluation_before_scheduler'
    resumed = trainer(tmp_path / 'reject', [ResumeTrainingState()], epochs=3)
    with pytest.raises((RuntimeError, ValueError), match='(?i)(evaluation|scheduler|resume)'):
        resumed.fit(TinyScheduledModel(), ckpt_path=path)


def test_new_callback_and_scheduler_state_roundtrip_matches_uninterrupted_stop(tmp_path):
    full_gate = stopper(min_lr_checks=4)
    full_model, full_trace = TinyScheduledModel(), EpochTrace()
    full = trainer(tmp_path / 'full', [full_trace, full_gate], epochs=15, max_steps=30)
    full.fit(full_model)
    partial_dir = tmp_path / 'partial'
    partial_gate = stopper(min_lr_checks=4)
    partial = trainer(partial_dir, [partial_gate, ResumeCheckpoint(dirpath=partial_dir)], epochs=4, max_steps=8)
    partial.fit(TinyScheduledModel())
    path = partial_dir / 'last.ckpt'
    restored_gate, restored_trace, restored_model = stopper(min_lr_checks=4), EpochTrace(), TinyScheduledModel()
    restored = trainer(tmp_path / 'restored', [ResumeTrainingState(), restored_trace, restored_gate], epochs=15, max_steps=30)
    restored.fit(restored_model, ckpt_path=path)
    assert restored.global_step == full.global_step
    assert restored_trace.rows == full_trace.rows[4:]
    assert_tree_equal(restored_gate.state_dict(), full_gate.state_dict())
    assert_tree_equal(restored_model.state_dict(), full_model.state_dict())


def test_native_legacy_epoch_end_checkpoint_resumes_at_next_real_epoch(tmp_path):
    old_dir = tmp_path / 'native_old'
    checkpoint = ModelCheckpoint(dirpath=old_dir, save_last=True, save_top_k=-1)
    old = trainer(old_dir, [checkpoint], epochs=2, max_steps=4)
    old.fit(TinyScheduledModel(scheduler=False))
    saved = torch.load(checkpoint.last_model_path, weights_only=False)
    assert saved['global_step'] == 4
    assert not saved['lr_schedulers']
    assert 'curator_checkpoint_phase' not in saved
    trace = EpochTrace()
    recovery = ResumeTrainingState()
    resumed = trainer(tmp_path / 'resumed', [recovery, trace], epochs=3, max_steps=6)
    resumed.fit(TinyScheduledModel(), ckpt_path=checkpoint.last_model_path)
    assert trace.rows == [(2, 4, 0.1)]
    assert resumed.global_step == 6
    assert resumed.lr_scheduler_configs[0].scheduler.last_epoch == 1
    assert recovery.report['scheduler_state'] == 'initialized_for_legacy_checkpoint'


@pytest.mark.parametrize('stage', ['validate', 'test', 'predict'])
def test_evaluation_checkpoint_is_loadable_for_nontraining_stages(tmp_path, stage):
    best_dir = tmp_path / 'best'
    best = ModelCheckpoint(dirpath=best_dir, monitor='val_loss', save_top_k=1)
    fit = trainer(best_dir, [ResumeCheckpoint(dirpath=tmp_path / 'resume'), best], epochs=1)
    fit.fit(TinyScheduledModel())
    saved = torch.load(best.best_model_path, weights_only=False)
    assert saved['curator_checkpoint_phase'] == 'evaluation_before_scheduler'
    recovery = ResumeTrainingState()
    evaluation = trainer(tmp_path / stage, [recovery])
    result = getattr(evaluation, stage)(TinyScheduledModel(), ckpt_path=best.best_model_path)
    assert result
    assert recovery.report is None, 'Evaluation does not restore training optimizer/scheduler state'


class BatchTrace(Callback):
    def __init__(self):
        self.rows = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.rows.append((int(trainer.current_epoch), int(trainer.global_step), batch_idx))


def test_mid_epoch_resource_stop_resumes_with_real_updates(tmp_path):
    partial_dir = tmp_path / 'partial'
    partial = trainer(partial_dir, [ResumeCheckpoint(dirpath=partial_dir)], epochs=8, max_steps=3)
    partial.fit(TinyScheduledModel())
    path = partial_dir / 'last.ckpt'
    saved = torch.load(path, weights_only=False)
    assert saved['global_step'] == 3
    progress = saved['loops']['fit_loop']['epoch_loop.batch_progress']
    assert not progress['is_last_batch'], 'Exercise a real partial-epoch interruption'
    epochs, batches = EpochTrace(), BatchTrace()
    resumed = trainer(tmp_path / 'resumed', [ResumeTrainingState(), epochs, batches], epochs=8, max_steps=8)
    resumed.fit(TinyScheduledModel(), ckpt_path=path)
    assert batches.rows[0][1] == 4
    assert [row[1] for row in batches.rows] == [4, 5, 6, 7, 8]
    assert resumed.global_step == 8
    assert int(next(iter(resumed.optimizers[0].state.values()))['step']) == 8
    assert all(after[1] > before[1] for before, after in zip(epochs.rows, epochs.rows[1:])), (
        'A resource-stop resume must not consume an empty training epoch')
    # Ordinary torch DataLoaders do not serialize sampler/iterator position.
    # This verifies optimizer continuation, not exact mid-epoch data replay.
