"""Full-training-state recovery and dedicated resumable checkpoint storage."""
import hashlib
import json
from pathlib import Path

import torch
from pytorch_lightning import Callback
from pytorch_lightning.trainer.states import TrainerFn


def _optimizer_digest(states):
    digest = hashlib.sha256()
    for optimizer in states:
        for index, state in sorted(optimizer['state'].items()):
            digest.update(str(index).encode())
            for name, value in sorted(state.items()):
                digest.update(name.encode())
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().contiguous()
                    digest.update(str((value.dtype, tuple(value.shape))).encode())
                    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
                else:
                    digest.update(repr(value).encode())
    return digest.hexdigest()


class ResumeTrainingState(Callback):
    """Check complete restoration, including legacy epoch-end loop compatibility.

    A legacy checkpoint with no scheduler restores model/Adam/LR/step but starts
    new scheduler and stopping histories. It is never called an exact migration
    of the former ad-hoc convergence controller.
    """

    def __init__(self, output_path=None):
        self.output_path = None if output_path is None else Path(output_path)
        self._expected = None
        self.report = None

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if trainer.state.fn != TrainerFn.FITTING:
            return
        schedulers = checkpoint.get('lr_schedulers', [])
        phase = checkpoint.get('curator_checkpoint_phase')
        if schedulers and phase == 'evaluation_before_scheduler':
            raise ValueError('Use resume/last.ckpt for full continuation; this best checkpoint precedes scheduler.step')
        optimizers = checkpoint['optimizer_states']
        self._expected = {
            'global_step': int(checkpoint['global_step']), 'epoch': int(checkpoint['epoch']),
            'optimizer_digest': _optimizer_digest(optimizers),
            'param_groups': [o['param_groups'] for o in optimizers],
            'schedulers': schedulers, 'phase': phase,
        }

    def on_train_start(self, trainer, pl_module):
        if self._expected is None:
            return
        expected = self._expected
        # Epoch-boundary checkpoints can retain completed batch counters. Clear
        # restart flags so resume does not replay an empty epoch (Lightning 2.5/2.6).
        loop = trainer.fit_loop
        if (loop.epoch_loop.batch_progress.is_last_batch
                and (loop.restarted_on_epoch_end
                     or loop.epoch_progress.current.processed == loop.epoch_progress.current.completed)):
            from pytorch_lightning.loops.loop import _Loop
            incomplete_epoch_end = loop.restarted_on_epoch_end
            _Loop.restarting.fset(loop, False)
            if incomplete_epoch_end:
                loop.epoch_progress.increment_completed()
            loop.update_restart_stage()
        actual = [o.state_dict() for o in trainer.optimizers]
        if (trainer.global_step != expected['global_step']
                or _optimizer_digest(actual) != expected['optimizer_digest']
                or [o['param_groups'] for o in actual] != expected['param_groups']):
            raise RuntimeError('Resume mismatch in global step, optimizer moments or parameter groups')
        schedulers = [c.scheduler.state_dict() for c in trainer.lr_scheduler_configs]
        if expected['schedulers'] and schedulers != expected['schedulers']:
            raise RuntimeError('Resume mismatch in learning-rate scheduler state')
        self.report = {'checkpoint': str(trainer.ckpt_path), 'global_step': trainer.global_step,
                       'optimizer_state_matches': True, 'optimizer_state_sha256': expected['optimizer_digest'],
                       'scheduler_state': 'restored' if expected['schedulers'] else ('initialized_for_legacy_checkpoint' if schedulers else 'absent'),
                       'checkpoint_phase': expected['phase'], 'bitwise_rng_replay_claimed': False}
        if self.output_path is not None and trainer.is_global_zero:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(json.dumps(self.report, indent=2) + '\n')
        self._expected = None


class ResumeCheckpoint(Callback):
    """Save post-scheduler training state; ModelCheckpoint still selects best.

    Lightning steps epoch-based ReduceLROnPlateau after epoch-end callbacks.
    Save at the next epoch start and at fit end to include that update. A best
    checkpoint remains valid for evaluation but is explicitly tagged as such.
    """

    def __init__(self, dirpath, filename='last.ckpt'):
        self.path = Path(dirpath) / filename
        self._saving_resume = False

    def _save(self, trainer):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._saving_resume = True
        try:
            trainer.save_checkpoint(self.path)
        finally:
            self._saving_resume = False

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.global_step > 0:
            self._save(trainer)

    def on_fit_end(self, trainer, pl_module):
        self._save(trainer)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['curator_checkpoint_phase'] = (
            'resume_after_scheduler' if self._saving_resume else 'evaluation_before_scheduler')
