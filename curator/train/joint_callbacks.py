"""Validation and checkpoint policies for the joint variational configuration."""

import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint

from .structured_output import StructuredModelOutput


class JointValidationLoss(Callback):
    """Monitor the dataset-aggregated loss at the predicted internal state."""

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        owners = [module for module in pl_module.model.modules()
                  if hasattr(module, "supports_state_modes")]
        if not owners or any(module.state_mode != "predicted" for module in owners):
            raise ValueError("joint validation monitor requires predicted internal states")
        if len(trainer.num_val_batches) != 1:
            raise ValueError("joint validation monitor requires one validation dataloader")
        if not pl_module.outputs or not all(
            isinstance(output, StructuredModelOutput) for output in pl_module.outputs
        ):
            raise ValueError("joint validation monitor requires structured loss outputs")
        # Callback epoch-end hooks precede LitNNP's accumulator reset. Each
        # output already synchronizes its separate numerators and denominators.
        value = sum(output.accumulate_loss("val") for output in pl_module.outputs)
        value = torch.tensor(value, dtype=torch.float64, device=pl_module.device)
        if not torch.isfinite(value):
            raise ValueError("joint validation loss is nonfinite")
        pl_module.log("val_joint_predicted_loss", value, on_step=False, on_epoch=True,
                      sync_dist=False, add_dataloader_idx=False)


class JointModelCheckpoint(ModelCheckpoint):
    """Keep ``last`` current even when this epoch does not improve the best loss.

    Lightning 2.6.1 refreshes last at epoch/validation end only when top-k saved a
    checkpoint. Restrict this policy to the new joint config; preserve top-k
    selection and the standard checkpoint serialization/resume implementation.
    """

    def on_train_start(self, trainer, pl_module):
        loop = trainer.fit_loop
        if loop.restarted_on_epoch_end and loop.epoch_loop.batch_progress.is_last_batch:
            # Lightning 2.6 forces FitLoop.restarting=True whenever max_steps is
            # finite, even for an epoch-end checkpoint. Replaying that completed
            # epoch advances its counters without training and consumes shuffle
            # state. Resume at the next epoch exactly as epoch-only runs do.
            # Bypass that FitLoop setter while retaining its recursive semantics.
            from pytorch_lightning.loops.loop import _Loop

            _Loop.restarting.fset(loop, False)
            loop.epoch_progress.increment_completed()
            loop.update_restart_stage()

    def on_validation_end(self, trainer, pl_module):
        due = (
            not self._should_skip_saving_checkpoint(trainer)
            and not self._should_save_on_train_epoch_end(trainer)
            and self._every_n_epochs >= 1
            and (trainer.current_epoch + 1) % self._every_n_epochs == 0
        )
        super().on_validation_end(trainer, pl_module)
        if due and self.save_last and self._last_global_step_saved != trainer.global_step:
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))

    def on_train_epoch_end(self, trainer, pl_module):
        due = (
            not self._should_skip_saving_checkpoint(trainer)
            and self._should_save_on_train_epoch_end(trainer)
            and self._every_n_epochs >= 1
            and (trainer.current_epoch + 1) % self._every_n_epochs == 0
        )
        super().on_train_epoch_end(trainer, pl_module)
        if due and self.save_last and self._last_global_step_saved != trainer.global_step:
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))
