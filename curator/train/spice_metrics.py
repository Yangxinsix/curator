"""Exact validation monitor for the unscaled SPICE energy/force/dipole MSE."""

import math

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import MeanSquaredError

from .model_output import ModelOutput


class SpiceValidationLoss(pl.Callback):
    """Sum physical MSE terms from their complete-epoch RMSE accumulators.

    Lightning runs callback epoch-end hooks before the module epoch-end hook,
    so the metric states are read before LitNNP clears them. Each observable
    retains its own scalar-observation denominator, including the final batch.
    This deliberately requires unsampled, unscaled, fully labeled data with
    unit geometry weights; other loss configurations need their own monitor.
    """

    @staticmethod
    def _outputs(trainer, pl_module):
        if len(trainer.num_val_batches) != 1:
            raise ValueError("SpiceValidationLoss requires exactly one validation dataloader")
        if getattr(pl_module, "rescale_layers", ()) or getattr(pl_module, "normalize_domain_loss", False):
            raise ValueError("SpiceValidationLoss requires physical losses without rescaling or domain normalization")
        outputs = list(pl_module.outputs)
        names = [output.name for output in outputs]
        if not names or len(names) != len(set(names)):
            raise ValueError("SpiceValidationLoss requires nonempty outputs with unique names")
        for output in outputs:
            if (
                type(output) is not ModelOutput
                or output.name not in {"energy", "forces", "dipole"}
                or output.prediction_property != output.name
                or output.target_property != output.name
                or type(output.loss_fn) is not nn.MSELoss
                or output.loss_fn.reduction != "mean"
                or output.is_penalty or output.per_atom_loss or output.only_train
                or output.sampler.enabled()
                or not math.isfinite(output.loss_weight) or output.loss_weight < 0
            ):
                raise ValueError("SpiceValidationLoss supports only ordinary energy/forces/dipole mean MSE outputs")
            metrics = getattr(output, "val_metrics", {})
            metric = metrics["rmse"] if "rmse" in metrics else None
            if (
                type(metric) is not MeanSquaredError or metric.squared
                or metric.num_outputs != 1 or not metric.sync_on_compute
            ):
                raise ValueError("SpiceValidationLoss requires val_metrics['rmse'] = MeanSquaredError(squared=False, num_outputs=1)")
        return outputs

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        for output in self._outputs(trainer, pl_module):
            # Tensor inputs alone do not promote TorchMetrics' accumulator.
            # Keep its device and explicitly use double precision from the
            # first batch, rather than converting a rounded result afterwards.
            metric = output.val_metrics["rmse"]
            if metric.update_count:
                raise ValueError("SPICE validation metrics were not reset before the epoch")
            metric.set_dtype(torch.float64)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking:
            return
        weight = torch.as_tensor(batch.get("weight", 1.0))
        if not torch.all(weight == 1):
            raise ValueError("SpiceValidationLoss requires unit geometry weights")
        for output in pl_module.outputs:
            mask = batch.get("masks", {}).get(output.target_property)
            if mask is not None and not torch.as_tensor(mask).all():
                raise ValueError("SpiceValidationLoss requires complete energy/force/dipole labels")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        total = torch.zeros((), dtype=torch.float64, device=pl_module.device)
        for output in self._outputs(trainer, pl_module):
            metric = output.val_metrics["rmse"]
            if not metric.update_count:
                raise ValueError("SPICE validation RMSE was not accumulated before the callback")
            rmse = metric.compute().detach().to(device=total.device, dtype=total.dtype).reshape(())
            if not torch.isfinite(rmse):
                raise ValueError("SPICE validation RMSE is nonfinite")
            total = total + output.loss_weight * rmse.square()
        # TorchMetrics.compute() has already synchronized numerator/count.
        pl_module.log("val_spice_loss_exact", total, on_step=False, on_epoch=True,
                      sync_dist=False, add_dataloader_idx=False)


__all__ = ["SpiceValidationLoss"]
