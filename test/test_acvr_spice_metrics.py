"""Exact SPICE monitoring with uneven batches and Lightning callback ordering."""

from functools import partial
from types import SimpleNamespace

import pytorch_lightning as pl
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError

from curator.model import LitNNP, NeuralNetworkPotential
from curator.train import ModelOutput
from curator.train.spice_metrics import SpiceValidationLoss


class _Predictions(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def forward(self, batch):
        feature = batch["feature"] + self.offset
        return {"energy": feature,
                "forces": 3 * feature[:, None, None].expand(-1, 2, 3),
                "dipole": 2 * feature[:, None].expand(-1, 3)}


class _ObservedTask(LitNNP):
    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            self.monitor_before_module_reset = self.trainer.callback_metrics["val_spice_loss_exact"].clone()
            self.counts_before_module_reset = [output.val_metrics["rmse"].update_count for output in self.outputs]
        super().on_validation_epoch_end()


def _outputs():
    return [ModelOutput(name, loss_fn=nn.MSELoss(), loss_weight=weight,
                        metrics={"rmse": MeanSquaredError(squared=False)})
            for name, weight in (("energy", 2.0), ("forces", 3.0), ("dipole", 0.5))]


@pytest.mark.parametrize("accelerator", ["cpu", "gpu"])
def test_exact_monitor_precedes_metric_reset_and_drives_checkpoint_with_short_final_batch(tmp_path, accelerator):
    if accelerator == "gpu" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    samples = [{"feature": torch.tensor(value, dtype=torch.float64),
                "energy": torch.tensor(0.0, dtype=torch.float64),
                "forces": torch.zeros(2, 3, dtype=torch.float64),
                "dipole": torch.zeros(3, dtype=torch.float64),
                "n_atoms": torch.tensor(2), "weight": torch.tensor(1.0)}
               for value in (1, 2, 3, 4, 10)]
    loader = DataLoader(samples, batch_size=2)
    task = _ObservedTask(NeuralNetworkPotential(_Predictions(), model_outputs=["energy", "forces", "dipole"]),
                         _outputs(), partial(torch.optim.SGD, lr=0.0))
    task.save_configuration(OmegaConf.create({"data": {}, "model": {}}))
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=tmp_path / accelerator, monitor="val_spice_loss_exact")
    trainer = pl.Trainer(accelerator=accelerator, devices=1, max_epochs=1, precision="64-true",
                         callbacks=[SpiceValidationLoss(), checkpoint], logger=False,
                         enable_progress_bar=False, enable_model_summary=False,
                         num_sanity_val_steps=2, inference_mode=False)
    trainer.fit(task, train_dataloaders=loader, val_dataloaders=loader)
    feature = torch.tensor([1, 2, 3, 4, 10], dtype=torch.float64)
    expected = (2 + 3 * 3**2 + 0.5 * 2**2) * feature.square().mean()
    assert expected.item() == 806.0
    actual = trainer.callback_metrics["val_spice_loss_exact"]
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-14, atol=1e-12)
    torch.testing.assert_close(task.monitor_before_module_reset.cpu(), expected, rtol=1e-14, atol=1e-12)
    assert task.counts_before_module_reset == [3, 3, 3]
    assert all(output.val_metrics["rmse"].update_count == 0 for output in task.outputs)
    assert checkpoint.best_model_path
    assert checkpoint.best_model_score.item() == pytest.approx(expected.item())
    # The old monitor still demonstrates the unequal-batch bias this callback
    # avoids. It remains available for backwards compatibility.
    assert trainer.callback_metrics["val_total_loss_epoch"].item() == pytest.approx(31 * (2.5 + 12.5 + 100) / 3)


@pytest.mark.parametrize("mismatch", ["l1", "squared_metric", "per_atom", "sampling", "rescale", "multi_loader"])
def test_unsupported_loss_configuration_fails_explicitly(mismatch):
    task = SimpleNamespace(outputs=_outputs(), rescale_layers=[], normalize_domain_loss=False)
    trainer = SimpleNamespace(sanity_checking=False, num_val_batches=[3])
    if mismatch == "l1":
        task.outputs[0].loss_fn = nn.L1Loss()
    elif mismatch == "squared_metric":
        task.outputs[0].val_metrics["rmse"] = MeanSquaredError(squared=True)
    elif mismatch == "per_atom":
        task.outputs[0].per_atom_loss = True
    elif mismatch == "sampling":
        task.outputs[0].sampler.num_samples = 1
    elif mismatch == "rescale":
        task.rescale_layers = [nn.Identity()]
    else:
        trainer.num_val_batches = [3, 3]
    with pytest.raises(ValueError, match="SpiceValidationLoss"):
        SpiceValidationLoss().on_validation_epoch_start(trainer, task)


def test_nonunit_weights_and_missing_labels_cannot_change_the_monitored_objective():
    callback = SpiceValidationLoss()
    trainer = SimpleNamespace(sanity_checking=False)
    task = SimpleNamespace(outputs=_outputs())
    with pytest.raises(ValueError, match="unit geometry weights"):
        callback.on_validation_batch_start(trainer, task, {"weight": torch.tensor([1.0, 2.0])}, 0)
    with pytest.raises(ValueError, match="complete"):
        callback.on_validation_batch_start(trainer, task, {"masks": {"energy": torch.tensor([True, False])}}, 0)
