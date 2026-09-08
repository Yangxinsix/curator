from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn
from torchmetrics import MeanAbsoluteError, MeanSquaredError

from curator.model.lit_module import LitNNP
from curator.train.metrics import PerAtomMAE
from curator.train.model_output import ModelOutput


def test_per_atom_energy_loss_matches_mace_normalization():
    output = ModelOutput(
        name="energy",
        loss_fn=torch.nn.MSELoss(),
        per_atom_loss=True,
    )
    pred = {"energy": torch.tensor([6.0, 8.0])}
    target = {
        "energy": torch.tensor([2.0, 4.0]),
        "n_atoms": torch.tensor([2, 4]),
    }

    loss, num_obs = output.calculate_loss(pred, target)

    assert torch.isclose(loss, torch.tensor(2.5))
    assert num_obs == 2


def test_per_atom_energy_loss_can_be_disabled():
    output = ModelOutput(
        name="energy",
        loss_fn=torch.nn.MSELoss(),
        per_atom_loss=False,
    )
    pred = {"energy": torch.tensor([6.0, 8.0])}
    target = {
        "energy": torch.tensor([2.0, 4.0]),
        "n_atoms": torch.tensor([2, 4]),
    }

    loss, _ = output.calculate_loss(pred, target)

    assert torch.isclose(loss, torch.tensor(16.0))


def test_per_atom_energy_loss_broadcasts_over_trailing_dimensions():
    output = ModelOutput(
        name="energy",
        loss_fn=torch.nn.MSELoss(),
        per_atom_loss=True,
    )
    pred = {"energy": torch.tensor([[6.0], [8.0]])}
    target = {
        "energy": torch.tensor([[2.0], [4.0]]),
        "n_atoms": torch.tensor([2, 4]),
    }

    loss, _ = output.calculate_loss(pred, target)

    assert torch.isclose(loss, torch.tensor(2.5))


def test_metric_states_are_independent_and_compute_over_the_full_epoch():
    output = ModelOutput(
        "forces",
        metrics={
            "mae": MeanAbsoluteError(),
            "rmse": MeanSquaredError(squared=False),
        },
    )

    assert output.train_metrics["mae"].sum_abs_error.data_ptr() != output.val_metrics["mae"].sum_abs_error.data_ptr()
    assert output.val_metrics["mae"].sum_abs_error.data_ptr() != output.test_metrics["mae"].sum_abs_error.data_ptr()

    first = output.calculate_metrics(
        {"forces": torch.tensor([10.0])},
        {"forces": torch.tensor([0.0])},
        "val",
    )
    second = output.calculate_metrics(
        {"forces": torch.tensor([2.0, 2.0, 2.0])},
        {"forces": torch.zeros(3)},
        "val",
    )

    assert torch.isclose(first["val_forces_mae"], torch.tensor(10.0))
    assert torch.isclose(second["val_forces_mae"], torch.tensor(2.0))
    epoch = output.accumulate_metrics("val")
    assert torch.isclose(epoch["mae"], torch.tensor(4.0))
    assert torch.isclose(epoch["rmse"], torch.sqrt(torch.tensor(28.0)))
    assert output.train_metrics["mae"].update_count == 0
    assert output.test_metrics["mae"].update_count == 0


def test_atoms_metric_compute_is_flattened_for_canonical_names():
    output = ModelOutput(
        "energy",
        metrics={
            "mae": PerAtomMAE(
                size_key="n_atoms",
                value_key="energy",
                compute_overall=True,
            )
        },
    )
    output.calculate_metrics(
        {"energy": torch.tensor([4.0, 9.0]), "n_atoms": torch.tensor([2, 3])},
        {"energy": torch.tensor([2.0, 3.0]), "n_atoms": torch.tensor([2, 3])},
        "val",
    )

    epoch = output.accumulate_metrics("val")

    assert epoch == {"mae_pa": 1.5, "mae": 4.0}


def test_epoch_metrics_use_canonical_names_and_reset_the_stage():
    output = ModelOutput("forces", metrics={"mae": MeanAbsoluteError()})
    output.calculate_metrics(
        {"forces": torch.tensor([10.0])},
        {"forces": torch.tensor([0.0])},
        "val",
    )
    output.calculate_metrics(
        {"forces": torch.tensor([2.0, 2.0, 2.0])},
        {"forces": torch.zeros(3)},
        "val",
    )
    task = LitNNP(
        model=nn.Module(),
        outputs=[output],
        optimizer=torch.optim.SGD,
    )
    task.log_dict = Mock()

    epoch = task._log_epoch_metrics("val")

    assert torch.isclose(epoch["val_forces_mae"], torch.tensor(4.0))
    logged = task.log_dict.call_args.args[0]
    assert set(logged) == {"val_forces_mae"}
    assert "val_forces_mae_batch" not in logged
    assert task.log_dict.call_args.kwargs["on_step"] is False
    assert task.log_dict.call_args.kwargs["on_epoch"] is True
    assert output.val_metrics["mae"].update_count == 0


def test_validation_batch_metric_has_a_step_only_name():
    output = ModelOutput(
        "forces",
        loss_fn=nn.MSELoss(),
        metrics={"mae": MeanAbsoluteError()},
    )
    task = LitNNP(
        model=nn.Identity(),
        outputs=[output],
        optimizer=torch.optim.SGD,
    )
    task.rescale_layers = []
    task.log = Mock()
    task.log_dict = Mock()
    task._write_log_only = Mock()
    task._trainer = SimpleNamespace(log_every_n_steps=1, datamodule=None, current_epoch=0)
    batch = {
        "forces": torch.tensor([1.0, 2.0]),
        "n_atoms": torch.tensor([2]),
    }

    task.validation_step(batch, batch_idx=0)

    logged = task.log_dict.call_args.args[0]
    assert set(logged) == {"val_forces_mae_batch"}
    assert task.log_dict.call_args.kwargs["on_step"] is True
    assert task.log_dict.call_args.kwargs["on_epoch"] is False


def test_sanity_validation_is_reset_without_logging():
    output = ModelOutput("forces", metrics={"mae": MeanAbsoluteError()})
    output.calculate_metrics(
        {"forces": torch.tensor([3.0])},
        {"forces": torch.tensor([0.0])},
        "val",
    )
    task = LitNNP(
        model=nn.Module(),
        outputs=[output],
        optimizer=torch.optim.SGD,
    )
    task.log_dict = Mock()
    task._trainer = SimpleNamespace(sanity_checking=True)

    task.on_validation_epoch_end()

    task.log_dict.assert_not_called()
    assert output.val_metrics["mae"].update_count == 0
