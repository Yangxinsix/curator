"""Regressions at the ordinary trainer and nested-data boundaries used by ACVR."""

from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys

import pytorch_lightning as pl
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from curator.data.utils import iter_batches
from curator.model import LitNNP, NeuralNetworkPotential
from curator.train import ModelOutput


class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(0.5))

    def forward(self, batch):
        return {"energy": self.slope * batch["feature"]}


@pytest.mark.parametrize("stage", ["validate", "test"])
def test_fresh_trainer_evaluates_without_fit_or_rescale_layers(stage):
    model = NeuralNetworkPotential(EnergyModel(), model_outputs=["energy"])
    task = LitNNP(
        model,
        [ModelOutput("energy", loss_fn=nn.MSELoss())],
        partial(torch.optim.SGD, lr=0.1),
    )
    loader = DataLoader([
        {"feature": torch.tensor(float(index)), "energy": torch.tensor(float(index)), "n_atoms": torch.tensor(1)}
        for index in (1, 2)
    ], batch_size=2)
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, inference_mode=False,
    )
    metrics = getattr(trainer, stage)(task, dataloaders=loader, verbose=False)
    prefix = "val" if stage == "validate" else "test"
    assert metrics[0][prefix + "_total_loss_epoch"] == pytest.approx(0.625)
    assert task.rescale_layers == []


def test_iter_batches_moves_nested_condition_tensors_and_preserves_indices():
    batch = {
        "n_atoms": torch.tensor([2]),
        "variational": {
            "condition_geometry_index": torch.tensor([0, 0]),
            "labels": {"energy": {"values": torch.tensor([1.0, 2.0]), "present": torch.tensor([True, False])}},
            "responses": [{"values": torch.ones(2), "unit": "1/eV"}],
        },
    }
    loader = DataLoader([batch], batch_size=None)
    converted = next(iter_batches(loader, batch_size=1, device="cpu", dtype=torch.float64))
    conditions = converted["variational"]
    assert conditions["labels"]["energy"]["values"].dtype == torch.float64
    assert conditions["responses"][0]["values"].dtype == torch.float64
    assert conditions["condition_geometry_index"].dtype == torch.long
    assert conditions["labels"]["energy"]["present"].dtype == torch.bool
    assert conditions["responses"][0]["unit"] == "1/eV"


def test_structured_loss_is_not_weighted_twice_alongside_legacy_loss():
    legacy = ModelOutput("energy", loss_fn=nn.MSELoss())
    structured = ModelOutput("variational", loss_fn=nn.MSELoss())
    structured.handles_batch_weight = True
    structured.handles_domain_loss = True
    task = LitNNP(nn.Identity(), [legacy, structured], torch.optim.SGD)
    losses = {"train_total_loss": torch.tensor(5.0), "train_energy_loss": torch.tensor(2.0), "train_variational_loss": torch.tensor(3.0)}
    task._apply_batch_weight(losses, {}, {"weight": torch.tensor([2.0, 4.0])})
    assert losses["train_energy_loss"] == 6
    assert losses["train_variational_loss"] == 3
    assert losses["train_total_loss"] == 9
    task.normalize_domain_loss = "relative"
    task._domain_loss_scales = {"0": {"variational": 7.0}}
    assert task._loss_scale_factor(structured, {}) == 1.0


def test_cli_physical_proposal_resume_and_independent_evaluation(tmp_path):
    """Exercise the public CLI, serialized optimizer, and held-out evaluator."""
    repository = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    dimensions = ["data.num_train=8", "data.num_val=2", "data.num_test=2", "data.num_conditions=3", "data.batch_size=4"]

    def run(name, config, overrides, evaluate=False):
        entrypoint = "evaluate" if evaluate else "train"
        command = [sys.executable, "-c", f"from curator.commands.hydra_entrypoints import {entrypoint}; {entrypoint}()",
                   "--config-name", config, f"run_path={tmp_path / name}", *overrides]
        completed = subprocess.run(command, cwd=repository, env=environment, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        (tmp_path / (name + ".log")).write_text(completed.stdout)
        assert completed.returncode == 0, completed.stdout[-12000:]
        if not evaluate:
            return torch.load(tmp_path / name / "model_path/last.ckpt", weights_only=False, map_location="cpu")

    physical = run("physical", "acvr_oracle", dimensions + ["trainer.max_epochs=2"])
    assert physical["epoch"] == 1 and physical["global_step"] == 4
    assert physical["optimizer_states"][0]["state"]
    for output in physical["outputs"]:
        assert output.accumulate_metrics("train") == {}
        assert output.accumulate_metrics("val") == {}
    source = tmp_path / "physical/model_path/last.ckpt"
    from curator.utils import load_model

    state_checkpoint = dict(physical)
    state_checkpoint.pop("model")
    state_checkpoint["model_save_mode"] = "state_dict"
    state_path = tmp_path / "physical_state_dict.ckpt"
    torch.save(state_checkpoint, state_path)
    rebuilt = load_model(state_path, device="cpu")
    for name, expected in physical["model"].state_dict().items():
        actual = rebuilt.state_dict()[name]
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            assert actual == expected
    assert physical["model_params"].representation.seed == 4
    assert physical["model_params"].representation.data_seed == 17
    assert "val_total_loss_exact" in physical["callbacks"][next(
        key for key in physical["callbacks"] if key.startswith("ModelCheckpoint")
    )]["monitor"]
    resumed = run("resumed", "acvr_oracle", dimensions + ["trainer.max_epochs=3", f"model_path={{path: {source}, mode: resume}}"])
    continuous = run("continuous", "acvr_oracle", dimensions + ["trainer.max_epochs=3"])
    assert resumed["epoch"] == 2 and resumed["global_step"] == continuous["global_step"] == 6
    for name, value in continuous["state_dict"].items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(resumed["state_dict"][name], value, rtol=0, atol=0)
        else:
            assert resumed["state_dict"][name] == value
    for index, expected in continuous["optimizer_states"][0]["state"].items():
        actual = resumed["optimizer_states"][0]["state"][index]
        for key, value in expected.items():
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)

    proposal = run("proposal", "acvr_oracle_proposal", dimensions + ["trainer.max_epochs=2", f"model_path.path={source}"])
    physical_keys = [name for name in physical["state_dict"] if ".physical." in name]
    proposal_keys = [name for name in physical["state_dict"] if ".proposal." in name]
    for name in physical_keys:
        torch.testing.assert_close(proposal["state_dict"][name], physical["state_dict"][name], rtol=0, atol=0)
    assert any(not torch.equal(proposal["state_dict"][name], physical["state_dict"][name]) for name in proposal_keys)
    assert proposal["model"].representation.stage == "proposal"
    assert not any(parameter.requires_grad for parameter in proposal["model"].representation.physical.parameters())

    run("evaluation", "acvr_oracle_evaluate", [f"model_path={tmp_path / 'proposal/model_path/last.ckpt'}",
        "datapath.num_train=8", "datapath.num_val=2", "datapath.num_test=2", "datapath.num_conditions=3"], evaluate=True)
    metrics = json.loads((tmp_path / "evaluation/metrics.json").read_text())
    assert metrics["num_structures"] == 2 and metrics["num_conditions"] == 6
    assert metrics["loss"] > 0
    assert set(metrics["terms"]) == {"condition_energy", "condition_forces", "variable_value", "response"}
    records = torch.load(tmp_path / "evaluation/results.pt", weights_only=False)["batches"]
    assert records and all(record["batch"]["variational"] for record in records)


def test_structured_epoch_monitor_uses_independent_term_denominators():
    from curator.train import VariationalJointLoss, VariationalModelOutput
    from test_acvr_output_lifecycle import _inputs

    output = VariationalModelOutput(VariationalJointLoss({"energy": 1.0}, {"energy": 1.0}))
    for errors in ([1.0], [3.0, 3.0, 3.0]):
        prediction, batch = _inputs(output, errors)
        output.calculate_loss(prediction, batch, subset="val")
    task = LitNNP(nn.Identity(), [output], torch.optim.SGD)
    metrics = task._collect_epoch_metrics("val")
    assert metrics["val_total_loss_exact"] == pytest.approx(7.0)
    assert metrics["val_variational_loss_exact"] == pytest.approx(7.0)
