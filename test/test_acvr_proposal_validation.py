"""Reference-failure accounting and the optional equilibrium validation gate."""

import copy
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

from curator.data.collate_atoms_data import collate_atoms_data
from curator.evaluate.variational import VariationalEvaluator
from curator.train.proposal_validation import ProposalEquilibriumValidation
from curator.train.variational_losses import VariationalJointLoss


def _objects():
    config = OmegaConf.load(Path(__file__).resolve().parents[1] / "curator/configs/acvr_periodic_charge.yaml")
    config.data.num_train, config.data.num_val, config.data.num_test = 2, 2, 1
    config.data.plane_side, config.data.batch_size = 2, 1
    config.model.representation.num_features = 4
    config.model.representation.num_interactions = 1
    config.model.representation.num_basis = 4
    config.model.output_modules[0].hidden_features = 4
    config.model.output_modules[0].internal_energy.hidden_features = 4
    config.model.output_modules[0].proposal.hidden_features = 4
    config.model.output_modules[0].proposal.num_heads = 2
    config.model.output_modules[0].proposal.communication = "pointwise"
    config.model.output_modules[0].residual_scale = 3.0
    torch.manual_seed(17)
    data = instantiate(config.data)
    data.setup()
    model = instantiate(config.model).double()
    model.initialize_modules(data)
    model.double()
    loss = instantiate(config.task.outputs[0].loss_fn)
    return config, data, model, loss


def test_evaluator_physical_residual_constraints_relative_error_and_timing(tmp_path):
    _, data, model, loss = _objects()
    evaluator = VariationalEvaluator(model, loss, output_dir=tmp_path, batch_size=1,
                                     state_mode="predicted", compare_converged=True,
                                     compute_responses=False, record_reference_failures=True)
    metrics = evaluator.evaluate(data.datasets["val"])
    assert metrics["num_structures"] == 2 and metrics["num_conditions"] == 12
    assert metrics["reference_solver"]["n_failed_conditions"] == 0
    assert metrics["reference_solver"]["n_converged"] == 12
    assert metrics["constraints"]["n_failed_conditions"] == 0
    assert metrics["constraints"]["n_conditions"] == 12
    assert metrics["constraints"]["max_absolute"] < 1e-12
    assert all(math.isfinite(value) and value > 0 for value in metrics["phase_seconds"].values())
    artifact = torch.load(tmp_path / "results.pt", map_location="cpu", weights_only=False)
    raw_residuals, predicted_states, reference_states = [], [], []
    for record in artifact["batches"]:
        prediction = record["prediction"]["variational_prediction"]
        residual = prediction["residuals"]["stationarity"]
        assert residual["normalization"]["scale"] == 3.0
        raw_residuals.append(3.0 * residual["values"])
        predicted_states.append(prediction["labels"]["variable_value"]["values"])
        reference_states.append(record["comparison_prediction"]["variational_prediction"]
                                ["labels"]["variable_value"]["values"])
    residual = torch.cat(raw_residuals)
    assert metrics["physical_stationarity"]["unit"] == "V"
    assert metrics["physical_stationarity"]["n_components"] == residual.numel()
    assert metrics["physical_stationarity"]["rms"] == pytest.approx(float(residual.square().mean().sqrt()))
    assert metrics["physical_stationarity"]["absolute_p95"] == pytest.approx(float(residual.abs().quantile(.95)))
    assert metrics["physical_stationarity"]["absolute_max"] == pytest.approx(float(residual.abs().max()))
    reference = torch.cat(reference_states)
    difference = torch.cat(predicted_states) - reference
    state = metrics["comparison"]["difference_from_predicted"]["variable_value"]
    assert state["relative_norm_error"] == pytest.approx(float(difference.norm() / reference.norm()))
    assert state["scalar_rmse"] == pytest.approx(float(difference.square().mean().sqrt()))
    assert json.loads((tmp_path / "metrics.json").read_text()) == json.loads(json.dumps(metrics))
    assert model.training and model.output_modules[0].state_mode == "predicted"


def test_conq_only_evaluation_handles_the_empty_control_derivative(tmp_path):
    _, data, model, _ = _objects()
    dataset = copy.copy(data.datasets["val"])
    dataset.samples = [replace(sample, variational=replace(sample.variational,
                               conditions=sample.variational.conditions[:3])) for sample in dataset.samples]
    model.output_modules[0].retain_response_graph = False
    loss = VariationalJointLoss(supervised_weights={"condition_energy": 1.0},
                               supervised_scales={"condition_energy": 1.0})
    metrics = VariationalEvaluator(model, loss, output_dir=tmp_path, batch_size=1,
                                   state_mode="predicted", compare_converged=True,
                                   compute_responses=False).evaluate(dataset)
    assert metrics["reference_solver"]["n_converged"] == 6
    control = metrics["comparison"]["difference_from_predicted"].get("control_derivative")
    assert control is None or (control["n_scalars"] == 0 and control["scalar_rmse"] is None)


@pytest.mark.parametrize("all_fail", [False, True])
def test_reference_failure_preserves_primary_metrics_and_full_denominator(tmp_path, monkeypatch, all_fail):
    _, data, model, loss = _objects()
    baseline = VariationalEvaluator(model, loss, output_dir=tmp_path / "baseline", batch_size=1,
                                     state_mode="predicted", compute_responses=False).evaluate(data.datasets["val"])
    output = model.output_modules[0]
    original = output.forward
    reference_calls = 0

    def sometimes_failing(batch):
        nonlocal reference_calls
        if output.state_mode == "converged":
            reference_calls += 1
            if reference_calls == 1 or all_fail:
                raise RuntimeError("deliberate reference failure")
        return original(batch)

    monkeypatch.setattr(output, "forward", sometimes_failing)
    evaluator = VariationalEvaluator(model, loss, output_dir=tmp_path / "failures", batch_size=1,
                                     state_mode="predicted", compare_converged=True,
                                     compute_responses=False, record_reference_failures=True)
    metrics = evaluator.evaluate(data.datasets["val"])
    assert metrics["terms"] == baseline["terms"]
    assert metrics["loss"] == baseline["loss"]
    assert metrics["physical_stationarity"] == baseline["physical_stationarity"]
    assert metrics["num_conditions"] == metrics["reference_solver"]["n_conditions"] == 12
    assert metrics["reference_solver"]["n_failed_conditions"] == (12 if all_fail else 6)
    assert metrics["reference_solver"]["n_converged"] == (0 if all_fail else 6)
    assert metrics["comparison"]["terms"]["condition_energy"]["coverage"] == (0 if all_fail else .5)
    records = torch.load(tmp_path / "failures/results.pt", weights_only=False)["batches"]
    assert len(records) == 2 and "prediction" in records[0] and "reference_failure" in records[0]
    assert records[0]["reference_failure"]["n_conditions"] == 6
    assert "comparison_prediction" not in records[0]
    assert ("comparison_prediction" in records[1]) != all_fail
    assert model.training and output.state_mode == "predicted"
    reference_calls = 0
    with pytest.raises(RuntimeError, match="deliberate reference failure"):
        VariationalEvaluator(model, loss, output_dir=tmp_path / "strict", batch_size=1,
                             state_mode="predicted", compare_converged=True,
                             compute_responses=False).evaluate(data.datasets["val"])


def test_callback_schedule_fixed_subset_gate_and_restore_contract(tmp_path):
    _, data, model, loss = _objects()
    logs, saved = {}, []
    module = SimpleNamespace(model=model, outputs=[SimpleNamespace(loss_fn=loss)],
                             device=torch.device("cpu"), log=lambda key, value, **kwargs: logs.update({key: value}))
    trainer = SimpleNamespace(sanity_checking=False, current_epoch=0, world_size=1,
                              val_dataloaders=data.val_dataloader(), default_root_dir=tmp_path,
                              save_checkpoint=lambda path: saved.append(path))
    callback = ProposalEquilibriumValidation(every_n_epochs=2, num_geometries=1,
                                             compute_responses=False, supervised_limits={"condition_energy": 1e6})
    callback.on_validation_epoch_end(trainer, module)
    assert not logs and not (tmp_path / "proposal_validation").exists()
    trainer.current_epoch = 1
    trainer.sanity_checking = True
    callback.on_validation_epoch_end(trainer, module)
    assert not logs
    trainer.sanity_checking = False
    callback.on_validation_epoch_end(trainer, module)
    metrics_path = tmp_path / "proposal_validation/epoch_0001/metrics.json"
    metrics = json.loads(metrics_path.read_text())
    assert metrics["num_structures"] == 1 and metrics["num_conditions"] == 6
    artifact = torch.load(metrics_path.parent / "results.pt", weights_only=False)
    assert artifact["batches"][0]["batch"]["variational"]["geometry_uid"] == (
        data.datasets["val"][0].variational.geometry_uid,)
    assert math.isfinite(callback.best_state_rmse) and not saved
    callback.on_train_epoch_end(trainer, module)
    assert saved == [tmp_path / "model_path/best_proposal_equilibrium.ckpt"]
    callback.on_train_epoch_end(trainer, module)
    assert len(saved) == 1
    restored = ProposalEquilibriumValidation(every_n_epochs=2, num_geometries=1,
                                              compute_responses=False, supervised_limits={"condition_energy": 1e6})
    restored.load_state_dict(copy.deepcopy(callback.state_dict()))
    assert restored.state_dict() == callback.state_dict()
    restored.on_validation_epoch_end(trainer, module)
    restored.on_train_epoch_end(trainer, module)
    assert len(saved) == 1  # Identical RMSE does not replace the best checkpoint.
    for change in ({"every_n_epochs": 1}, {"compute_responses": True}, {"num_geometries": 2},
                   {"supervised_limits": {"condition_energy": 1.0}}):
        kwargs = dict(every_n_epochs=2, num_geometries=1, compute_responses=False,
                      supervised_limits={"condition_energy": 1e6})
        kwargs.update(change)
        with pytest.raises(ValueError, match="differs from checkpoint"):
            ProposalEquilibriumValidation(**kwargs).load_state_dict(callback.state_dict())
    trainer.val_dataloaders = DataLoader(Subset(data.datasets["val"], [1, 0]), batch_size=1,
                                        collate_fn=collate_atoms_data)
    with pytest.raises(ValueError, match="fixed geometry identities"):
        restored.on_validation_epoch_end(trainer, module)


def test_constraint_failures_are_counted_and_disqualify_checkpoint(tmp_path, monkeypatch):
    _, data, model, loss = _objects()
    output = model.output_modules[0]
    original = output.forward

    def broken_constraint_report(batch):
        result = original(batch)
        if output.state_mode == "predicted":
            violation = result["variational_prediction"]["constraint_violation"]
            violation["values"] = violation["values"].clone()
            violation["values"][0] = 1e-4
        return result

    monkeypatch.setattr(output, "forward", broken_constraint_report)
    saved = []
    module = SimpleNamespace(model=model, outputs=[SimpleNamespace(loss_fn=loss)],
                             device=torch.device("cpu"), log=lambda *args, **kwargs: None)
    trainer = SimpleNamespace(sanity_checking=False, current_epoch=0, world_size=1,
                              val_dataloaders=data.val_dataloader(), default_root_dir=tmp_path,
                              save_checkpoint=lambda path: saved.append(path))
    callback = ProposalEquilibriumValidation(every_n_epochs=1, num_geometries=1,
                                             compute_responses=False, supervised_limits={"condition_energy": 1e6})
    callback.on_validation_epoch_end(trainer, module)
    callback.on_train_epoch_end(trainer, module)
    metrics = json.loads((tmp_path / "proposal_validation/epoch_0000/metrics.json").read_text())
    assert metrics["constraints"]["n_failed_conditions"] == 1
    assert metrics["constraints"]["n_conditions"] == 6
    assert metrics["constraints"]["max_absolute"] == pytest.approx(1e-4)
    assert metrics["reference_solver"]["n_failed_conditions"] == 0
    assert not saved and callback.best_state_rmse == math.inf


@pytest.mark.parametrize("limits", [None, {"condition_energy": 1e6}])
def test_real_training_callback_saves_only_a_qualified_checkpoint(tmp_path, limits):
    config, data, model, _ = _objects()
    task = instantiate(config.task, model=model)
    task.save_configuration(config)
    callback = ProposalEquilibriumValidation(every_n_epochs=1, num_geometries=1,
                                             compute_responses=False, supervised_limits=limits)
    trainer = pl.Trainer(default_root_dir=tmp_path, accelerator="cpu", devices=1,
                         precision="64-true", max_epochs=1, limit_train_batches=1,
                         limit_val_batches=1, num_sanity_val_steps=0, logger=False,
                         enable_progress_bar=False, enable_model_summary=False,
                         enable_checkpointing=False, inference_mode=False, callbacks=[callback])
    trainer.fit(task, datamodule=data)
    assert (tmp_path / "proposal_validation/epoch_0000/metrics.json").exists()
    path = tmp_path / "model_path/best_proposal_equilibrium.ckpt"
    assert path.exists() == bool(limits)
    if limits:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["global_step"] == 1
        state = checkpoint["callbacks"][callback.state_key]
        assert state == callback.state_dict()
        assert math.isfinite(state["best_state_rmse"])
        assert state["subset_identity"] == ((data.datasets["val"][0].variational.geometry_uid,
                                              data.datasets["val"][0].variational.geometry_hash),)
    else:
        assert callback.best_state_rmse == math.inf
