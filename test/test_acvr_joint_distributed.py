"""Actual PaiNN joint training under Lightning's automatic optimization."""

import argparse
import copy
from dataclasses import asdict
from datetime import timedelta
from functools import partial
import os
from pathlib import Path
import subprocess
import sys

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import pytest
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import torch

from curator.model import LitNNP
from curator.train import VariationalModelOutput
from curator.data import properties
from curator.data.variational_data import canonical_boundary_hash
from curator.variational import BoundarySpec, ControlSpec, GroupSpec, LinearTermSpec, VariableSpec


def _objects(batch_size=4, *, compute_responses=False):
    configs = Path(__file__).resolve().parents[1] / "curator/configs"
    with initialize_config_dir(version_base=None, config_dir=str(configs)):
        config = compose(config_name="acvr_joint", overrides=[
            "data.num_train=4", "data.num_val=2", "data.num_test=2",
            "data.num_conditions=2", f"data.batch_size={batch_size}",
            "model.representation.num_features=4", "model.representation.num_interactions=1",
            "model.representation.num_basis=4", "model.output_modules.0.hidden_features=4",
        ])
    torch.manual_seed(config.seed)
    data = instantiate(config.data)
    data.shuffle = False
    data.setup()
    model = instantiate(config.model).double()
    model.initialize_modules(data)
    model.double()
    model.output_modules[0].compute_responses = compute_responses
    loss = instantiate(config.task.outputs[0].loss_fn)
    if compute_responses:
        loss.response_weight = 1.0
        loss.response_scales = {"1/eV": 1.0}
    task = LitNNP(model, [VariationalModelOutput(loss)],
                  partial(torch.optim.SGD, lr=0.003, momentum=0.2))
    assert task.automatic_optimization
    return task, data


class SaveTrainingState(pl.Callback):
    def __init__(self, directory):
        self.directory = Path(directory)
        self.gradients = None

    def on_before_optimizer_step(self, trainer, module, optimizer):
        self.gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in module.model.named_parameters()
        }

    def on_train_end(self, trainer, module):
        self.directory.mkdir(parents=True, exist_ok=True)
        torch.save({
            "parameters": {name: parameter.detach().cpu().clone()
                           for name, parameter in module.model.named_parameters()},
            "gradients": self.gradients,
            "global_step": trainer.global_step,
            "world_size": trainer.world_size,
            "automatic_optimization": module.automatic_optimization,
        }, self.directory / f"rank{trainer.global_rank}.pt")


def _train_worker(directory, *, devices=1, batch_size=4, accumulate=1, precision="64-true",
                  residual_after_first=False):
    torch.set_num_threads(1)
    task, data = _objects(batch_size)
    if precision == "bf16-mixed":
        task.float()
        for dataset in data.datasets.values():
            dataset.samples = [sample.to(dtype=torch.float32) for sample in dataset.samples]
    callbacks = [SaveTrainingState(directory)]
    if residual_after_first:
        task.optimizer = partial(torch.optim.AdamW, lr=0.003, weight_decay=0.2)
        callbacks.append(ResidualAfterSupervisedStep())
    strategy = (
        DDPStrategy(start_method="fork", process_group_backend="gloo",
                    find_unused_parameters=True, timeout=timedelta(seconds=60))
        if devices > 1 else "auto"
    )
    trainer = pl.Trainer(
        accelerator="cpu", devices=devices, strategy=strategy, precision=precision,
        max_epochs=1, accumulate_grad_batches=accumulate,
        inference_mode=False, deterministic=True, logger=False, enable_checkpointing=False,
        enable_model_summary=False, enable_progress_bar=False, num_sanity_val_steps=0,
        limit_val_batches=0, callbacks=callbacks,
        default_root_dir=directory,
    )
    trainer.fit(task, datamodule=data)


def _run(tmp_path, name, *options):
    directory = tmp_path / name
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(directory), *options]
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=environment,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=150)
    (tmp_path / f"{name}.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout[-16000:]
    return torch.load(directory / "rank0.pt", map_location="cpu", weights_only=False)


def _assert_training_equal(actual, expected):
    assert actual["global_step"] == expected["global_step"] == 1
    assert actual["automatic_optimization"] and expected["automatic_optimization"]
    for group in ("parameters", "gradients"):
        assert actual[group].keys() == expected[group].keys()
        for name, value in expected[group].items():
            if value is None:
                assert actual[group][name] is None, name
            else:
                torch.testing.assert_close(actual[group][name], value, rtol=2e-9, atol=2e-11, msg=name)


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
                    reason="CPU DDP requires gloo")
def test_two_rank_joint_training_matches_single_process(tmp_path):
    reference = _run(tmp_path, "single")
    distributed = _run(tmp_path, "distributed", "--devices", "2", "--batch-size", "2")
    assert distributed["world_size"] == 2
    _assert_training_equal(distributed, reference)
    second_rank = torch.load(tmp_path / "distributed/rank1.pt", map_location="cpu", weights_only=False)
    _assert_training_equal(second_rank, distributed)


def test_joint_gradient_accumulation_matches_effective_batch(tmp_path):
    reference = _run(tmp_path, "batch4")
    accumulated = _run(tmp_path, "batch2_accum2", "--batch-size", "2", "--accumulate", "2")
    _assert_training_equal(accumulated, reference)


class ResidualAfterSupervisedStep(pl.Callback):
    """Populate AdamW momentum first, then test actual residual-only updates."""

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        if batch_idx == 1:
            loss = module.outputs[0].loss_fn
            loss.supervised_weights = {name: 0.0 for name in loss.supervised_weights}

    def on_before_optimizer_step(self, trainer, module, optimizer):
        if trainer.global_step == 1:
            for name, parameter in module.model.named_parameters():
                if ".proposal." not in name:
                    assert parameter.grad is None, name
            self.checked_gradients = True

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        optimizer = trainer.optimizers[0]
        if batch_idx == 0:
            self.parameters = {name: parameter.detach().clone()
                               for name, parameter in module.model.named_parameters()}
            self.states = {name: copy.deepcopy(optimizer.state[parameter])
                           for name, parameter in module.model.named_parameters()
                           if ".proposal." not in name and parameter in optimizer.state}
            assert self.states
            assert any(state["exp_avg"].abs().max() > 0 for state in self.states.values())
        elif batch_idx == 1:
            assert self.checked_gradients
            changed_proposal = False
            for name, parameter in module.model.named_parameters():
                if ".proposal." in name:
                    changed_proposal |= not torch.equal(parameter, self.parameters[name])
                    continue
                torch.testing.assert_close(parameter, self.parameters[name], rtol=0, atol=0, msg=name)
                if name in self.states:
                    for key, value in self.states[name].items():
                        torch.testing.assert_close(optimizer.state[parameter][key], value, rtol=0, atol=0, msg=name)
            assert changed_proposal
            self.completed = True

    def on_train_end(self, trainer, module):
        assert self.completed and trainer.global_step == 2


def test_residual_only_step_keeps_physics_and_existing_adamw_momentum_unchanged(tmp_path):
    torch.set_num_threads(1)
    task, data = _objects(batch_size=2)
    task.optimizer = partial(torch.optim.AdamW, lr=0.003, weight_decay=0.2)
    check = ResidualAfterSupervisedStep()
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, precision="64-true", max_epochs=1,
        inference_mode=False, logger=False, enable_checkpointing=False,
        enable_model_summary=False, enable_progress_bar=False, num_sanity_val_steps=0,
        limit_val_batches=0, callbacks=[check], default_root_dir=tmp_path,
    )
    trainer.fit(task, datamodule=data)
    assert check.completed and trainer.global_step == 2


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
                    reason="CPU DDP requires gloo")
def test_ddp_residual_only_step_keeps_existing_physics_momentum_unchanged(tmp_path):
    result = _run(tmp_path, "ddp_residual", "--devices", "2", "--batch-size", "1", "--residual-after-first")
    assert result["world_size"] == result["global_step"] == 2
    assert any(value is not None for name, value in result["gradients"].items() if ".proposal." in name)
    assert all(value is None for name, value in result["gradients"].items() if ".proposal." not in name)


def test_joint_bfloat16_mixed_precision_has_finite_gradients(tmp_path):
    result = _run(tmp_path, "bf16", "--precision", "bf16-mixed")
    assert result["global_step"] == 1
    assert any(value is not None for value in result["gradients"].values())
    for group in ("parameters", "gradients"):
        for value in result[group].values():
            if value is not None:
                assert torch.isfinite(value).all()


def test_field_response_with_geometry_dependent_controls_includes_conjugate_coupling():
    torch.set_num_threads(1)
    task, data = _objects(batch_size=1)
    model = task.model.eval()
    output = model.output_modules[0]
    output.state_mode = "converged"
    output.compute_responses = True
    boundary = BoundarySpec(
        VariableSpec("internal", "scalar", 2, "dimensionless", "eV"),
        groups=(GroupSpec.all(),),
        controls=(ControlSpec("weighted_field", (LinearTermSpec("all", weight_field="weights"),),
                              (0.2, -0.1), unit="eV"),),
    )
    payload = asdict(boundary)

    def evaluate(field=None):
        batch = next(iter(data.train_dataloader()))
        positions = batch[properties.positions].detach().requires_grad_(True)
        batch[properties.positions] = positions
        batch["weights"] = positions[:, 0] + 2.0
        batch["variational"] = {
            "condition_geometry_index": torch.tensor([0]), "boundary_spec": (payload,),
            "boundary_hash": (canonical_boundary_hash(payload),),
            "condition_state_ptr": torch.tensor([0, 4]),
        }
        if field is not None:
            batch["external_field"] = torch.tensor([[field, 0.0]], dtype=torch.float64)
        return model(batch)["variational_prediction"]

    # No external tensor at the central point: the built-in reference field
    # already requires gradients through geometric control weights. It must be
    # a different autograd node from the independent local field perturbation.
    central = evaluate()
    response = central["responses"]["values"].reshape(4, 4)
    step = 1e-5
    lower, upper = evaluate(-step), evaluate(step)
    derivative = (upper["labels"]["variable_value"]["values"]
                  - lower["labels"]["variable_value"]["values"]) / (2 * step)
    torch.testing.assert_close(response[:, ::2].sum(-1), derivative, atol=2e-7, rtol=2e-5)


def test_optional_response_supervision_backpropagates_through_real_joint_gnn(tmp_path):
    torch.set_num_threads(1)
    task, data = _objects(compute_responses=True)
    batch = next(iter(data.train_dataloader()))
    result = task.outputs[0].loss_fn(task.model(batch), batch)
    response_loss = result["terms"]["response"]["loss"]
    assert result["terms"]["response"]["status"] == "active"
    assert torch.isfinite(response_loss) and response_loss.requires_grad
    # Response-only supervision must actually reach the predicted state mapping,
    # instead of merely reporting a detached Jacobian as a diagnostic tensor.
    proposal_gradients = torch.autograd.grad(
        response_loss, tuple(task.model.output_modules[0].proposal.parameters()), allow_unused=True,
    )
    assert any(value is not None and value.abs().max() > 0 for value in proposal_gradients)
    callback = SaveTrainingState(tmp_path)
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, precision="64-true", max_epochs=1,
        inference_mode=False, logger=False, enable_checkpointing=False,
        enable_model_summary=False, enable_progress_bar=False, num_sanity_val_steps=0,
        limit_val_batches=0, callbacks=[callback], default_root_dir=tmp_path,
    )
    trainer.fit(task, datamodule=data)
    assert trainer.global_step == 1
    for group in task.model.parameter_groups():
        gradients = [parameter.grad for parameter in group.params if parameter.grad is not None]
        assert gradients and all(torch.isfinite(value).all() for value in gradients), group.name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--precision", default="64-true")
    parser.add_argument("--residual-after-first", action="store_true")
    arguments = parser.parse_args()
    _train_worker(arguments.worker, devices=arguments.devices, batch_size=arguments.batch_size,
                  accumulate=arguments.accumulate, precision=arguments.precision,
                  residual_after_first=arguments.residual_after_first)
