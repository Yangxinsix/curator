"""Numerical block-update, accumulation and full-state resumption regressions."""

import copy
from functools import partial

import pytest
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from curator.data import properties
from curator.model.base import ParameterGroup
from curator.model.lit_module import LitNNP
from curator.train.variational_losses import VariationalJointLoss, VariationalModelOutput
from curator.train.variational_optimization import VariationalOptimizationTask
from curator.train.optimizer_initialization import (
    OptimizerStateInitializer, optimizer_parameter_names, restore_optimizer_state,
)


class TinyEnergy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Parameter(torch.tensor(0.6))
        self.baseline = torch.nn.Parameter(torch.tensor(0.2))
        self.functional = torch.nn.Parameter(torch.tensor(0.3))
        self.proposal = torch.nn.Parameter(torch.tensor(0.8))
        self.output_modules = torch.nn.ModuleList()
        self._initialized = True

    def parameter_groups(self):
        return [ParameterGroup(name, [getattr(self, name)]) for name in
                ("backbone", "baseline", "functional", "proposal")]

    def forward(self, batch):
        assert "used" not in batch, "inner forwards must receive pristine batches"
        batch["used"] = True
        r = batch["r"].requires_grad_(True)
        q = self.proposal * self.backbone * r + self.functional
        energy = self.baseline * r.square() + q.square() / 2
        force = -torch.autograd.grad(energy.sum(), r, create_graph=True)[0]
        residual = self.proposal * self.backbone.detach() * r.detach() + self.functional.detach() - 0.1
        return {"energy": energy, "force": force, "residual": residual}


class TinyLoss(VariationalJointLoss):
    def __init__(self):
        super().__init__({"condition_energy": 2., "condition_forces": 3.},
                         {"condition_energy": 1., "condition_forces": 1.},
                         residual_weights={"stationarity": 1.})

    def forward(self, prediction, batch):
        values = {"condition_energy": prediction["energy"] - batch["energy"],
                  "condition_forces": prediction["force"] - batch["force"],
                  "residual:stationarity": prediction["residual"]}
        terms = {}
        for name, error in values.items():
            weight = self.term_coefficients[name]
            numerator, denominator = error.square().sum(), error.new_tensor(error.numel())
            terms[name] = {"loss": weight * numerator / denominator,
                           "numerator": numerator, "denominator": denominator,
                           "coefficient": weight, "n_geometries": error.numel()}
        return {"loss": sum(term["loss"] for term in terms.values()), "terms": terms}


def batches(n=5):
    return [{"r": torch.tensor([0.3 + i * 0.12]), "energy": torch.tensor([0.1]),
             "force": torch.tensor([-0.2]), properties.n_atoms: torch.tensor([1])} for i in range(n)]


def task(rule="joint", inner=0):
    result = VariationalOptimizationTask(
        model=TinyEnergy(), outputs=[VariationalModelOutput(TinyLoss())],
        optimizer=partial(torch.optim.Adam, lr=0.02, amsgrad=True),
        update_rule=rule, inner_steps=inner, accumulation_steps=3)
    result.save_configuration(OmegaConf.create({"data": {}, "model": {}}))
    return result


def fit(module, data=None, epochs=1, callbacks=None, checkpoint=None):
    trainer = pl.Trainer(accelerator="cpu", devices=1, precision="32-true",
                         max_epochs=epochs, logger=False, enable_checkpointing=False,
                         enable_model_summary=False, enable_progress_bar=False,
                         num_sanity_val_steps=0, gradient_clip_val=None,
                         callbacks=callbacks or [])
    trainer.fit(module, train_dataloaders=DataLoader(data or batches(), batch_size=None),
                ckpt_path=checkpoint)
    return trainer


def equal_state(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal_state(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            equal_state(a, b)
    elif torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        assert actual == expected


def test_joint_matches_accumulated_adam_including_short_final_window():
    module = task()
    reference = copy.deepcopy(module.model)
    criterion = TinyLoss()
    optimizer = torch.optim.Adam([{"name": g.name, "params": list(g.params)}
                                  for g in reference.parameter_groups()], lr=0.02, amsgrad=True)
    data = batches(5)
    for start in (0, 3):
        optimizer.zero_grad(set_to_none=True)
        for original in data[start:start + 3]:
            batch = copy.deepcopy(original)
            (criterion(reference(batch), batch)["loss"] / 3).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 10.)
        optimizer.step()
    trainer = fit(module, data)
    equal_state(module.model.state_dict(), reference.state_dict())
    equal_state(trainer.optimizers[0].state_dict(), optimizer.state_dict())
    assert module.update_counts["outer_steps"] == 2
    assert module.update_counts["ef_structures"] == 5
    assert not module._pending_batches


class CheckBlocks(pl.Callback):
    def __init__(self):
        self.before = []
        self.after = []

    def on_before_optimizer_step(self, trainer, module, optimizer):
        self.before.append((copy.deepcopy(module.model.state_dict()),
                            copy.deepcopy(optimizer.state_dict()),
                            [p.grad is not None for p in module.model.parameters()]))

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if outputs is not None:
            self.after.append((copy.deepcopy(module.model.state_dict()),
                               copy.deepcopy(trainer.optimizers[0].state_dict())))


def test_alternating_updates_leave_inactive_parameters_and_adam_steps_unchanged(tmp_path):
    source = task()
    source_trainer = fit(source, batches(3))
    source_optimizer = source_trainer.optimizers[0]
    path = tmp_path / "moments.pt"
    torch.save({"optimizer_state": source_optimizer.state_dict(),
                "parameter_names": optimizer_parameter_names(source, source_optimizer)}, path)
    module = task("alternating_ef", 2)
    module.model.load_state_dict(source.model.state_dict())
    spy = CheckBlocks()
    fit(module, batches(3), callbacks=[OptimizerStateInitializer(path), spy])
    assert [record[2] for record in spy.before] == [
        [False, False, False, True], [False, False, False, True], [True, True, True, False]]
    first, second, outer = spy.before
    for name in ("backbone", "baseline", "functional"):
        equal_state(first[0][name], second[0][name])
        equal_state(first[0][name], outer[0][name])
    for index in range(3):
        equal_state(first[1]["state"][index], second[1]["state"][index])
        equal_state(first[1]["state"][index], outer[1]["state"][index])
    equal_state(outer[1]["state"][3], spy.after[-1][1]["state"][3])
    equal_state(outer[0]["proposal"], spy.after[-1][0]["proposal"])
    final_state = spy.after[-1][1]["state"]
    assert [int(final_state[i]["step"]) for i in range(4)] == [2, 2, 2, 3]
    assert module.update_counts == {"outer_steps": 1, "inner_steps": 2,
        "ef_structures": 3, "residual_only_structures": 6, "forward_minibatches": 9}


def test_frozen_proposal_keeps_coordinate_and_backbone_force_derivatives():
    model = TinyEnergy()
    batch = batches(1)[0]
    pred = model(batch)
    r = batch["r"]
    expected_force = -2 * model.baseline * r - (
        model.proposal * model.backbone * r + model.functional) * model.proposal * model.backbone
    torch.testing.assert_close(pred["force"], expected_force)
    grad = torch.autograd.grad(pred["force"].sum(), model.backbone)[0]
    expected_grad = -2 * model.proposal.square() * model.backbone * r - model.functional * model.proposal
    torch.testing.assert_close(grad, expected_grad.sum())
    # The actual block backward selects theta inputs; it must yield this same
    # derivative while keeping proposal.grad=None.
    module = task("alternating_ef", 1)
    optimizer = module.configure_optimizers()
    active = module._active_parameters(optimizer, "ef")
    force = module.model(batches(1)[0])["force"]
    force.sum().backward(inputs=active)
    torch.testing.assert_close(module.model.backbone.grad, expected_grad.sum())
    assert module.model.proposal.grad is None


def test_full_checkpoint_resume_matches_continuous_updates(tmp_path):
    continuous = task("alternating_ef", 2)
    continuous_trainer = fit(continuous, epochs=2)
    first = task("alternating_ef", 2)
    first_trainer = fit(first)
    path = tmp_path / "state.ckpt"
    first_trainer.save_checkpoint(path)
    resumed = task("alternating_ef", 2)
    resumed_trainer = fit(resumed, epochs=2, checkpoint=path)
    equal_state(resumed.model.state_dict(), continuous.model.state_dict())
    equal_state(resumed_trainer.optimizers[0].state_dict(), continuous_trainer.optimizers[0].state_dict())
    assert resumed.update_counts == continuous.update_counts


def test_restore_lr_multiplier_preserves_all_moments_and_initializer_never_replays(tmp_path):
    source = task()
    trainer = fit(source, batches(3))
    optimizer = trainer.optimizers[0]
    before = copy.deepcopy(optimizer.state_dict())
    audit = restore_optimizer_state(optimizer, before, {"proposal": 4.})
    equal_state(optimizer.state_dict()["state"], before["state"])
    assert audit["proposal"]["lr"] == 0.08
    path = tmp_path / "optimizer.pt"
    torch.save({"optimizer_state": before,
                "parameter_names": optimizer_parameter_names(source, optimizer)}, path)
    branch = task()
    branch.model.load_state_dict(source.model.state_dict())
    initializer = OptimizerStateInitializer(path, {"proposal": 4.})
    branch_trainer = fit(branch, batches(3), callbacks=[initializer])
    checkpoint = tmp_path / "branch.ckpt"
    branch_trainer.save_checkpoint(checkpoint)
    path.unlink()  # A full resume must not read or reapply the source optimizer.
    resumed = task()
    restored_initializer = OptimizerStateInitializer(path, {"proposal": 4.})
    fit(resumed, batches(3), epochs=2, callbacks=[restored_initializer], checkpoint=checkpoint)
    assert restored_initializer.audit == initializer.audit
    assert resumed.trainer.optimizers[0].param_groups[-1]["lr"] == 0.08


def test_reject_checkpoint_inside_pending_window():
    module = task()
    module._pending_batches = batches(1)
    with pytest.raises(RuntimeError, match="outer-update boundaries"):
        module.on_save_checkpoint({})


def test_joint_is_identical_to_native_lightning_automatic_accumulation():
    control = task()
    automatic = LitNNP(model=copy.deepcopy(control.model),
                       outputs=[VariationalModelOutput(TinyLoss())],
                       optimizer=partial(torch.optim.Adam, lr=0.02, amsgrad=True))
    automatic.save_configuration(OmegaConf.create({"data": {}, "model": {}}))
    trainer = pl.Trainer(accelerator="cpu", devices=1, precision="32-true",
                         max_epochs=1, logger=False, enable_checkpointing=False,
                         enable_model_summary=False, enable_progress_bar=False,
                         accumulate_grad_batches=3, gradient_clip_val=10.,
                         num_sanity_val_steps=0)
    trainer.fit(automatic, train_dataloaders=DataLoader(batches(5), batch_size=None))
    manual_trainer = fit(control)
    equal_state(control.model.state_dict(), automatic.model.state_dict())
    equal_state(manual_trainer.optimizers[0].state_dict(), trainer.optimizers[0].state_dict())


def test_alternating_joint_keeps_outer_proposal_gradients():
    module = task("alternating_joint", 2)
    spy = CheckBlocks()
    fit(module, batches(3), callbacks=[spy])
    assert [record[2] for record in spy.before] == [
        [False, False, False, True], [False, False, False, True], [True, True, True, True]]
    state = module.trainer.optimizers[0].state_dict()["state"]
    assert [int(state[i]["step"]) for i in range(4)] == [1, 1, 1, 3]


def test_native_plateau_scheduler_and_resume_checkpoint_preserve_ratios_and_state(tmp_path):
    from curator.train.joint_callbacks import JointValidationLoss, JointModelCheckpoint
    from curator.train.resume_callbacks import ResumeCheckpoint, ResumeTrainingState

    def build():
        module = task("alternating_ef", 2)
        module.model.supports_state_modes = True
        module.model.state_mode = "predicted"
        module.optimizer_groups = {"proposal": {"lr": 0.08}}
        module.scheduler = partial(torch.optim.lr_scheduler.ReduceLROnPlateau,
                                   mode="min", factor=0.5, patience=0,
                                   threshold=1000., threshold_mode="abs", eps=0.,
                                   min_lr=[0.005, 0.005, 0.005, 0.020])
        module.scheduler_monitor = "val_joint_predicted_loss"
        return module

    def run(module, directory, epochs, checkpoint=None):
        callbacks = [JointValidationLoss(), ResumeTrainingState(),
                     JointModelCheckpoint(dirpath=directory / "best", save_top_k=-1,
                                          monitor="val_joint_predicted_loss"),
                     ResumeCheckpoint(directory / "resume")]
        trainer = pl.Trainer(accelerator="cpu", devices=1, precision="32-true",
                             max_epochs=epochs, logger=False, enable_model_summary=False,
                             enable_progress_bar=False, num_sanity_val_steps=0,
                             inference_mode=False, gradient_clip_val=None,
                             callbacks=callbacks)
        trainer.fit(module, train_dataloaders=DataLoader(batches(3), batch_size=None),
                    val_dataloaders=DataLoader(batches(2), batch_size=None),
                    ckpt_path=checkpoint)
        return trainer

    continuous = build()
    reference = run(continuous, tmp_path / "continuous", 4)
    first = build()
    run(first, tmp_path / "first", 2)
    path = tmp_path / "first/resume/last.ckpt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["curator_checkpoint_phase"] == "resume_after_scheduler"
    assert checkpoint["lr_schedulers"][0]["last_epoch"] == 2
    assert [group["lr"] for group in checkpoint["optimizer_states"][0]["param_groups"]] == [
        0.01, 0.01, 0.01, 0.04]
    resumed = build()
    restored = run(resumed, tmp_path / "resumed", 4, path)
    equal_state(resumed.model.state_dict(), continuous.model.state_dict())
    equal_state(restored.optimizers[0].state_dict(), reference.optimizers[0].state_dict())
    equal_state(restored.lr_scheduler_configs[0].scheduler.state_dict(),
                reference.lr_scheduler_configs[0].scheduler.state_dict())
    assert [group["lr"] for group in restored.optimizers[0].param_groups] == [
        0.005, 0.005, 0.005, 0.02]
    assert resumed._last_scheduler_epoch == 3


def test_hydra_scheduler_converts_ordered_floors_and_preserves_lr_ratio():
    from hydra.utils import instantiate

    module = task()
    module.optimizer_groups = {"proposal": {"lr": 0.08}}
    config = OmegaConf.create({
        "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
        "_partial_": True, "_convert_": "all", "mode": "min",
        "factor": 0.5, "patience": 0, "eps": 0.0,
        "min_lr": [0.005, 0.005, 0.005, 0.020],
    })
    module.scheduler = instantiate(config)
    module.scheduler_monitor = "val_joint_predicted_loss"
    configured = module.configure_optimizers()
    scheduler = configured["lr_scheduler"]["scheduler"]
    assert scheduler.min_lrs == [0.005, 0.005, 0.005, 0.020]
    assert all(type(floor) is float for floor in scheduler.min_lrs)
    for _ in range(4):
        scheduler.step(1.0)
        rates = [group["lr"] for group in configured["optimizer"].param_groups]
        assert rates[-1] / rates[0] == 4.0
    assert rates == [0.005, 0.005, 0.005, 0.020]
