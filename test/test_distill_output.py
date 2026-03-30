from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from curator.data import properties
from curator.layer import EnergyHessianOutput, GradientOutput
from curator.layer._energy_hessian import EnergyHessianSample
from curator.model import NeuralNetworkPotential
from curator.train import PerAtomMAE
from curator.train.distill import (
    collect_distill_output_columns,
    is_offline_distill_output,
    prepare_teacher_model_for_offline_distillation,
)
from curator.train.model_output import DistillOutput
from curator.utils import scatter_add


def _component_path(name: str) -> str:
    return os.path.join("curator", "configs", "task", "outputs", "components", f"{name}.yaml")


def _task_path(name: str = "default_task.yaml") -> str:
    return os.path.join("curator", "configs", "task", name)


def _resolve_component_with_task(component: str):
    root = OmegaConf.create({"task": OmegaConf.load(_task_path())})
    root.component = OmegaConf.load(_component_path(component))
    return root.component


def _resolved_component_container(component: str, loss_weights: dict | None = None):
    root = OmegaConf.create({"task": OmegaConf.load(_task_path())})
    if loss_weights is not None:
        root.task.loss_weights = OmegaConf.create(loss_weights)
    root.component = OmegaConf.load(_component_path(component))
    return OmegaConf.to_container(root.component, resolve=True)


def _sample_batch() -> dict:
    positions = torch.tensor(
        [[0.0, 0.2, -0.1], [0.3, -0.4, 0.5]],
        dtype=torch.float32,
        requires_grad=True,
    )
    return {
        properties.n_atoms: torch.tensor([2], dtype=torch.long),
        properties.atomic_numbers: torch.tensor([1, 8], dtype=torch.long),
        properties.positions: positions,
        properties.image_idx: torch.tensor([0, 0], dtype=torch.long),
    }


class CountingEnergyRepresentation(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([float(scale)], dtype=torch.float32))
        self.register_buffer("forward_calls", torch.zeros((), dtype=torch.long))
        self.model_outputs = [properties.energy]

    def forward(self, data: dict) -> dict:
        self.forward_calls.add_(1)
        per_atom_energy = self.weight * data[properties.positions].square().sum(dim=-1, keepdim=True)
        data[properties.energy] = scatter_add(per_atom_energy, data[properties.image_idx], dim=0)
        return data


def _build_model(scale: float) -> NeuralNetworkPotential:
    return NeuralNetworkPotential(
        representation=CountingEnergyRepresentation(scale),
        output_modules=[
            GradientOutput(
                grad_on_edge_diff=False,
                grad_on_positions=True,
                model_outputs=[properties.forces],
            )
        ],
    )


def _build_hessian_model(
    scale: float,
    hessian_outputs: list[str],
    num_samples: int | None = None,
    mask_key: str | None = None,
) -> NeuralNetworkPotential:
    return NeuralNetworkPotential(
        representation=CountingEnergyRepresentation(scale),
        output_modules=[
            GradientOutput(
                grad_on_edge_diff=False,
                grad_on_positions=True,
                model_outputs=[properties.forces],
            ),
            EnergyHessianOutput(
                model_outputs=hessian_outputs,
                num_samples=num_samples,
                mask_key=mask_key,
            ),
        ],
    )


def _save_teacher(tmp_path: Path, scale: float = 2.0) -> str:
    teacher_path = tmp_path / "teacher.ckpt"
    torch.save(_build_model(scale), teacher_path)
    return str(teacher_path)


def test_distill_output_energy_loss_is_finite(tmp_path):
    teacher_path = _save_teacher(tmp_path, scale=2.0)
    student = _build_model(scale=1.0)
    batch = _sample_batch()
    pred = student(batch)

    output = DistillOutput(
        name="energy_distill",
        teacher_model_path=teacher_path,
        student_property=properties.energy,
        teacher_property=properties.energy,
    )

    loss, num_obs = output.calculate_loss(pred, batch, True)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert num_obs == pred[properties.energy].view(-1).shape[0]
    teacher_params = {id(param) for param in output._teacher_state["model"].parameters()}
    assert teacher_params
    assert all(id(param) not in teacher_params for param in output.parameters())


def test_distill_output_forces_loss_is_finite(tmp_path):
    teacher_path = _save_teacher(tmp_path, scale=2.0)
    student = _build_model(scale=1.0)
    batch = _sample_batch()
    pred = student(batch)

    output = DistillOutput(
        name="forces_distill",
        teacher_model_path=teacher_path,
        student_property=properties.forces,
        teacher_property=properties.forces,
    )

    loss, num_obs = output.calculate_loss(pred, batch, True)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert num_obs == pred[properties.forces].view(-1).shape[0]


def test_distill_outputs_share_teacher_forward_via_batch_cache(tmp_path):
    teacher_path = _save_teacher(tmp_path, scale=2.0)
    student = _build_model(scale=1.0)
    batch = _sample_batch()
    pred = student(batch)

    energy_output = DistillOutput(
        name="energy_distill",
        teacher_model_path=teacher_path,
        student_property=properties.energy,
        teacher_property=properties.energy,
        cache_key="distill_teacher",
    )
    forces_output = DistillOutput(
        name="forces_distill",
        teacher_model_path=teacher_path,
        student_property=properties.forces,
        teacher_property=properties.forces,
        cache_key="distill_teacher",
    )

    energy_loss, _ = energy_output.calculate_loss(pred, batch, True)
    forces_loss, _ = forces_output.calculate_loss(pred, batch, True)

    assert torch.isfinite(energy_loss)
    assert torch.isfinite(forces_loss)
    assert "__distill_teacher_cache__" in batch
    assert "distill_teacher" in batch["__distill_teacher_cache__"]
    teacher_model = energy_output._teacher_state["model"]
    assert int(teacher_model.representation.forward_calls.item()) == 1
    assert not forces_output._teacher_state


def test_distill_output_reads_offline_teacher_labels_from_batch():
    student = _build_model(scale=1.0)
    batch = _sample_batch()
    pred = student(batch)
    batch["teacher_energy"] = pred[properties.energy].detach() + 1.0

    output = DistillOutput(
        name="energy_distill",
        teacher_model_path=None,
        student_property=properties.energy,
        teacher_property="teacher_energy",
    )

    loss, num_obs = output.calculate_loss(pred, batch, True)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert num_obs == pred[properties.energy].view(-1).shape[0]
    assert not output._teacher_state


def test_distill_output_metrics_keep_auxiliary_keys():
    student = _build_model(scale=1.0)
    batch = _sample_batch()
    pred = student(batch)
    batch["teacher_energy"] = pred[properties.energy].detach() + 1.0

    output = DistillOutput(
        name="energy_distill",
        teacher_model_path=None,
        student_property=properties.energy,
        teacher_property="teacher_energy",
        metrics={
            "mae_pa": PerAtomMAE(
                size_key=properties.n_atoms,
                value_key=properties.energy,
            ),
        },
    )

    metrics = output.calculate_metrics(pred, batch, "train")

    assert metrics["train_energy_distill_mae_pa"] == pytest.approx(0.5)


def test_distill_output_sampled_hessian_rows_match_full_hessian_targets():
    torch.manual_seed(0)
    student = _build_hessian_model(
        scale=1.0,
        hessian_outputs=[properties.energy_hessian_sampled, properties.energy_hessian_sample_indices],
        num_samples=2,
        mask_key="hessian_mask",
    )
    teacher = _build_hessian_model(
        scale=1.0,
        hessian_outputs=[properties.energy_hessian],
    )
    batch = _sample_batch()
    batch["hessian_mask"] = torch.tensor([True, False], dtype=torch.bool)
    pred = student(batch)

    teacher_batch = _sample_batch()
    teacher_batch["hessian_mask"] = batch["hessian_mask"].clone()
    teacher_pred = teacher(teacher_batch)
    batch["teacher_energy_hessian"] = teacher_pred[properties.energy_hessian].detach()

    output = DistillOutput(
        name="hessian_distill",
        teacher_model_path=None,
        student_property=properties.energy_hessian_sampled,
        teacher_property="teacher_energy_hessian",
        sample_index_key=properties.energy_hessian_sample_indices,
        sample_fn=EnergyHessianSample(mask_key="hessian_mask"),
    )

    loss, num_obs = output.calculate_loss(pred, batch, True)

    assert torch.allclose(loss, torch.zeros_like(loss))
    assert num_obs == sum(rows.numel() for rows in pred[properties.energy_hessian_sampled])


def test_distill_output_offline_missing_teacher_label_raises_key_error():
    student = _build_model(scale=1.0)
    batch = _sample_batch()
    pred = student(batch)

    output = DistillOutput(
        name="energy_distill",
        teacher_model_path=None,
        student_property=properties.energy,
        teacher_property="teacher_energy",
    )

    with pytest.raises(KeyError, match="teacher_energy"):
        output.calculate_loss(pred, batch, True)


@pytest.mark.parametrize(
    ("component", "student", "teacher"),
    [
        ("energy_distill", "energy", "teacher_energy"),
        ("forces_distill", "forces", "teacher_forces"),
    ],
)
def test_distill_component_config(component, student, teacher):
    cfg = _resolve_component_with_task(component)
    assert cfg._target_ == "curator.train.model_output.DistillOutput"
    assert cfg.name == component
    assert cfg.student_property == student
    assert cfg.teacher_property == teacher
    assert cfg.teacher_model_path is None
    assert cfg.teacher_cfg is None
    assert cfg.only_train is True
    assert cfg.cache_key == "distill_teacher"


def test_energy_hessian_distill_component_config():
    cfg = _resolve_component_with_task("energy_hessian_distill")
    assert cfg._target_ == "curator.train.model_output.DistillOutput"
    assert cfg.name == "energy_hessian_distill"
    assert cfg.student_property == properties.energy_hessian_sampled
    assert cfg.teacher_property == "teacher_energy_hessian"
    assert cfg.teacher_output_property == properties.energy_hessian
    assert cfg.sample_index_key == properties.energy_hessian_sample_indices
    assert cfg.sample_fn._target_ == "curator.layer._energy_hessian.EnergyHessianSample"
    assert cfg.sample_fn.mask_key is None
    assert cfg.teacher_model_path is None
    assert cfg.teacher_cfg is None
    assert cfg.only_train is True
    assert cfg.cache_key == "distill_teacher"


def test_energy_force_distill_includes_components():
    cfg = OmegaConf.load(os.path.join("curator", "configs", "task", "outputs", "energy_force_distill.yaml"))
    defaults = OmegaConf.to_container(cfg.defaults, resolve=False)
    assert isinstance(defaults, list)
    defaults = [item for item in defaults if isinstance(item, dict)]
    names = {list(item.keys())[0].split("@")[-1] for item in defaults}
    assert "energy" in names
    assert "forces" in names
    assert "energy_distill" in names
    assert "forces_distill" in names


def test_energy_force_hessian_distill_includes_components():
    cfg = OmegaConf.load(os.path.join("curator", "configs", "task", "outputs", "energy_force_hessian_distill.yaml"))
    defaults = OmegaConf.to_container(cfg.defaults, resolve=False)
    assert isinstance(defaults, list)
    defaults = [item for item in defaults if isinstance(item, dict)]
    names = {list(item.keys())[0].split("@")[-1] for item in defaults}
    assert "energy" in names
    assert "forces" in names
    assert "energy_hessian_distill" in names


@pytest.mark.parametrize(
    ("component", "expected"),
    [
        ("energy", 1.0),
        ("forces", 100.0),
        ("energy_distill", 1.0),
        ("energy_hessian_distill", 1.0),
        ("forces_distill", 100.0),
        ("virial", 1.0),
        ("residual_forces", 1.0),
    ],
)
def test_output_component_loss_weight_defaults(component, expected):
    cfg = _resolved_component_container(component)
    assert cfg["loss_weight"] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("component", "loss_weights", "expected"),
    [
        ("energy", {"energy": 2.5}, 2.5),
        ("forces", {"forces": 42.0}, 42.0),
        ("energy_distill", {"energy_distill": 0.25}, 0.25),
        ("energy_hessian_distill", {"energy_hessian_distill": 0.5}, 0.5),
        ("forces_distill", {"forces_distill": 12.0}, 12.0),
        ("virial", {"virial": 3.0}, 3.0),
        ("residual_forces", {"residual_forces": 0.1}, 0.1),
    ],
)
def test_output_component_loss_weight_uses_task_loss_weights(component, loss_weights, expected):
    cfg = _resolved_component_container(component, loss_weights=loss_weights)
    assert cfg["loss_weight"] == pytest.approx(expected)


def test_energy_force_hessian_distill_output_params():
    cfg = OmegaConf.load(
        os.path.join(
            "curator",
            "configs",
            "task",
            "output_params",
            "energy_force_hessian_distill_params.yaml",
        )
    )
    container = OmegaConf.to_container(cfg, resolve=True)
    modules = container["model"]["output_modules"]
    assert modules[2]["_target_"] == "curator.layer.EnergyHessianOutput"
    assert modules[2]["model_outputs"] == [
        properties.energy_hessian_sampled,
        properties.energy_hessian_sample_indices,
    ]
    assert modules[2]["num_samples"] == 4
    assert modules[2]["mask_key"] is None
    assert container["task"]["hessian_num_samples"] == 4
    assert container["task"]["hessian_mask_key"] is None


def test_collect_distill_output_columns_skips_sampling_outputs():
    outputs = OmegaConf.create(
        {
            "energy_distill": {
                "_target_": "curator.train.model_output.DistillOutput",
                "name": "energy_distill",
                "student_property": properties.energy,
                "teacher_property": "teacher_energy",
            },
            "hessian_distill": {
                "_target_": "curator.train.model_output.DistillOutput",
                "name": "hessian_distill",
                "student_property": properties.energy,
                "teacher_property": "teacher_energy",
                "sample_index_key": "energy_sample_indices",
            },
        }
    )

    columns = collect_distill_output_columns(outputs)

    assert columns == {properties.energy: "teacher_energy"}


def test_sampled_hessian_distill_is_treated_as_offline_when_teacher_output_is_full_hessian():
    cfg = OmegaConf.create(
        {
            "_target_": "curator.train.model_output.DistillOutput",
            "name": "energy_hessian_distill",
            "student_property": properties.energy_hessian_sampled,
            "teacher_property": "teacher_energy_hessian",
            "teacher_output_property": properties.energy_hessian,
            "sample_index_key": properties.energy_hessian_sample_indices,
            "sample_fn": {
                "_target_": "curator.layer._energy_hessian.EnergyHessianSample",
            },
        }
    )

    assert is_offline_distill_output(cfg) is True
    assert collect_distill_output_columns(OmegaConf.create({"hessian": cfg})) == {
        properties.energy_hessian: "teacher_energy_hessian"
    }


def test_prepare_teacher_model_for_offline_hessian_distillation_adds_required_outputs():
    model = _build_model(scale=1.0)
    assert properties.energy_hessian not in model.model_outputs

    prepare_teacher_model_for_offline_distillation(
        model,
        {properties.energy_hessian: "teacher_energy_hessian"},
    )

    assert properties.forces in model.model_outputs
    assert properties.energy_hessian in model.model_outputs
    assert any(isinstance(module, EnergyHessianOutput) for module in model.output_modules)
