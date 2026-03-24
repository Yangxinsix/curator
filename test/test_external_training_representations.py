from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from curator.data import properties
from curator.layer import GradientOutput
from curator.model import AllegroRepresentation, ESENRepresentation, LitNNP, NeuralNetworkPotential
from curator.train import FreezeSchedule
from curator.train.model_output import ModelOutput


def _sample_batch():
    edge_diff = torch.tensor(
        [[0.0, 0.0, 0.9], [0.0, 0.0, -0.9]],
        dtype=torch.float32,
        requires_grad=True,
    )
    return {
        properties.n_atoms: torch.tensor([2], dtype=torch.long),
        properties.atomic_numbers: torch.tensor([1, 8], dtype=torch.long),
        properties.positions: torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]],
            dtype=torch.float32,
        ),
        properties.cell: torch.eye(3, dtype=torch.float32).view(1, 3, 3),
        properties.pbc: torch.tensor([[False, False, False]], dtype=torch.bool),
        properties.edge_idx: torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        properties.edge_diff: edge_diff,
        properties.edge_dist: torch.linalg.norm(edge_diff.detach(), dim=-1),
        properties.cell_displacements: torch.zeros((2, 3), dtype=torch.float32),
    }


class DummyAllegroBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.type_names = ["H", "O"]
        self.edge_norm = type("EdgeNorm", (), {"r_max": 5.0})()
        self.edge_feature_block = nn.Linear(3, 6)

    def forward(self, batch):
        edge_features = self.edge_feature_block(batch["edge_vectors"])
        batch["edge_features"] = edge_features
        return batch


class DummyESENBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.cutoff = 6.0
        self.pre_block = nn.Linear(3, 6)
        self.energy_block = nn.Linear(6, 1)

    def forward(self, batch):
        edge_index = batch["edge_index"]
        edge_vectors = batch["edge_vectors"]
        num_nodes = int(batch["num_nodes"].view(-1)[0].item())
        node_seed = torch.zeros((num_nodes, 3), dtype=edge_vectors.dtype, device=edge_vectors.device)
        node_seed.index_add_(0, edge_index[0], edge_vectors)
        node_features = self.pre_block(node_seed)
        batch["node_features"] = node_features
        batch["energy_logits"] = self.energy_block(node_features)
        return batch


def _representation_cfg(target: str, feature_dim: int = 8):
    return OmegaConf.create(
        {
            "_target_": target,
            "feature_dim": feature_dim,
        }
    )


def _build_task(model: NeuralNetworkPotential) -> LitNNP:
    task = LitNNP(
        model=model,
        outputs=[
            ModelOutput(properties.energy, loss_fn=nn.MSELoss()),
            ModelOutput(properties.forces, loss_fn=nn.MSELoss()),
            ModelOutput(properties.stress, loss_fn=nn.MSELoss()),
        ],
        optimizer=torch.optim.Adam,
    )
    task.rescale_layers = []
    return task


def test_hydra_instantiates_trainable_external_representations():
    allegro = instantiate(
        _representation_cfg("curator.model.AllegroRepresentation"),
        backbone=DummyAllegroBackbone(),
        species=["H", "O"],
    )
    esen = instantiate(
        _representation_cfg("curator.model.ESENRepresentation"),
        backbone=DummyESENBackbone(),
    )
    assert isinstance(allegro, AllegroRepresentation)
    assert isinstance(esen, ESENRepresentation)


@pytest.mark.parametrize(
    ("representation", "kwargs"),
    [
        (AllegroRepresentation, {"backbone": DummyAllegroBackbone(), "species": ["H", "O"]}),
        (ESENRepresentation, {"backbone": DummyESENBackbone()}),
    ],
)
def test_external_representations_run_full_training_step(representation, kwargs):
    rep = representation(feature_dim=8, **kwargs)
    rep_batch = _sample_batch()
    rep_out = rep(rep_batch)
    assert properties.node_feat in rep_out
    assert properties.node_embedding in rep_out
    assert properties.atomic_energy in rep_out
    assert properties.energy in rep_out

    model = NeuralNetworkPotential(
        representation=rep,
        output_modules=[GradientOutput(model_outputs=[properties.forces, properties.stress])],
    )
    batch = _sample_batch()
    pred = model(batch)
    assert properties.energy in pred
    assert properties.forces in pred
    assert properties.stress in pred

    target_batch = _sample_batch()
    target_batch[properties.energy] = pred[properties.energy].detach()
    target_batch[properties.forces] = pred[properties.forces].detach()
    target_batch[properties.stress] = pred[properties.stress].detach()

    task = _build_task(model)
    loss_dict, _, _ = task._train_step_single(target_batch)
    loss = loss_dict["train_total_loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in rep.projection.parameters())
    assert any(parameter.grad is not None for parameter in rep.readout.parameters())


def test_external_representations_expose_named_groups():
    rep = AllegroRepresentation(
        backbone=DummyAllegroBackbone(),
        species=["H", "O"],
        feature_dim=8,
    )
    module_groups = rep.module_groups()
    assert set(module_groups) == {"backbone", "projection", "readout"}

    parameter_groups = rep.parameter_groups()
    assert [group.name for group in parameter_groups] == ["backbone", "projection", "readout"]
    grouped = {group.name: group.params for group in parameter_groups}
    assert grouped["backbone"]
    assert all(parameter.requires_grad for parameter in grouped["backbone"])
    assert all(parameter.requires_grad for parameter in grouped["readout"])


def test_litnnp_configures_differential_optimizer_groups_for_external_backbone():
    rep = AllegroRepresentation(
        backbone=DummyAllegroBackbone(),
        species=["H", "O"],
        feature_dim=8,
    )
    model = NeuralNetworkPotential(representation=rep)
    task = LitNNP(
        model=model,
        outputs=[ModelOutput(properties.energy, loss_fn=nn.MSELoss())],
        optimizer=partial(torch.optim.AdamW, lr=1e-3, weight_decay=1e-2),
        optimizer_groups={
            "backbone": {"lr": 1e-4},
            "projection": {"lr": 5e-4},
            "readout": {"lr": 1e-3},
        },
    )

    optimizer = task.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW)

    grouped = {group["name"]: group for group in optimizer.param_groups}
    assert set(grouped) == {"backbone", "projection", "readout"}
    assert grouped["backbone"]["lr"] == pytest.approx(1e-4)
    assert grouped["projection"]["lr"] == pytest.approx(5e-4)
    assert grouped["readout"]["lr"] == pytest.approx(1e-3)

    seen = set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            assert id(param) not in seen
            seen.add(id(param))
    total_trainable = sum(1 for param in model.parameters() if param.requires_grad)
    assert len(seen) == total_trainable


def test_freeze_schedule_controls_named_groups_and_learning_rates(caplog):
    rep = AllegroRepresentation(
        backbone=DummyAllegroBackbone(),
        species=["H", "O"],
        feature_dim=8,
    )
    model = NeuralNetworkPotential(representation=rep)
    task = LitNNP(
        model=model,
        outputs=[ModelOutput(properties.energy, loss_fn=nn.MSELoss())],
        optimizer=partial(torch.optim.AdamW, lr=1e-3),
    )
    optimizer = task.configure_optimizers()
    trainer = SimpleNamespace(current_epoch=0, optimizers=[optimizer])

    callback = FreezeSchedule(
        stages=[
            {"start_epoch": 0, "freeze": ["backbone"], "lr": {"readout": 2e-3}},
            {"start_epoch": 1, "unfreeze": ["backbone"], "lr": {"backbone": 1e-4}},
        ],
    )
    with caplog.at_level(logging.INFO, logger="curator.train.callbacks"):
        callback.on_fit_start(trainer, task)

    grouped = {group["name"]: group for group in optimizer.param_groups}
    assert grouped["readout"]["lr"] == pytest.approx(2e-3)
    assert all(not parameter.requires_grad for parameter in rep.backbone.parameters())
    assert all(parameter.requires_grad for parameter in rep.readout.parameters())
    fit_messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "Applying freeze schedule stage at epoch 0" in fit_messages
    assert "Trainability updates:" in fit_messages
    assert "Learning-rate updates:" in fit_messages
    assert "backbone" in fit_messages
    assert "readout" in fit_messages
    assert "updated_params" in fit_messages
    assert "updated_scalars" in fit_messages

    caplog.clear()
    trainer.current_epoch = 1
    with caplog.at_level(logging.INFO, logger="curator.train.callbacks"):
        callback.on_train_epoch_start(trainer, task)

    assert grouped["backbone"]["lr"] == pytest.approx(1e-4)
    assert all(parameter.requires_grad for parameter in rep.backbone.parameters())
    epoch_messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "Applying freeze schedule stage at epoch 1" in epoch_messages
    assert "Trainability updates:" in epoch_messages
    assert "Learning-rate updates:" in epoch_messages
    assert "backbone" in epoch_messages


def test_optimizer_groups_raise_on_unknown_group_name():
    rep = AllegroRepresentation(
        backbone=DummyAllegroBackbone(),
        species=["H", "O"],
        feature_dim=8,
    )
    model = NeuralNetworkPotential(representation=rep)
    task = LitNNP(
        model=model,
        outputs=[ModelOutput(properties.energy, loss_fn=nn.MSELoss())],
        optimizer=partial(torch.optim.AdamW, lr=1e-3),
        optimizer_groups={"unknown_group": {"lr": 1e-4}},
    )

    with pytest.raises(KeyError, match="unknown_group"):
        task.configure_optimizers()


def test_freeze_schedule_raises_on_unknown_group_name():
    rep = AllegroRepresentation(
        backbone=DummyAllegroBackbone(),
        species=["H", "O"],
        feature_dim=8,
    )
    model = NeuralNetworkPotential(representation=rep)
    task = LitNNP(
        model=model,
        outputs=[ModelOutput(properties.energy, loss_fn=nn.MSELoss())],
        optimizer=partial(torch.optim.AdamW, lr=1e-3),
    )
    optimizer = task.configure_optimizers()
    trainer = SimpleNamespace(current_epoch=0, optimizers=[optimizer])
    callback = FreezeSchedule(stages=[{"start_epoch": 0, "freeze": ["unknown_group"]}])

    with pytest.raises(KeyError, match="unknown_group"):
        callback.on_fit_start(trainer, task)


def test_external_representations_load_prefixed_backbone_checkpoints(tmp_path: Path):
    source = DummyAllegroBackbone()
    checkpoint_path = tmp_path / "allegro_backbone.ckpt"
    state_dict = {
        f"model.representation.backbone.{key}": value.clone()
        for key, value in source.state_dict().items()
    }
    state_dict["model.representation.backbone.unused"] = torch.ones(1)
    torch.save({"state_dict": state_dict}, checkpoint_path)

    rep = AllegroRepresentation(
        backbone=DummyAllegroBackbone(),
        species=["H", "O"],
        feature_dim=8,
        pretrained_path=str(checkpoint_path),
        strict_load=False,
    )
    for key, value in source.state_dict().items():
        assert torch.allclose(rep.backbone.state_dict()[key], value)


def test_missing_external_feature_layer_fails_fast():
    with pytest.raises(ValueError, match="Cannot find feature layer"):
        ESENRepresentation(
            backbone=DummyESENBackbone(),
            feature_dim=8,
            feature_layer="missing_block",
        )
