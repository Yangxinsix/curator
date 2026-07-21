from types import SimpleNamespace

import torch

from curator.data import properties
from curator.data.datamodule import DataContext
from curator.layer import GlobalRescaleShift
from curator.model import LitNNP


def _scale_forces_layer(scale=1.25):
    context = DataContext(
        head_scale_shift={
            properties.energy: {"mean": 0.0, "std": scale},
            properties.forces: {"mean": 0.0, "std": scale},
        }
    )
    datamodule = SimpleNamespace(
        scale_forces=True,
        rescale_shift_heads=[],
        build_context=lambda heads: context,
    )
    layer = GlobalRescaleShift(heads=["energy"])
    layer.setup_from_datamodule(datamodule)
    return layer


def test_scale_forces_adds_force_head_with_energy_scale():
    layer = _scale_forces_layer(scale=2.5)
    layer.eval()
    original = {
        properties.atomic_energy: torch.tensor([1.0, 2.0]),
        properties.image_idx: torch.tensor([0, 0]),
        properties.n_atoms: torch.tensor([2]),
        properties.forces: torch.ones(2, 3),
    }

    scaled = layer.scale(original)

    assert [head.key for head in layer.heads] == [properties.energy, properties.forces]
    assert torch.equal(layer.scales[0].scale, layer.scales[1].scale)
    assert torch.allclose(scaled[properties.energy], torch.tensor([7.5]))
    assert torch.allclose(scaled[properties.forces], original[properties.forces] * 2.5)

    restored = layer.unscale(scaled, force_process=True)
    for key, value in original.items():
        if key != properties.n_atoms:
            assert torch.allclose(restored[key], value)


def test_force_target_is_normalized_by_energy_scale():
    layer = _scale_forces_layer(scale=1.25)
    physical = torch.tensor([[1.0, -2.0, 3.0]])

    normalized = layer.unscale({properties.forces: physical}, force_process=True)

    assert torch.allclose(normalized[properties.forces], physical / 1.25)
    assert torch.allclose(
        layer.scale(normalized, force_process=True)[properties.forces],
        physical,
    )


def test_multi_domain_validation_total_sums_already_weighted_losses():
    datamodule = SimpleNamespace(
        domain_modules={
            "replay": SimpleNamespace(val_dataset=range(1000)),
            "lifepo4": SimpleNamespace(val_dataset=range(225)),
        },
        domain_to_id={"replay": 0, "lifepo4": 1},
    )
    metrics = {
        "val_total_loss_epoch/dataloader_idx_0": torch.tensor(2.0),
        "val_total_loss_epoch/dataloader_idx_1": torch.tensor(4.0),
    }

    total = LitNNP._multi_domain_validation_total(metrics, datamodule)

    assert total == 6.0
