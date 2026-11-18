from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from curator.commands.train import LoadedModel, _load_model, _prepare_model
from curator.data import properties
from curator.data.datamodule import DataContext
from curator.data.properties import HeadConfig
from curator.layer import EnergyHessianOutput, GlobalRescaleShift, GradientOutput, PairwiseDistance
from curator.model import NeuralNetworkPotential


def _datamodule(scale):
    context = DataContext(
        head_scale_shift={
            properties.energy: {"mean": 0.0, "std": scale},
            properties.forces: {"mean": 0.0, "std": scale},
        }
    )
    return SimpleNamespace(
        scale_forces=True,
        rescale_shift_heads=[],
        build_context=lambda heads: context,
    )


def _model(scale):
    return NeuralNetworkPotential(
        representation=torch.nn.Identity(),
        output_modules=[
            GlobalRescaleShift(
                heads=[HeadConfig(key=properties.energy, scale_by=scale)]
            )
        ],
    )


def test_loaded_model_preserves_checkpoint_rescale_state():
    model = _model(7.0)
    prepared = _prepare_model(
        LoadedModel(model=model, initialize_from_datamodule=False),
        config=OmegaConf.create({"wrapper": None, "compile": False}),
        datamodule=_datamodule(99.0),
    )

    assert prepared._initialized
    assert prepared.output_modules[0].scale_by == [7.0]


def test_new_model_initializes_rescale_from_datamodule():
    model = _model(7.0)
    prepared = _prepare_model(
        LoadedModel(model=model),
        config=OmegaConf.create({"wrapper": None, "compile": False}),
        datamodule=_datamodule(3.0),
    )

    assert prepared.output_modules[0].scale_by == [3.0, 3.0]


def test_resume_preserves_checkpoint_initialization(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"model": _model(7.0)}, checkpoint)
    config = OmegaConf.create(
        {
            "model_path": {"path": str(checkpoint), "mode": "resume"},
            "model": {"_target_": "torch.nn.Identity"},
        }
    )

    loaded = _load_model(config)

    assert loaded.resume_from == checkpoint
    assert not loaded.initialize_from_datamodule


@pytest.mark.parametrize("mode", ["resume", "model"])
def test_cuda_checkpoint_metadata_can_load_without_cuda(tmp_path, monkeypatch, mode):
    checkpoint = tmp_path / "cuda_model.ckpt"
    model = _model(7.0)
    model.register_buffer("device_probe", torch.tensor([3.0]))
    # Write a genuine CUDA storage tag without requiring a GPU for this test.
    # The default deserializer is restored before loading the checkpoint.
    with monkeypatch.context() as serialization:
        serialization.setattr(
            torch.serialization, "_package_registry", list(torch.serialization._package_registry)
        )
        torch.serialization.register_package(0, lambda storage: "cuda:0", lambda storage, location: None)
        torch.save({"model": model}, checkpoint)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = OmegaConf.create({
        "model_path": {"path": str(checkpoint), "mode": mode},
        "model": {"_target_": "torch.nn.Identity"},
    })

    loaded = _load_model(config)

    if mode == "resume":
        assert loaded.resume_from == checkpoint
    else:
        assert loaded.model.device_probe.device.type == "cpu"
        torch.testing.assert_close(loaded.model.device_probe, torch.tensor([3.0]))


def test_loaded_model_finds_hessian_config_after_outputs_become_a_list(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    torch.save(
        {
            "model": NeuralNetworkPotential(
                representation=torch.nn.Identity(),
                input_modules=[PairwiseDistance()],
                output_modules=[GradientOutput()],
            )
        },
        checkpoint,
    )
    config = OmegaConf.create(
        {
            "model_path": {"path": str(checkpoint), "mode": "model"},
            "model": {
                "output_modules": [
                    {
                        "_target_": "curator.layer.EnergyHessianOutput",
                        "vectorize": False,
                    }
                ]
            },
        }
    )

    loaded = _load_model(config)

    assert not loaded.initialize_from_datamodule
    assert [type(module) for module in loaded.model.output_modules] == [
        GradientOutput,
        EnergyHessianOutput,
    ]
    assert loaded.model.input_modules[0].compute_distance_from_R
    assert loaded.model.input_modules[0].compute_forces
