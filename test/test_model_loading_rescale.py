from types import SimpleNamespace

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
