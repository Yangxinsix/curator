from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from curator.config_utils import read_user_config


CONFIG_DIR = str((Path(__file__).parents[1] / "curator" / "configs").resolve())
DIRECT_FORCE_OVERRIDE = "model/force_output@model.output_modules.force_output=direct"
HESSIAN_OUTPUTS = (
    "energy_force_hessian_distill",
    "energy_force_projected_hessian_distill",
    "energy_force_teacher_hessian_distill",
    "energy_force_teacher_projected_hessian_distill",
    "energy_force_teacher_dynamic_projected_hessian_distill",
)


def _compose(*overrides):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="train", overrides=list(overrides))


def _output_targets(config):
    return [str(module._target_) for module in config.model.output_modules]


def test_force_output_override_replaces_the_stable_slot():
    gradient = _compose()
    direct = _compose(DIRECT_FORCE_OVERRIDE)

    assert list(gradient.model.output_modules) == ["force_output", "global_rescale_shift"]
    assert gradient.model.output_modules.force_output._target_ == "curator.layer.GradientOutput"
    assert list(direct.model.output_modules) == ["force_output", "global_rescale_shift"]
    assert direct.model.output_modules.force_output._target_ == "curator.layer.DirectForceOutput"


def test_output_modules_default_composes_without_a_duplicate_package():
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        config = compose(config_name="model/output_modules/default")

    assert list(config.model.output_modules) == ["force_output", "global_rescale_shift"]


def test_qeq_config_declares_generic_physical_output_composition():
    config = _compose("model=qeq", "task/outputs=energy_force_residual")
    rescale = config.model.output_modules[-1]

    assert list(rescale.physical_contributions) == [
        {
            "source": "qeq_energy",
            "destination": "energy",
            "scale_like": "energy",
        },
        {
            "source": "ewald_forces",
            "destination": "forces",
            "scale_like": "forces",
        },
    ]
    assert list(rescale.normalized_copies) == [
        {
            "source": "chemical_potential_residual",
            "destination": "chemical_potential_residual_normalized",
            "scale_like": "energy",
        }
    ]
    assert (
        config.task.outputs.chemical_potential_residual.name
        == "chemical_potential_residual_normalized"
    )


@pytest.mark.parametrize("outputs", HESSIAN_OUTPUTS)
def test_hessian_output_params_do_not_select_a_force_implementation(outputs):
    config = _compose(f"task/outputs={outputs}", DIRECT_FORCE_OVERRIDE)

    assert list(config.model.output_modules) == [
        "force_output",
        "global_rescale_shift",
        "energy_hessian_output",
    ]
    assert config.model.output_modules.force_output._target_ == "curator.layer.DirectForceOutput"
    assert config.model.output_modules.energy_hessian_output._target_ == "curator.layer.EnergyHessianOutput"
    assert "gradient_output" not in config.model.output_modules
    assert config.model.input_modules.pairwise_distance.compute_distance_from_R is True
    assert config.model.input_modules.pairwise_distance.compute_forces is True


def test_read_user_config_accepts_force_output_selection_in_defaults():
    user_config = OmegaConf.create(
        {
            "defaults": [
                {"task/outputs": "energy_force_hessian_distill"},
                {
                    "model/force_output@model.output_modules.force_output": "direct",
                },
            ],
        }
    )

    config = read_user_config(user_config, config_path=CONFIG_DIR, config_name="train")

    assert _output_targets(config) == [
        "curator.layer.DirectForceOutput",
        "curator.layer.GlobalRescaleShift",
        "curator.layer.EnergyHessianOutput",
    ]


def test_read_user_config_preserves_legacy_list_style_output_modules():
    user_config = OmegaConf.create(
        {
            "model": {
                "output_modules": [
                    {"_target_": "curator.layer.GradientOutput"},
                    {"_target_": "curator.layer.GlobalRescaleShift"},
                ],
            },
        }
    )

    config = read_user_config(user_config, config_path=CONFIG_DIR, config_name="train")

    assert _output_targets(config) == [
        "curator.layer.GradientOutput",
        "curator.layer.GlobalRescaleShift",
    ]


def test_read_user_config_migrates_legacy_dict_force_slot():
    user_config = OmegaConf.create(
        {
            "model": {
                "output_modules": {
                    "gradient_output": {
                        "_target_": "curator.layer.DirectForceOutput",
                    }
                }
            }
        }
    )

    config = read_user_config(user_config, config_path=CONFIG_DIR, config_name="train")

    assert _output_targets(config) == [
        "curator.layer.DirectForceOutput",
        "curator.layer.GlobalRescaleShift",
    ]


@pytest.mark.parametrize(
    "outputs",
    [
        "energy_force_virial",
        "energy_force_virial_pa",
        "energy_force_virial_per_species",
    ],
)
def test_virial_output_params_use_stable_dict_slots(outputs):
    config = _compose(f"task/outputs={outputs}")

    assert "force_output" in config.model.output_modules
    assert config.model.output_modules.force_output._target_ == "curator.layer.GradientOutput"
