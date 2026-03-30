import unittest
from pathlib import Path

from hydra.utils import instantiate

from curator.layer.wrappers import apply_wrappers, export_wrapper_config, set_wrapper_config
from curator.model import MACE, NeuralNetworkPotential
from curator.utils import read_user_config, update_config_from_datamodule


class ReadUserConfigOverridesTest(unittest.TestCase):
    def setUp(self):
        set_wrapper_config(use_cueq=False, use_elora=False, wrapper_stack=None)
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_can_override_representation_via_hydra_style(self):
        cfg = read_user_config(
            config_path="curator/configs",
            overrides=[
                "model/representation=mace",
                "data.species=[Li,Fe,P,O]",
            ],
        )
        self.assertEqual(cfg.model.representation._target_, "curator.model.MACE")

        model = instantiate(cfg.model)
        self.assertIsInstance(model, NeuralNetworkPotential)
        self.assertIsInstance(model.representation, MACE)

    def test_addon_config_is_kept_outside_representation_schema(self):
        cfg = read_user_config(
            config_path="curator/configs",
            overrides=[
                "model/representation=mace",
                "+addon.wrapper_stack=elora",
                "+addon.elora_rank=8",
                "model.representation.num_features=4",
                "data.species=[Li,Fe,P,O]",
            ],
        )

        self.assertEqual(cfg.addon.wrapper_stack, "elora")
        self.assertNotIn("use_elora", cfg.model.representation)
        self.assertNotIn("wrapper_stack", cfg.model.representation)

        model = instantiate(cfg.model)
        patched = apply_wrappers(model, cfg.addon)
        metadata = export_wrapper_config(patched)

        self.assertTrue(metadata["use_elora"])
        self.assertFalse(metadata["use_cueq"])
        self.assertEqual(metadata["wrapper_stack"], "elora")

    def test_cueq_elora_addon_combines_flags(self):
        cfg = read_user_config(
            config_path="curator/configs",
            overrides=[
                "model/representation=mace",
                "+addon.wrapper_stack=cueq+elora",
                "model.representation.num_features=4",
                "data.species=[Li,Fe,P,O]",
            ],
        )

        self.assertEqual(cfg.addon.wrapper_stack, "cueq+elora")
        model = instantiate(cfg.model)
        patched = apply_wrappers(model, cfg.addon)
        metadata = export_wrapper_config(patched)

        self.assertTrue(metadata["use_cueq"])
        self.assertTrue(metadata["use_elora"])
        self.assertEqual(metadata["wrapper_stack"], "cueq+elora")

    def test_can_override_data_field(self):
        custom_path = "/tmp/path_to_dataset"
        cfg = read_user_config(
            config_path="curator/configs",
            overrides=[f"data.datapath={custom_path}"],
        )
        self.assertEqual(cfg.data.datapath, custom_path)

    def test_qeq_readout_heads_instantiates(self):
        cfg_path = self.repo_root / "test" / "qeq" / "qeq.yaml"
        cfg = read_user_config(cfg_path, config_path="curator/configs", config_name="train.yaml")
        model = instantiate(cfg.model)
        self.assertIsInstance(model, NeuralNetworkPotential)
        self.assertEqual(model.representation.readout.model_outputs, ["energy", "atomic_charge"])
        self.assertEqual(model.representation.readout.aggregation_modes, ["sum", "none"])

    def test_datamodule_update_keeps_explicit_readout_heads(self):
        cfg_path = self.repo_root / "test" / "qeq" / "qeq.yaml"
        cfg = read_user_config(cfg_path, config_path="curator/configs", config_name="train.yaml")
        cfg.data.datapath = str(self.repo_root / "example" / "LiFePO4.traj")
        cfg.data.transforms = []
        datamodule = instantiate(cfg.data)
        datamodule.setup()

        update_config_from_datamodule(cfg, datamodule)

        heads = cfg.model.representation.readout.heads
        self.assertEqual(heads[0], "energy")
        self.assertEqual(heads[1].key, "atomic_charge")
        self.assertIsNone(heads[1].reduction)


if __name__ == "__main__":
    unittest.main()
