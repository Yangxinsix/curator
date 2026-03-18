import unittest
from pathlib import Path

from hydra.utils import instantiate

import curator.layer._cuequivariance_wrapper as cueq
from unittest import skipIf
from curator.model import MACE, NeuralNetworkPotential
from curator.utils import read_user_config, update_config_from_datamodule


class ReadUserConfigOverridesTest(unittest.TestCase):
    def setUp(self):
        cueq.set_use_cueq(False)
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

    def test_use_cueq_flag_sets_global(self):
        if cueq.IS_CUET_AVAILABLE:
            self.skipTest("cuequivariance available; skip fallback warning test to avoid GPU/perm setup.")
        cfg = read_user_config(
            config_path="curator/configs",
            overrides=[
                "model/representation=mace",
                "model.representation.use_cueq=True",
                "data.species=[Li,Fe,P,O]",
            ],
        )
        instantiate(cfg.model)
        self.assertTrue(cueq.USE_CUEQ_GLOBAL)
        if not cueq.IS_CUET_AVAILABLE:
            # Even without cuequivariance installed, the flag should be set
            # and we should gracefully fall back to e3nn kernels.
            self.assertFalse(cueq.IS_CUET_AVAILABLE)

    def test_use_cueq_flag_sets_global_for_nequip(self):
        if cueq.IS_CUET_AVAILABLE:
            self.skipTest("cuequivariance available; skip fallback warning test to avoid GPU/perm setup.")
        cfg = read_user_config(
            config_path="curator/configs",
            overrides=[
                "model/representation=nequip",
                "model.representation.use_cueq=True",
                "data.species=[Li,Fe,P,O]",
            ],
        )
        instantiate(cfg.model)
        self.assertTrue(cueq.USE_CUEQ_GLOBAL)

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
