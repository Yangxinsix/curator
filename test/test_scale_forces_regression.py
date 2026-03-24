import unittest
from pathlib import Path

from curator.data.datamodule import build_datamodule


class ScaleForcesRegressionTest(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_energy_scale_uses_forces_rms_when_enabled(self):
        traj_path = str(self.repo_root / "example" / "LiFePO4.traj")
        datamodule = build_datamodule(
            datapath=traj_path,
            data_type="Ase",
            batch_size=4,
            compute_neighbor_list=False,
            cutoff=5.0,
            num_train=0.5,
            num_val=0.5,
            num_test=0,
            train_val_split="random",
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            species="auto",
            avg_num_neighbors="auto",
            atomic_energies=None,
            normalization=True,
            atomwise_normalization=True,
            scale_forces=True,
            transforms=[],
        )
        datamodule.setup()

        _, energy_scale = datamodule._get_scale_shift()
        forces_rms = datamodule._get_rms("forces")

        self.assertAlmostEqual(energy_scale, forces_rms, places=6)


if __name__ == "__main__":
    unittest.main()
