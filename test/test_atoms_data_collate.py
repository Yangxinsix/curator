import unittest

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from curator.data import properties
from curator.data.atoms_data import AtomsData
from curator.data.collate_atoms_data import collate_atoms_data


class DummyDataset(Dataset):
    def __init__(self, targets_fn, task: str):
        self.targets_fn = targets_fn
        self.task = task

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        n_atoms = 2
        atoms = {
            properties.Z: torch.tensor([1, 1], dtype=torch.long),
            properties.positions: torch.zeros((n_atoms, 3)),
            properties.n_atoms: torch.tensor([n_atoms], dtype=torch.long),
            properties.image_idx: torch.zeros((n_atoms,), dtype=torch.long),
        }
        targets = self.targets_fn(n_atoms)
        return AtomsData(atoms=atoms, targets=targets, task=self.task)


class AtomsDataCollateTest(unittest.TestCase):
    def test_masks_with_mixed_targets(self):
        ds_energy_forces = DummyDataset(
            lambda n: {
                properties.energy: torch.tensor([1.0]),
                properties.forces: torch.zeros((n, 3)),
            },
            task="energy_forces",
        )
        ds_stress = DummyDataset(
            lambda _: {properties.stress: torch.zeros((1, 6))},
            task="stress",
        )

        dataset = ConcatDataset([ds_energy_forces, ds_stress])
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_atoms_data)
        batch = next(iter(loader))

        self.assertIsInstance(ds_energy_forces[0], AtomsData)
        self.assertIsInstance(ds_stress[0], AtomsData)

        self.assertEqual(batch["masks"][properties.energy].tolist(), [True, False])
        self.assertEqual(batch["masks"][properties.forces].tolist(), [True, False])
        self.assertEqual(batch["masks"][properties.stress].tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
