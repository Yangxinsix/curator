import os
import json

import h5py
import numpy as np
import pytest
import torch
from ase import units

from curator.data import (
    AtomsDataModule,
    SpiceDataset,
    collate_atoms_data,
    ion_pair_distance_split,
    properties,
)


def _write_group(handle, name, subset, atomic_numbers, conformations, charges):
    group = handle.create_group(name)
    string = h5py.string_dtype()
    group.create_dataset("subset", data=[subset], dtype=string)
    group.create_dataset("smiles", data=["[Na+:1].[Cl-:2]"], dtype=string)
    group.create_dataset("atomic_numbers", data=np.asarray(atomic_numbers, dtype=np.int16))
    geometry = group.create_dataset(
        "conformations", data=np.asarray(conformations, dtype=np.float32)
    )
    geometry.attrs["units"] = "bohr"
    n_conformations = len(conformations)
    n_atoms = len(atomic_numbers)
    energy = group.create_dataset(
        "formation_energy",
        data=np.linspace(-0.2, -0.1, n_conformations, dtype=np.float64),
    )
    energy.attrs["units"] = "hartree"
    gradient = group.create_dataset(
        "dft_total_gradient",
        data=np.arange(n_conformations * n_atoms * 3, dtype=np.float32).reshape(
            n_conformations, n_atoms, 3
        )
        / 100.0,
    )
    gradient.attrs["units"] = "hartree/bohr"
    mbis = group.create_dataset(
        "mbis_charges", data=np.asarray(charges, dtype=np.float32)
    )
    mbis.attrs["units"] = "elementary_charge"
    dipole = group.create_dataset(
        "scf_dipole",
        data=np.full((n_conformations, 3), 0.25, dtype=np.float32),
    )
    dipole.attrs["units"] = "elementary_charge*bohr"


@pytest.fixture
def spice_file(tmp_path):
    path = tmp_path / "SPICE-mini.hdf5"
    with h5py.File(path, "w") as handle:
        _write_group(
            handle,
            "NaCl",
            "SPICE Ion Pairs Single Points Dataset v1.2",
            [11, 17],
            [
                [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            ],
            [
                [[0.8], [-0.8]],
                [[0.9], [-0.9]],
            ],
        )
        _write_group(
            handle,
            "peptide",
            "SPICE Dipeptides Single Points Dataset v1.3",
            [1, 6, 8],
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            [[[0.1], [0.2], [-0.3]]],
        )
    return path


def test_spice_subset_reader_converts_units_signs_and_charge_labels(spice_file):
    dataset = SpiceDataset(
        str(spice_file),
        compute_neighbor_list=False,
        subsets=["ion_pairs"],
        default_dtype=torch.float64,
    )

    assert len(dataset) == 2
    assert dataset.get_n_atoms(0) == 2
    sample = dataset[0]

    torch.testing.assert_close(
        sample[properties.positions][1, 0],
        torch.tensor(4.0 * units.Bohr, dtype=torch.float64),
    )
    torch.testing.assert_close(
        sample[properties.energy],
        torch.tensor([-0.2 * units.Hartree], dtype=torch.float64),
    )
    expected_gradient = torch.arange(6, dtype=torch.float64).reshape(2, 3) / 100.0
    torch.testing.assert_close(
        sample[properties.forces],
        -expected_gradient * units.Hartree / units.Bohr,
    )
    torch.testing.assert_close(
        sample[properties.atomic_charge],
        torch.tensor([0.8, -0.8], dtype=torch.float64),
    )
    torch.testing.assert_close(
        sample[properties.total_charge], torch.tensor([0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        sample[properties.dipole],
        torch.full((1, 3), 0.25 * units.Bohr, dtype=torch.float64),
    )
    assert sample.meta["subset"].startswith("SPICE Ion Pairs")


def test_spice_ragged_collation_and_missing_label_masks(spice_file):
    dataset = SpiceDataset(
        str(spice_file),
        compute_neighbor_list=False,
        subsets=["ion_pairs", "dipeptides"],
    )
    batch = collate_atoms_data([dataset[0], dataset[2]])

    torch.testing.assert_close(batch[properties.n_atoms], torch.tensor([2, 3]))
    assert batch[properties.positions].shape == (5, 3)
    assert batch[properties.atomic_charge].shape == (5,)
    assert batch[properties.energy].shape == (2,)
    assert batch["masks"][properties.atomic_charge].tolist() == [True, True]


def test_spice_capability_manifest_prevents_constant_potential_overclaim(spice_file):
    dataset = SpiceDataset(
        str(spice_file),
        compute_neighbor_list=False,
        subsets=["ion_pairs"],
    )

    dataset.capabilities.require_labels(["energy", "forces", "total_charge"])
    with pytest.raises(ValueError, match="cannot validate"):
        dataset.capabilities.require_claims(["constant_electrode_potential"])
    assert "constant_electrode_potential" in dataset.capabilities.cannot_validate


def test_atoms_datamodule_constructs_spice_reader_with_dataset_kwargs(spice_file):
    module = AtomsDataModule(
        batch_size=2,
        data_type="Spice",
        datapath=str(spice_file),
        cutoff=5.0,
        compute_neighbor_list=False,
        dataset_kwargs={"subsets": ["ion_pairs"]},
        num_train=1,
        num_val=1,
        val_only=False,
        num_workers=0,
    )

    dataset = module.setup_dataset("Spice", str(spice_file))

    assert isinstance(dataset, SpiceDataset)
    assert len(dataset) == 2


def test_ion_pair_distance_split_uses_flattened_dataset_indices(tmp_path):
    path = tmp_path / "ion-distances.hdf5"
    distances_angstrom = [4.0, 5.5, 6.0, 6.5, 7.0]
    conformations = [
        [[0.0, 0.0, 0.0], [distance / units.Bohr, 0.0, 0.0]]
        for distance in distances_angstrom
    ]
    charges = [[[0.8], [-0.8]]] * len(conformations)
    with h5py.File(path, "w") as handle:
        _write_group(
            handle,
            "first",
            "SPICE Ion Pairs",
            [11, 17],
            conformations,
            charges,
        )
        _write_group(
            handle,
            "second",
            "SPICE Ion Pairs",
            [11, 17],
            conformations,
            charges,
        )

    dataset = SpiceDataset(
        str(path),
        compute_neighbor_list=False,
        subsets=["ion_pairs"],
        stride=2,
    )
    split = ion_pair_distance_split(dataset, 5.5, 6.5)

    assert split == {
        "train": [0, 3],
        "validation": [1, 4],
        "test": [2, 5],
    }
    assert sorted(split["train"] + split["validation"] + split["test"]) == list(
        range(len(dataset))
    )


def test_ion_pair_distance_split_rejects_non_pair_and_empty_split(spice_file):
    mixed = SpiceDataset(
        str(spice_file),
        compute_neighbor_list=False,
        subsets=["ion_pairs", "dipeptides"],
    )
    with pytest.raises(ValueError, match="exactly two atoms"):
        ion_pair_distance_split(mixed)

    pairs = SpiceDataset(
        str(spice_file), compute_neighbor_list=False, subsets=["ion_pairs"]
    )
    with pytest.raises(ValueError, match="empty"):
        ion_pair_distance_split(pairs, 1.0, 2.0)


def test_distance_split_is_consumed_by_atoms_datamodule(spice_file, tmp_path):
    dataset = SpiceDataset(
        str(spice_file), compute_neighbor_list=False, subsets=["ion_pairs"]
    )
    split = {"train": [0], "validation": [], "test": [1]}
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split))
    module = AtomsDataModule(
        batch_size=1,
        data_type="Spice",
        datapath=str(spice_file),
        cutoff=5.0,
        compute_neighbor_list=False,
        split_file=str(path),
        num_train=0,
        num_val=0,
        num_test=0,
        val_only=False,
        num_workers=0,
    )
    module.dataset = dataset
    module.datalen = len(dataset)
    module._split_data()

    assert module.train_dataset.indices == split["train"]
    assert module.val_dataset.indices == split["validation"]
    assert module.test_dataset.indices == split["test"]


def test_atoms_datamodule_rejects_overlapping_split_indices(spice_file, tmp_path):
    path = tmp_path / "bad-split.json"
    path.write_text(json.dumps({"train": [0], "validation": [0], "test": [1]}))
    module = AtomsDataModule(
        batch_size=1,
        split_file=str(path),
        num_train=0,
        num_val=0,
        num_test=0,
        val_only=False,
    )
    module.dataset = SpiceDataset(
        str(spice_file), compute_neighbor_list=False, subsets=["ion_pairs"]
    )
    module.datalen = len(module.dataset)

    with pytest.raises(ValueError, match="unique and disjoint"):
        module._split_data()


def test_spice_reader_infers_total_charge_without_partition_charges(tmp_path):
    path = tmp_path / "missing-charge.hdf5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("molecule")
        string = h5py.string_dtype()
        group.create_dataset("subset", data=["SPICE Ion Pairs"], dtype=string)
        group.create_dataset("smiles", data=["[Ca+2].[Cl-]"], dtype=string)
        group.create_dataset("atomic_numbers", data=np.asarray([11, 17], dtype=np.int16))
        group.create_dataset(
            "conformations", data=np.zeros((1, 2, 3), dtype=np.float32)
        )
        group.create_dataset("formation_energy", data=np.zeros(1))
        group.create_dataset("dft_total_gradient", data=np.zeros((1, 2, 3)))

    dataset = SpiceDataset(
        str(path),
        compute_neighbor_list=False,
        subsets=["ion_pairs"],
        require_total_charge=True,
    )
    sample = dataset[0]

    torch.testing.assert_close(
        sample[properties.total_charge], torch.tensor([1.0])
    )
    assert properties.atomic_charge not in sample
    assert "total_charge" in dataset.capabilities.labels
    assert "atomic_charge_mbis" not in dataset.capabilities.labels
    assert "partition_charge_auxiliary" not in dataset.capabilities.can_validate


def test_spice_reader_rejects_explicit_incompatible_units(spice_file):
    with h5py.File(spice_file, "r+") as handle:
        handle["NaCl/conformations"].attrs["units"] = "angstrom"

    with pytest.raises(ValueError, match="unsupported units"):
        SpiceDataset(
            str(spice_file),
            compute_neighbor_list=False,
            subsets=["ion_pairs"],
        )


@pytest.mark.skipif(
    "CURATOR_SPICE_ION_PAIRS" not in os.environ,
    reason="official SPICE ion-pair artifact was not provided",
)
def test_official_spice_ion_pair_artifact_smoke():
    dataset = SpiceDataset(
        os.environ["CURATOR_SPICE_ION_PAIRS"],
        compute_neighbor_list=False,
        subsets=["ion_pairs"],
        default_dtype=torch.float64,
    )

    assert len(dataset._group_names) == 28
    assert len(dataset) >= 1400
    assert set(dataset._atom_counts.tolist()) == {2}
    assert "atomic_charge_mbis" not in dataset.capabilities.labels
    for index in (0, len(dataset) // 2, len(dataset) - 1):
        sample = dataset[index]
        assert sample[properties.positions].shape == (2, 3)
        assert sample[properties.forces].shape == (2, 3)
        assert float(sample[properties.total_charge].item()).is_integer()
        assert properties.atomic_charge not in sample


@pytest.mark.skipif(
    "CURATOR_SPICE_ION_PAIRS" not in os.environ,
    reason="official SPICE ion-pair artifact was not provided",
)
def test_official_spice_ion_pair_distance_extrapolation_split():
    dataset = SpiceDataset(
        os.environ["CURATOR_SPICE_ION_PAIRS"],
        compute_neighbor_list=False,
        subsets=["ion_pairs"],
    )
    split = ion_pair_distance_split(dataset, 5.5, 6.5)

    assert {name: len(indices) for name, indices in split.items()} == {
        "train": 868,
        "validation": 280,
        "test": 280,
    }
    all_indices = split["train"] + split["validation"] + split["test"]
    assert len(all_indices) == len(set(all_indices)) == len(dataset)
    assert sorted(all_indices) == list(range(len(dataset)))
