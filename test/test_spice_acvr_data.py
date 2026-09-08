"""SPICE manifests, physical label conversion, split isolation, and replay."""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch
from ase import units

from curator.data import properties
from curator.data.spice_acvr import SpiceACVRDataModule, SpiceACVRDataset


@pytest.fixture
def source(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "SPICE.hdf5"
    distances = np.array([3.0, 4.0, 5.0, 5.49, 6.0, 6.49, 7.0, 8.0])
    with h5py.File(path, "w") as handle:
        for name, numbers, smiles in (
            ("neutral", [11, 9], "[Na+:1].[F-:2]"),
            ("charged", [11, 11], "[Na+:1].[Na+:2]"),
        ):
            group = handle.create_group(name)
            group["subset"] = [b"SPICE Ion Pairs Single Points Dataset v1.2"]
            group["smiles"] = [smiles.encode()]
            group["atomic_numbers"] = numbers
            positions = np.zeros((len(distances), 2, 3))
            positions[:, 0, 0] = 0.25
            positions[:, 1, 0] = distances + 0.25
            group["conformations"] = positions / units.Bohr
            group["conformations"].attrs["units"] = "bohr"
            group["formation_energy"] = np.arange(len(distances)) / units.Hartree
            group["formation_energy"].attrs["units"] = "hartree"
            group["dft_total_gradient"] = np.ones_like(positions) * units.Bohr / units.Hartree
            group["dft_total_gradient"].attrs["units"] = "hartree/bohr"
            dipoles = np.zeros((len(distances), 3))
            dipoles[:, 0] = distances
            group["scf_dipole"] = dipoles / units.Bohr
            group["scf_dipole"].attrs["units"] = "elementary_charge*bohr"
            group["scf_dipole"].attrs["origin"] = [0.0, 0.0, 0.0]
    return path


def _order(loader):
    return [meta["source_index"] for batch in loader for meta in batch["meta"]]


def test_physical_inputs_labels_origin_metadata_and_json_manifest(source):
    data = SpiceACVRDataModule(source, batch_size=3, shuffle=False)
    data.setup()
    assert [len(data.datasets[split]) for split in ("train", "val", "test")] == [8, 4, 4]
    assert all(dataset.metadata == data.metadata for dataset in data.datasets.values())
    assert data.metadata["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert json.loads(json.dumps(data.metadata)) == data.metadata
    assert data.metadata["species"] == ["F", "Na"]
    assert data.metadata["units"]["dipole"] == "e*Angstrom"
    assert not data.metadata["split_rule"]["seed_dependent"]

    sample = data.train_dataset[0]
    assert set(sample.targets) == {properties.energy, properties.forces, properties.dipole}
    assert sample.atoms[properties.total_charge].item() == 2.0
    assert sample[properties.positions][0, 0].item() == pytest.approx(0.25)
    assert sample[properties.positions][1, 0].item() == pytest.approx(3.25)
    torch.testing.assert_close(sample[properties.forces], -torch.ones(2, 3, dtype=torch.float64))
    torch.testing.assert_close(sample[properties.dipole], torch.tensor([[3.0, 0, 0]], dtype=torch.float64))
    assert sample.meta["dipole_origin"] == "source_cartesian_origin"
    assert sample.meta["dipole_source_attributes"]["origin"] == [0.0, 0.0, 0.0]
    assert properties.atomic_charge not in sample
    assert properties.edge_idx not in sample

    batch = next(iter(data.train_dataloader()))
    assert "variational" not in batch
    assert batch[properties.positions].shape == (6, 3)
    assert batch[properties.energy].shape == (3,)
    assert batch[properties.forces].shape == (6, 3)
    assert batch[properties.dipole].shape == (3, 3)
    assert batch[properties.total_charge].shape == (3,)
    assert batch[properties.positions].dtype == batch[properties.forces].dtype == torch.float64
    assert all(mask.all() for mask in batch["masks"].values())
    assert batch["meta"][0]["dipole_source_attributes"]["origin"] == [0.0, 0.0, 0.0]

    # A model may mark positions for gradients or edit a returned sample;
    # neither operation may mutate the cached source or another split.
    sample[properties.positions].zero_()
    sample.meta["dipole_source_attributes"]["origin"][0] = 100
    assert data.train_dataset[0][properties.positions][0, 0].item() == pytest.approx(0.25)
    assert data.train_dataset[0].meta["dipole_source_attributes"]["origin"][0] == 0


def test_distance_split_is_seed_independent_and_external_file_is_validated(source, tmp_path):
    first = SpiceACVRDataModule(source, seed=4)
    second = SpiceACVRDataModule(source, seed=42)
    first.setup()
    second.setup()
    assert first.metadata == second.metadata
    indices = first.metadata["split_indices"]
    assert set(indices["train"]).isdisjoint(indices["val"])
    assert set(indices["train"]).isdisjoint(indices["test"])
    assert set(indices["val"]).isdisjoint(indices["test"])
    assert sorted(sum(indices.values(), [])) == list(range(16))
    assert all(sample.meta["distance_angstrom"] <= 5.5 for sample in first.train_dataset)
    assert all(5.5 < sample.meta["distance_angstrom"] <= 6.5 for sample in first.val_dataset)
    assert all(sample.meta["distance_angstrom"] > 6.5 for sample in first.test_dataset)
    split_file = tmp_path / "split.json"
    external = {"train": indices["train"], "validation": indices["val"], "test": indices["test"]}
    split_file.write_text(json.dumps(external))
    supplied = SpiceACVRDataset(source, split_file=split_file)
    assert supplied.metadata == first.metadata
    external["test"][0] = external["train"][0]
    split_file.write_text(json.dumps(external))
    with pytest.raises(ValueError, match="distance split|leakage"):
        SpiceACVRDataset(source, split_file=split_file)


def test_limits_cover_every_pair_and_are_manifested_for_every_split(source):
    datasets = {
        split: SpiceACVRDataset(source, split=split, train_limit=4, val_limit=2, test_limit=2)
        for split in ("train", "val", "test")
    }
    metadata = datasets["train"].metadata
    assert all(dataset.metadata == metadata for dataset in datasets.values())
    assert [len(datasets[split]) for split in ("train", "val", "test")] == [4, 2, 2]
    for split, dataset in datasets.items():
        assert {sample.meta["group"] for sample in dataset} == {"charged", "neutral"}
        assert set(dataset.indices).issubset(metadata["full_split_indices"][split])
        assert metadata["split_statistics"][split]["pair_count"] == 2
    assert metadata["split_indices"]["train"] == [0, 3, 8, 11]
    with pytest.raises(ValueError, match="cover all"):
        SpiceACVRDataset(source, train_limit=1)


def test_checkpoint_restores_next_epoch_order_without_validation_consuming_train_rng(source):
    uninterrupted = SpiceACVRDataModule(source, batch_size=3, seed=17)
    _order(uninterrupted.train_dataloader())
    state = copy.deepcopy(uninterrupted.state_dict())
    expected = _order(uninterrupted.train_dataloader())
    restored = SpiceACVRDataModule(source, batch_size=3, seed=17)
    restored.load_state_dict(state)
    before = restored.generator.get_state().clone()
    _order(restored.val_dataloader())
    _order(restored.test_dataloader())
    assert torch.equal(before, restored.generator.get_state())
    assert _order(restored.train_dataloader()) == expected
    incompatible = SpiceACVRDataModule(source, train_limit=4)
    with pytest.raises(ValueError, match="differ from the checkpoint"):
        incompatible.load_state_dict(state)


@pytest.mark.parametrize("corruption", ["nan", "dipole_missing", "wrong_unit", "overlapping_atoms"])
def test_bad_physical_data_is_rejected(source, corruption):
    import h5py

    with h5py.File(source, "r+") as handle:
        group = handle["charged"]
        if corruption == "nan":
            group["dft_total_gradient"][0, 0, 0] = np.nan
        elif corruption == "dipole_missing":
            del group["scf_dipole"]
        elif corruption == "wrong_unit":
            group["scf_dipole"].attrs["units"] = "Debye"
        else:
            group["conformations"][0, 1] = group["conformations"][0, 0]
    with pytest.raises(ValueError, match="nonfinite|scf_dipole|unsupported units|positive"):
        SpiceACVRDataset(source)
