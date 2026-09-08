"""Cached, auditable SPICE ion-pair data for fixed-charge ACVR training."""

import copy
import hashlib
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from ase.data import chemical_symbols
from torch.utils.data import DataLoader, Dataset

from . import properties
from .atoms_data import AtomsData
from .collate_atoms_data import collate_atoms_data
from .spice import SpiceDataset, ion_pair_distance_split


SPICE_ACVR_UNITS = {
    "positions": "Angstrom", "energy": "eV", "forces": "eV/Angstrom",
    "dipole": "e*Angstrom", "total_charge": "e", "atomic_charge": "e",
}


def _dtype(value):
    if value in ("float64", torch.float64):
        return torch.float64
    if value in ("float32", torch.float32):
        return torch.float32
    raise ValueError("default_dtype must be float64 or float32")


def _json_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (np.ndarray, np.generic)):
        return _json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _validate_sample(sample, index):
    shapes = {
        properties.positions: (2, 3), properties.atomic_numbers: (2,),
        properties.n_atoms: (1,), properties.energy: (1,),
        properties.forces: (2, 3), properties.dipole: (1, 3),
        properties.total_charge: (1,),
    }
    for key, shape in shapes.items():
        if key not in sample:
            raise ValueError(f"SPICE conformation {index} is missing required {key}")
        value = sample[key]
        if tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"SPICE conformation {index} has invalid {key} shape or nonfinite values")
    if sample[properties.n_atoms].item() != 2:
        raise ValueError("SPICE ACVR distance split requires two-atom ion pairs")
    numbers = sample[properties.atomic_numbers]
    if torch.any(numbers < 1) or torch.any(numbers >= len(chemical_symbols)):
        raise ValueError("SPICE contains invalid atomic numbers")
    charge = sample[properties.total_charge]
    if not torch.equal(charge, charge.round()):
        raise ValueError("SPICE total_charge must be an integer formal charge")
    distance = torch.linalg.vector_norm(sample[properties.positions][1] - sample[properties.positions][0])
    if distance <= 0:
        raise ValueError("SPICE ion-pair separation must be positive")
    return float(distance)


@lru_cache(maxsize=2)
def _load_source(path, size, mtime_ns, dtype):
    """The small subset is read once; file stat changes invalidate this cache."""
    del size, mtime_ns
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    reader = SpiceDataset(
        path, compute_neighbor_list=False, default_dtype=dtype,
        subsets=("ion_pairs",), include_dipole=True,
        include_atomic_charges=False, require_total_charge=True,
        require_forces=True, return_atoms_data=True,
    )
    try:
        raw_splits = ion_pair_distance_split(reader, 5.5, 6.5)
        splits = {"train": raw_splits["train"], "val": raw_splits["validation"], "test": raw_splits["test"]}
        samples, distances, dipole_attributes, storage_dtypes = [], [], {}, {}
        for group in reader._group_names:
            entry = reader._handle()[group]
            if "scf_dipole" not in entry:
                raise ValueError(f"SPICE ion pair {group} is missing required scf_dipole")
            dipole_attributes[group] = {
                str(key): _json_value(value) for key, value in entry["scf_dipole"].attrs.items()
            }
            storage_dtypes[group] = {
                key: str(entry[key].dtype) for key in (
                    "conformations", "formation_energy", "dft_total_gradient", "scf_dipole"
                )
            }
        split_by_index = {index: split for split, indices in splits.items() for index in indices}
        for index in range(len(reader)):
            sample = reader[index]
            distance = _validate_sample(sample, index)
            # Q is a boundary-condition input, not a partition-charge target.
            sample.atoms[properties.total_charge] = sample.targets.pop(properties.total_charge)
            sample.atoms[properties.cell] = torch.zeros((3, 3), dtype=dtype)
            sample.task = "spice_acvr"
            sample.meta.update({
                "source_index": index, "split": split_by_index[index],
                "distance_angstrom": distance,
                "dipole_origin": "source_cartesian_origin",
                "coordinate_transform": "none",
                "dipole_source_attributes": dipole_attributes[sample.meta["group"]],
                "label_units": dict(SPICE_ACVR_UNITS),
            })
            samples.append(sample)
            distances.append(distance)
        numbers = sorted({int(number) for sample in samples for number in sample[properties.atomic_numbers]})
        metadata = {
            "schema": "spice_acvr_ion_pairs_v1",
            "source_sha256": digest.hexdigest(),
            "source": {
                "dataset": "SPICE", "path": path, "energy_key": "formation_energy",
                "subsets": sorted({sample.meta["subset"] for sample in samples}),
                "groups": list(reader._group_names), "num_conformations": len(samples),
                "energy_semantics": "DFT formation energy relative to the source SPICE atomic references",
                "force_semantics": "negative DFT total-energy Cartesian gradient; reference energies are coordinate independent",
                "total_charge_semantics": "integer net charge from mapped-SMILES formal charges; MBIS sum checked when available",
                "dipole_semantics": "SCF molecular dipole including electronic and nuclear contributions in the stored coordinate frame",
                "dipole_source_attributes": dipole_attributes,
                "storage_dtypes": storage_dtypes,
            },
            "units": dict(SPICE_ACVR_UNITS),
            "gauge": {
                "boundary": "fixed_total_charge", "electrostatics": "isolated_nonperiodic",
                "energy_zero": "SPICE formation_energy atomic references",
                "dipole_origin": "source_cartesian_origin", "coordinate_transform": "none",
                "charged_dipoles": "origin dependent; stored positions and dipoles must be transformed together",
                "origin_verification": "source attributes preserved; no additional origin inference or recentering",
            },
            "labels": ["energy", "forces", "dipole"], "boundary_inputs": ["total_charge"],
            "species": [chemical_symbols[number] for number in numbers], "atomic_numbers": numbers,
            "default_dtype": str(dtype).removeprefix("torch."),
            "split_rule": {
                "kind": "ion_pair_distance", "unit": "Angstrom",
                "train": "r <= 5.5", "val": "5.5 < r <= 6.5", "test": "r > 6.5",
                "seed_dependent": False,
            },
            "full_split_indices": splits,
        }
        return tuple(samples), tuple(distances), metadata
    finally:
        if reader._file is not None:
            reader._file.close()
            reader._file = None


def _validate_split_file(path, expected):
    if path is None:
        return
    with open(Path(path).expanduser(), encoding="utf-8") as handle:
        actual = json.load(handle)
    if "validation" in actual:
        if "val" in actual:
            raise ValueError("split file cannot contain both validation and val")
        actual["val"] = actual.pop("validation")
    if set(actual) != set(expected):
        raise ValueError("split file must define exactly train, validation/val, and test")
    for split, indices in actual.items():
        if not isinstance(indices, list) or any(type(index) is not int for index in indices):
            raise ValueError("split indices must be integer lists")
        if len(indices) != len(set(indices)) or set(indices) != set(expected[split]):
            raise ValueError(f"{split} indices disagree with the fixed distance split or contain leakage/duplicates")


def _limited_indices(indices, limit, samples):
    if limit is None:
        return list(indices)
    if type(limit) is not int or limit < 1:
        raise ValueError("dataset limits must be positive integers or None")
    if limit >= len(indices):
        return list(indices)
    by_pair = defaultdict(list)
    for index in indices:
        by_pair[samples[index].meta["group"]].append(index)
    if limit < len(by_pair):
        raise ValueError(f"limit {limit} cannot cover all {len(by_pair)} ion pairs")
    counts = dict.fromkeys(by_pair, 0)
    for _ in range(limit):
        pair = min((name for name in by_pair if counts[name] < len(by_pair[name])),
                   key=lambda name: (counts[name], name))
        counts[pair] += 1
    selected = []
    for pair, values in by_pair.items():
        values = sorted(values, key=lambda index: samples[index].meta["distance_angstrom"])
        selected.extend(values[index] for index in np.linspace(0, len(values) - 1, counts[pair], dtype=int))
    return sorted(selected)


def _split_statistics(samples, indices):
    def value_range(key):
        values = torch.cat([samples[index][key].reshape(-1) for index in indices])
        return [float(values.min()), float(values.max())]

    distances = [samples[index].meta["distance_angstrom"] for index in indices]
    return {
        "count": len(indices), "pair_count": len({samples[index].meta["group"] for index in indices}),
        "distance_range_angstrom": [min(distances), max(distances)],
        "energy_range_eV": value_range(properties.energy),
        "force_component_range_eV_per_Angstrom": value_range(properties.forces),
        "dipole_component_range_e_Angstrom": value_range(properties.dipole),
        "total_charge_values_e": sorted({float(samples[index][properties.total_charge].item()) for index in indices}),
    }


class SpiceACVRDataset(Dataset):
    """One distance split, with identical source/split manifests across phases."""

    def __init__(self, datapath, split="train", split_file=None, default_dtype="float64",
                 train_limit=None, val_limit=None, test_limit=None):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.datapath = str(Path(datapath).expanduser().resolve())
        stat = Path(self.datapath).stat()
        self.default_dtype = _dtype(default_dtype)
        samples, _, metadata = _load_source(self.datapath, stat.st_size, stat.st_mtime_ns, self.default_dtype)
        _validate_split_file(split_file, metadata["full_split_indices"])
        self.metadata = copy.deepcopy(metadata)
        limits = {"train": train_limit, "val": val_limit, "test": test_limit}
        selected = {
            phase: _limited_indices(indices, limits[phase], samples)
            for phase, indices in metadata["full_split_indices"].items()
        }
        self.metadata.update({
            "splits": selected, "split_indices": selected,
            "limits": limits, "limit_selection": "balanced pair quotas; deterministic distance linspace within each pair",
            "split_statistics": {phase: _split_statistics(samples, indices) for phase, indices in selected.items()},
        })
        self.split, self.indices = split, selected[split]
        self.species = self.metadata["species"]
        self.samples = tuple(samples[index] for index in self.indices)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return AtomsData(
            atoms={key: value.clone() for key, value in sample.atoms.items()},
            targets={key: value.clone() for key, value in sample.targets.items()},
            task=sample.task, weight=sample.weight, meta=copy.deepcopy(sample.meta),
        )

    def get_n_atoms(self, index):
        return int(self.samples[index][properties.n_atoms].item())


class SpiceACVRDataModule(pl.LightningDataModule):
    """Normal atom batches with replayable training shuffles and fixed splits."""

    def __init__(self, datapath, split_file=None, batch_size=32, seed=17, num_workers=0,
                 shuffle=True, default_dtype="float64", heads=None,
                 train_limit=None, val_limit=None, test_limit=None):
        super().__init__()
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(num_workers) is not int or num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")
        self.datapath = str(Path(datapath).expanduser().resolve())
        self.train_path = self.datapath
        self.split_file = split_file
        self.batch_size, self.num_workers, self.shuffle = batch_size, num_workers, bool(shuffle)
        self.seed = int(seed)
        self.default_dtype = _dtype(default_dtype)
        self.heads = heads or ["energy", "forces", "dipole"]
        self.dataset_kwargs = dict(datapath=self.datapath, split_file=split_file,
                                   default_dtype=self.default_dtype, train_limit=train_limit,
                                   val_limit=val_limit, test_limit=test_limit)
        self.generator = torch.Generator().manual_seed(self.seed)
        self.datasets = {}

    def setup(self, stage=None):
        if not self.datasets:
            self.datasets = {split: SpiceACVRDataset(split=split, **self.dataset_kwargs)
                             for split in ("train", "val", "test")}
            self.train_dataset, self.val_dataset, self.test_dataset = (
                self.datasets[split] for split in ("train", "val", "test")
            )
            self.metadata = self.train_dataset.metadata
            self.species = self.metadata["species"]
            self.dataset = self.train_dataset
            self.train_idx, self.val_idx, self.test_idx = (
                self.metadata["splits"][split] for split in ("train", "val", "test")
            )

    def _loader(self, split):
        self.setup()
        generator = self.generator if split == "train" else torch.Generator().manual_seed(
            self.seed + (1 if split == "val" else 2)
        )
        return DataLoader(self.datasets[split], batch_size=self.batch_size,
                          shuffle=self.shuffle and split == "train", generator=generator,
                          num_workers=self.num_workers, collate_fn=collate_atoms_data)

    def train_dataloader(self):
        return self._loader("train")

    def val_dataloader(self):
        return self._loader("val")

    def test_dataloader(self):
        return self._loader("test")

    def state_dict(self):
        self.setup()
        return {"metadata": self.metadata, "training_seed": self.seed,
                "generator_state": self.generator.get_state()}

    def load_state_dict(self, state_dict):
        self.setup()
        if state_dict["metadata"] != self.metadata:
            raise ValueError("SPICE source, units, or distance splits differ from the checkpoint")
        self.seed = int(state_dict.get("training_seed", self.seed))
        self.generator.set_state(state_dict["generator_state"])


__all__ = ["SPICE_ACVR_UNITS", "SpiceACVRDataset", "SpiceACVRDataModule"]
