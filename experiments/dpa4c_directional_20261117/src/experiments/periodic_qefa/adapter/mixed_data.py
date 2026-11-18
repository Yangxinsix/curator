"""Batch fixed-Q and open/fixed-Q potential ASE sources using Curator's boundary compiler."""
import json
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from curator.data import properties
from curator.data.acvr_ef import ACVREFDataModule
from curator.data.collate_atoms_data import collate_atoms_data


def _collate(samples):
    batch = collate_atoms_data(samples)
    # A dense per-geometry total-charge tensor exists only for an all-fixed
    # batch. Each condition has its own authoritative (possibly empty) equality.
    if any(properties.total_charge not in sample.atoms for sample in samples):
        batch.pop(properties.total_charge, None)
    return batch


class MixedChargeData:
    def __init__(self, sources, energy_convention, batch_size, val_batch_size, seed):
        self.sources = sources
        self.energy_convention = energy_convention
        self.generator = torch.Generator().manual_seed(seed)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.datasets = {split: ConcatDataset([source.datasets[split] for source in sources])
                         for split in ("train", "validation", "test")}

    def train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size, shuffle=True,
                          generator=self.generator, collate_fn=_collate)

    def val_dataloader(self):
        return DataLoader(self.datasets["validation"], batch_size=self.val_batch_size, collate_fn=_collate)

    def test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.val_batch_size, collate_fn=_collate)


def build_data(config, *, base_dir="."):
    """Each source specifies an ASE path, split JSON and a native boundary recipe.

    Repeated geometries at distinct boundary conditions must belong to the same
    split. Energies across all sources use the declared common label convention.
    """
    base_dir = Path(base_dir)
    convention = config["energy_convention"]
    if convention not in {"internal", "thermodynamic"}:
        raise ValueError("data.energy_convention must be internal or thermodynamic")
    if not config["sources"]:
        raise ValueError("data.sources must contain at least one ASE source")
    batch_size = int(config.get("batch_size", 8))
    val_batch_size = int(config.get("val_batch_size", batch_size))
    seed = int(config.get("seed", 3407))
    sources, seen = [], {}
    for cfg in config["sources"]:
        if cfg.get("energy_convention", convention) != convention:
            raise ValueError("Convert energy labels to the configured common convention before combining sources")
        path = base_dir / cfg["datapath"]
        split_path = base_dir / cfg["split_file"]
        split = json.loads(split_path.read_text())
        reference = cfg.get("energy_reference_file")
        module = ACVREFDataModule(
            datapath=str(path), split_file=str(split_path), boundary=cfg["boundary"],
            energy_convention=convention, default_dtype=config.get("default_dtype", "float64"),
            energy_reference_file=str(base_dir / reference) if reference else None,
            num_train=len(split["train"]), num_val=len(split["validation"]), num_test=len(split["test"]),
            batch_size=batch_size, val_batch_size=val_batch_size, seed=seed,
        )
        module.setup()
        for phase, subset in module.datasets.items():
            for index in subset.indices:
                geometry = subset.samples[index].variational.geometry_hash
                previous = seen.setdefault(geometry, phase)
                if previous != phase:
                    raise ValueError(f"Same geometry occurs in {previous} and {phase} at {path}, frame {index}")
        sources.append(module)
    return MixedChargeData(sources, convention, batch_size, val_batch_size, seed)
