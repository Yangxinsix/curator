#!/usr/bin/env python3
"""Audit TRECI paired conditions and write reproducible manifests."""

import argparse
import hashlib
import json
from pathlib import Path

from curator.data import TreciDataset, properties, treci_geometry_split
from curator.data.treci import _CONDITIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    dataset = TreciDataset(str(args.dataset_root), compute_neighbor_list=False)
    args.output_directory.mkdir(parents=True, exist_ok=True)

    fully_labeled = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if all(
            condition.electron_count is not None
            and properties.electrode_potential in condition.labels
            and properties.variable_value in condition.labels
            for condition in sample.variational.conditions
        ):
            fully_labeled.append(sample.variational.geometry_uid)

    files = {}
    for label, potential, relative_path in _CONDITIONS:
        path = args.dataset_root / relative_path
        files[label] = {
            "path": relative_path,
            "nominal_potential_v_vs_she": potential,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    uid_text = "\n".join(dataset._geometry_uids) + "\n"
    manifest = {
        "dataset": "TRECI Cu/water common-geometry potential matrix",
        "source": "https://github.com/michelegiovannibianchi/TRECI",
        "git_commit": "24a0fe816e8aa93b6b604df9aa354806a9867cf8",
        "license": "MIT",
        "geometry_count": len(dataset),
        "conditions_per_geometry": 8,
        "numeric_fixed_potentials_per_geometry": 7,
        "fully_electron_and_bader_labeled_geometry_count": len(fully_labeled),
        "geometry_uid_algorithm": "sha256(little-endian int32 atomic numbers || float64 positions)",
        "sorted_geometry_uid_list_sha256": hashlib.sha256(uid_text.encode("ascii")).hexdigest(),
        "files": files,
        "capabilities": dataset.capabilities.to_dict(),
    }
    (args.output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (args.output_directory / "geometry_split.json").write_text(
        json.dumps(treci_geometry_split(dataset), indent=2) + "\n"
    )
    (args.output_directory / "fully_labeled_geometry_uids.txt").write_text(
        "\n".join(sorted(fully_labeled)) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
