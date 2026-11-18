#!/usr/bin/env python3
"""Prepare matched proposal arms with frozen training scales and source snapshots.

This writes reviewable configs and provenance; it does not submit training jobs.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

from curator.commands.common import ensure_resolvers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "curator/configs/acvr_periodic_charge.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=["mlp", "pointwise", "mp", "attention", "ewald"])
    parser.add_argument("--override", action="append", default=[], help="Hydra override; repeat as needed")
    parser.add_argument("--validation-geometries", type=int, default=4)
    args = parser.parse_args()
    ensure_resolvers()
    config_path = args.config.resolve()
    with initialize_config_dir(config_dir=str(config_path.parent), version_base=None):
        config = compose(config_name=config_path.stem, overrides=args.override)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(int(config.seed))
    data = instantiate(config.data)
    data.setup("fit")
    metadata = data.metadata
    scales = metadata.get("supervised_scales", metadata.get("training_statistics", {}).get("supervised_scales"))
    if not scales:
        raise ValueError("this preparer requires explicit TRAIN-only supervised scales in dataset metadata")
    loss = config.task.outputs[0].loss_fn
    loss.supervised_scales = {name: scales[name] for name, weight in loss.supervised_weights.items() if weight > 0}
    head = config.model.output_modules[0]
    periodic = head.get("internal_energy", {}).get("_target_", "").endswith("EwaldChargeEnergy")
    arms = args.arms or (["mlp", "pointwise", "mp", "attention", "ewald"] if periodic
                         else ["mlp", "pointwise", "mp", "attention"])
    if "ewald" in arms and not periodic:
        raise ValueError("Ewald communication requires the common periodic charge energy")
    # A prepared experiment always starts fresh. Migrating a trained functional
    # is an explicit option of the separate fixed-functional diagnostic runner.
    if config.model_path is not None:
        raise ValueError("matched joint arms require model_path=null")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata_raw = (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode()
    (args.output / "data_metadata.json").write_bytes(metadata_raw)
    source_hashes = {}
    with tarfile.open(args.output / "source.tar.gz", "w:gz") as archive:
        files = sorted(set((ROOT / "curator").rglob("*.py"))
                       | set((ROOT / "curator/configs").rglob("*.yaml"))
                       | set((ROOT / "scripts").glob("acvr*.py"))
                       | set((ROOT / "docs/acvr").glob("*.md"))
                       | {ROOT / "pyproject.toml"})
        for path in files:
            relative = str(path.relative_to(ROOT))
            source_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.add(path, arcname=relative)
    (args.output / "source_hashes.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    manifest = {"schema": "acvr_proposal_joint_arms_v1", "seed": int(config.seed),
                "arms": arms, "config_source": str(config_path), "overrides": args.override,
                "shared_initialization_arms": [arm for arm in arms if arm != "mlp"],
                "mlp_initialization": "legacy constructor order; separate historical architecture baseline",
                "runtime": {"python": sys.version, "torch": torch.__version__,
                            "pytorch_lightning": version("pytorch-lightning"), "e3nn": version("e3nn")},
                "data_metadata_sha256": hashlib.sha256(metadata_raw).hexdigest(),
                "supervised_scales": dict(loss.supervised_scales), "scale_fit_split": "train",
                "source_archive_sha256": hashlib.sha256((args.output / "source.tar.gz").read_bytes()).hexdigest(),
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                "training_submitted": False}
    for arm in arms:
        candidate = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        candidate.run_path = str((args.output / arm).resolve())
        candidate.trainer.default_root_dir = candidate.run_path
        candidate.trainer.logger.save_dir = candidate.run_path
        for callback in candidate.trainer.callbacks:
            if "dirpath" in callback:
                callback.dirpath = str(Path(candidate.run_path) / "model_path")
        if "hydra" in candidate:
            candidate.hydra.run.dir = str(Path(candidate.run_path) / "hydra")
        if arm == "mlp":
            candidate.model.output_modules[0].proposal = "mlp"
        else:
            candidate.model.output_modules[0].proposal.communication = arm
        candidate.trainer.callbacks.append({
            "_target_": "curator.train.proposal_validation.ProposalEquilibriumValidation",
            "every_n_epochs": 5, "num_geometries": args.validation_geometries,
            "compute_responses": True, "supervised_limits": {}})
        OmegaConf.save(candidate, args.output / f"{arm}.yaml", resolve=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "arms": arms, "training_submitted": False}))


if __name__ == "__main__":
    main()
