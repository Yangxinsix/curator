#!/usr/bin/env python3
"""Generate and audit field conditions; never construct or train a proposal."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from curator.data.periodic_charge_oracle import PeriodicChargeOracleDataset
from curator.data.periodic_charge_coverage import expand_field_conditions, teacher_matrices


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics(values):
    values = torch.tensor(values, dtype=torch.float64)
    return {"count": len(values), "min": float(values.min()),
            "median": float(values.median()), "max": float(values.max())}


def audit_sample(sample, metadata):
    """Check dense-KKT labels with an independent reduced positive-definite solve."""
    n, side = metadata["n_atoms"], metadata["n_atoms"] // 2
    hessian, chi, constant = teacher_matrices(sample, metadata)
    groups = hessian.new_zeros((2, n))
    groups[0, :side], groups[1, side:] = 1, 1
    conditions = sample.variational.conditions
    by_uid = {condition.condition_uid: condition for condition in conditions}
    if len(by_uid) != len(conditions):
        raise ValueError("duplicate condition UID")
    result = {"geometry_uid": sample.variational.geometry_uid,
              "geometry_hash": sample.variational.geometry_hash, "conditions": len(conditions)}
    errors = defaultdict(float)
    ranks, response = {}, defaultdict(list)
    local = torch.stack([condition.inputs["local_external_field"].flatten() for condition in conditions])
    errors["group_field_sum_V"] = float((local @ groups.T).abs().max())
    result["maximum_local_field_V"] = float(local.abs().max())
    for kind in ("conq", "conp"):
        selected = [condition for condition in conditions if condition.provenance["boundary_kind"] == kind]
        constraint = groups if kind == "conq" else hessian.new_ones((1, n))
        tangent = torch.linalg.svd(constraint, full_matrices=True).Vh[constraint.shape[0]:].T
        reduced = tangent.T @ hessian @ tangent
        spectrum = torch.linalg.eigvalsh(reduced)
        if float(spectrum.min()) <= 1e-8:
            raise ValueError("teacher is not stable on the feasible space")
        result[f"{kind}_minimum_curvature_V_per_e"] = float(spectrum.min())
        result[f"{kind}_condition_number"] = float(spectrum.max() / spectrum.min())
        targets = torch.stack([c.inputs["constraint_target"] for c in selected], dim=1)
        drives = torch.stack([c.inputs["local_external_field"].flatten() for c in selected], dim=1)
        if kind == "conp":
            drives += groups.T @ torch.stack([c.inputs["control_value"] for c in selected], dim=1)
        charge = torch.stack([c.labels["variable_value"].flatten() for c in selected], dim=1)
        origin = constraint.T @ torch.linalg.solve(constraint @ constraint.T, targets)
        rhs = tangent.T @ (drives - chi[:, None] - hessian @ origin)
        independent = origin + tangent @ torch.cholesky_solve(rhs, torch.linalg.cholesky(reduced))
        gradient = hessian @ charge + chi[:, None] - drives
        multipliers = -torch.stack([c.labels["constraint_derivative"] for c in selected], dim=1)
        expected_energy = constant + ((chi[:, None] - drives) * charge).sum(0)
        expected_energy += 0.5 * (charge * (hessian @ charge)).sum(0)
        checks = {
            "independent_state_e": (charge - independent).abs().max(),
            "constraint_e": (constraint @ charge - targets).abs().max(),
            "kkt_V": (gradient + constraint.T @ multipliers).abs().max(),
            "energy_eV": (expected_energy - torch.cat([c.labels["condition_energy"] for c in selected])).abs().max(),
        }
        if kind == "conp":
            checks["control_derivative_e"] = (groups @ charge + torch.stack(
                [c.labels["control_derivative"] for c in selected], dim=1)).abs().max()
        for name, value in checks.items():
            errors[name] = max(errors[name], float(value))
        result[f"{kind}_maximum_abs_charge_e"] = float(charge.abs().max())
        for group, start in (("left", 0), ("right", side)):
            amplitudes = sorted({c.provenance["field_peak_V"] for c in selected
                                 if "field_peak_V" in c.provenance})
            for amplitude in amplitudes:
                fields = torch.stack([c.inputs["local_external_field"].flatten()[start:start + side]
                                      for c in selected if c.provenance.get("perturbed_group") == group
                                      and c.provenance.get("direction_family") == "dense_zero_sum"
                                      and c.provenance["field_peak_V"] == amplitude
                                      and c.provenance["sign"] == 1])
                singular = torch.linalg.svdvals(fields)
                rank = int((singular > 1e-10).sum())
                if rank != side - 1:
                    raise ValueError("zero-sum field basis does not have full rank")
                ranks[f"{kind}/{group}/{amplitude:g}"] = {
                    "rank": rank, "required_rank": side - 1,
                    "basis_condition_number": float(singular.max() / singular.min())}
    for condition in conditions:
        condition.validate(n)
        p = condition.provenance
        if "pair_uid" not in p:
            continue
        errors["field_peak_V"] = max(errors["field_peak_V"], abs(
            float(condition.inputs["local_external_field"].abs().max()) - p["field_peak_V"]))
        if p["sign"] != 1:
            continue
        negative = by_uid[p["pair_uid"] + ":-1"]
        base = by_uid[p["base_condition_uid"]]
        if negative.provenance["base_condition_uid"] != base.condition_uid:
            raise ValueError("positive and negative conditions have different baselines")
        delta = condition.labels["variable_value"] - base.labels["variable_value"]
        symmetry = delta + negative.labels["variable_value"] - base.labels["variable_value"]
        errors["charge_pair_linearity_e"] = max(errors["charge_pair_linearity_e"], float(symmetry.abs().max()))
        errors["field_pair_V"] = max(errors["field_pair_V"], float((
            condition.inputs["local_external_field"] + negative.inputs["local_external_field"]
        ).abs().max()))
        if p["perturbed_group"] != "both":
            remote = delta[side:] if p["perturbed_group"] == "left" else delta[:side]
            key = f"{p['boundary_kind']}/{p['perturbed_group']}/{p['direction_family']}/{p['field_peak_V']:g}"
            response[key].append(float(remote.square().mean().sqrt()))
    if any(not torch.isfinite(torch.tensor(value)) or value > 1e-10 for value in errors.values()):
        raise ValueError(f"data audit failed: {dict(errors)}")
    result["maximum_errors"] = dict(errors)
    result["direction_coverage"] = ranks
    return result, response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--condition-seed", type=int, default=20260914)
    args = parser.parse_args()
    torch.set_num_threads(1)
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise FileExistsError("output already contains a frozen manifest; choose a new directory")
    amplitudes = [0.1, 0.3, 1.0]
    protocol = {"schema": "periodic_charge_field_coverage_v2", "geometry_seed": args.seed,
                "condition_seed": args.condition_seed, "amplitudes_V": amplitudes,
                "amplitude_definition": "maximum absolute local per-site potential",
                "normalization": "each direction divided by its maximum absolute component; group sums zero",
                "families": ["full dense zero-sum basis on each side", "two random sparse pairs per side",
                             "two fields perturbing both sides"],
                "signs": [1, -1], "legacy_conditions": "all six preserved exactly",
                "split_policy": "reuse original geometry split; never split conditions of one geometry",
                "selection_rule": "0.1 V legacy peak, 0.3 V intermediate, 1.0 V cap matching existing group-control extent",
                "acceptance": "finite exact labels; all zero-sum ranks 15; independent solve and pair errors <=1e-10",
                "training_submitted": False}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    old_manifest = json.loads(args.legacy_manifest.read_text()) if args.legacy_manifest else None
    manifest = {"archives": {}, "probes": {}, "source_sha256": {}, "training_submitted": False,
                "test_use": "generation and data-integrity checks only; no model evaluation or selection"}
    audit = {"splits": {}, "probes": {}, "legacy_manifest": str(args.legacy_manifest),
             "legacy_geometry_hash_matches": {}}
    hashes, cached_train = {}, None
    for split in ("train", "val", "test"):
        cache = args.output / "legacy_train.pt"
        if split == "train" and cache.exists():
            source = torch.load(cache, weights_only=False, map_location="cpu")
            samples, metadata = source["samples"], source["metadata"]
            if metadata["seed"] != args.seed:
                raise ValueError("cached geometry seed mismatch")
        else:
            source = PeriodicChargeOracleDataset(split=split, seed=args.seed)
            samples, metadata = source.samples, source.metadata
        if split == "train":
            cached_train = samples[:8]
            manifest["dataset_metadata"] = {**metadata, "source_schema": metadata["schema"],
                "schema": protocol["schema"], "field_design": protocol,
                "num_conditions": 6 + (2 * (metadata["n_atoms"] // 2 - 1) + 6) * 3 * 2 * 2}
        hashes[split] = {sample.variational.geometry_hash for sample in samples}
        old = (old_manifest or {}).get("splits", {}).get(split)
        if old:
            expected = {row["uid"]: row["hash"] for row in old["geometries"]}
            actual = {sample.variational.geometry_uid: sample.variational.geometry_hash for sample in samples}
            if actual != expected:
                raise ValueError("legacy geometry hashes differ")
            audit["legacy_geometry_hash_matches"][split] = len(actual)
        expanded, rows, response = [], [], defaultdict(list)
        for index, sample in enumerate(samples):
            item = expand_field_conditions(sample, metadata, amplitudes, args.condition_seed)
            row, changes = audit_sample(item, metadata)
            expanded.append(item)
            rows.append(row)
            for key, values in changes.items():
                response[key].extend(values)
            if (index + 1) % 32 == 0:
                print(json.dumps({"split": split, "completed_geometries": index + 1}), flush=True)
        path = args.output / f"{split}.pt"
        torch.save({"split": split, "geometry_indices": metadata["splits"][split], "samples": expanded}, path)
        manifest["archives"][split] = {"path": path.name, "sha256": digest(path),
            "bytes": path.stat().st_size, "geometries": len(expanded),
            "conditions": sum(len(item.variational.conditions) for item in expanded)}
        audit["splits"][split] = {"rows": rows, "maximum_errors": {
            name: max(row["maximum_errors"].get(name, 0) for row in rows)
            for name in rows[0]["maximum_errors"]}}
        # Do not use TEST response distributions in calibration or reporting.
        if split != "test":
            audit["splits"][split]["remote_charge_change_rms_e"] = {
                key: statistics(values) for key, values in response.items()}
        print(json.dumps({"saved": path.name, **manifest["archives"][split]}), flush=True)
    if any(hashes[a] & hashes[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("geometry leakage between data splits")
    audit["geometry_splits_disjoint"] = True
    # Separate probes distinguish unseen directions from unseen amplitudes on
    # the same eight TRAIN geometries. Neither archive enters training/scales.
    for name, seed, levels in (("new_directions", args.condition_seed + 1, amplitudes),
                               ("new_amplitudes", args.condition_seed, [0.2, 0.6])):
        items, rows = [], []
        for sample in cached_train:
            item = expand_field_conditions(sample, metadata, levels, seed)
            row, _ = audit_sample(item, metadata)
            items.append(item)
            rows.append(row)
        path = args.output / f"probe_{name}.pt"
        torch.save({"role": f"same-TRAIN-geometry held-out {name}; exclude from training and scale fits",
                    "samples": items}, path)
        manifest["probes"][name] = {"path": path.name, "sha256": digest(path), "bytes": path.stat().st_size,
                                     "geometries": len(items), "condition_seed": seed, "amplitudes_V": levels}
        audit["probes"][name] = rows
    sources = [Path(__file__), ROOT / "curator/data/periodic_charge_coverage.py",
               ROOT / "curator/data/periodic_charge_oracle.py"]
    snapshot = args.output / "source"
    for source in sources:
        target = snapshot / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        manifest["source_sha256"][str(source.relative_to(ROOT))] = digest(source)
    if args.legacy_manifest:
        manifest["legacy_manifest_sha256"] = digest(args.legacy_manifest)
    manifest["elapsed_seconds"] = time.monotonic() - start
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"completed": str(args.output), "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
