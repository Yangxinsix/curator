"""Collect completed SPICE seed runs without selecting models on test errors."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    args = parser.parse_args()
    campaign = Path(args.campaign).resolve()
    manifest = json.loads((campaign / "manifest.json").read_text())
    source_hashes = json.loads(Path(manifest["source_hashes"]).read_text())
    for filename, expected in source_hashes.items():
        path = Path(manifest["source_snapshot"]) / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Campaign source snapshot changed: {filename}")
    rows, reports = [], []
    dataset_metadata = None
    for job in manifest["jobs"]:
        if job.get("kind") != "evaluate":
            continue
        path = Path(job["success_artifacts"][0])
        report = json.loads(path.read_text())
        if report.get("schema") != "spice_acvr_evaluation_v1" or report.get("model_seed") != job["seed"]:
            raise ValueError(f"Evaluation report schema/seed differs from job {job['id']}")
        expected_variant = "short_range" if job["experiment"] == "short_range" else "acvr"
        if report["variant"] != expected_variant or report["stage"] != job["stage"]:
            raise ValueError(f"Evaluation report variant/stage differs from job {job['id']}")
        if report.get("units") != {"energy": "eV", "forces": "eV/Angstrom", "dipole": "e*Angstrom"}:
            raise ValueError(f"Evaluation report units differ from job {job['id']}")
        if dataset_metadata is None:
            dataset_metadata = report["dataset_metadata"]
        elif report["dataset_metadata"] != dataset_metadata:
            raise ValueError("Cannot aggregate evaluations with different dataset source/split manifests")
        if set(report["splits"]) != {"val", "test"}:
            raise ValueError(f"Evaluation report lacks val/test splits: {job['id']}")
        reports.append({"job": job["id"], "metrics": str(path), "checkpoint": report["checkpoint"],
                        "checkpoint_sha256": report["checkpoint_sha256"]})
        for split, metrics in report["splits"].items():
            row = {"experiment": job["experiment"], "seed": job["seed"], "stage": report["stage"],
                   "split": split, "n_geometries": metrics["n_geometries"]}
            for label in ("energy", "forces", "dipole"):
                row[label + "_rmse"] = metrics[label]["rmse"]
            for prefix in ("proposal", "untrained_proposal"):
                if prefix in metrics:
                    audit = metrics[prefix]
                    row[prefix + "_projected_state_rmse_e"] = audit["projected_proposal_state_rmse_e"]
                    for tolerance, values in audit.items():
                        if isinstance(values, dict) and "accepted_projected" in values:
                            row[prefix + "_acceptance_fraction_at_" + tolerance] = values["accepted_projected"] / values["n_geometries"]
            rows.append(row)
    aggregate = []
    keys = sorted({(row["experiment"], row["stage"], row["split"]) for row in rows})
    for experiment, stage, split in keys:
        group = [row for row in rows if (row["experiment"], row["stage"], row["split"]) == (experiment, stage, split)]
        if sorted(row["seed"] for row in group) != sorted(manifest["experiment"]["model_seeds"]):
            raise ValueError(f"Missing or duplicate model seeds for {experiment}/{stage}/{split}")
        result = {"experiment": experiment, "stage": stage, "split": split, "seeds": [row["seed"] for row in group]}
        for label in group[0]:
            if label.endswith("_rmse") or label.startswith(("proposal_", "untrained_proposal_")):
                values = [row[label] for row in group]
                result[label + "_mean"] = statistics.mean(values)
                result[label + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate.append(result)
    with (campaign / "metrics_by_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    output = {"schema": "spice_campaign_summary_v1", "runs": reports, "aggregate": aggregate,
              "dataset_metadata": dataset_metadata, "source_hashes": manifest["source_hashes"],
              "environment_manifest": manifest["environment_manifest"],
              "units": {"energy": "eV", "forces": "eV/Angstrom", "dipole": "e*Angstrom"},
              "scope": "Diatomic distance extrapolation; physical predictions use exact internal solves; no constant-potential or speedup claim"}
    temporary = campaign / "summary.json.tmp"
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(campaign / "summary.json")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
