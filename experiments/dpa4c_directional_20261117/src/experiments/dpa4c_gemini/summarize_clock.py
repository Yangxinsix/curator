"""Tabulate observed wall-budget checkpoints on the same 32 validation structures."""

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile


def main(args):
    files = sorted(args.runs.glob(f"{args.system or '*'}/*/*/validation_wall_curve.json"))
    if not files:
        raise FileNotFoundError(f"No completed wall-budget evaluation under {args.runs}")
    rows, panels, cards = [], {}, defaultdict(set)
    for path in files:
        curve = json.loads(path.read_text())
        system, arm, seed = curve["system"], curve["arm"], int(curve["seed"])
        if args.seed is not None and seed != args.seed:
            continue
        ids = curve["source_ids"]
        if len(ids) != len(set(ids)) or ids != panels.setdefault(system, ids):
            raise ValueError(f"Wall comparisons require the same validation panel in {system}")
        mode = "pure" if arm in ("A0", "B0") else "proposal"
        for point in curve["points"]:
            budget = int(point["budget_seconds"])
            gpu = point["gpu"]
            cards[system, seed, budget].add(gpu)
            summary = point.get("summary", {})
            row = dict(system=system, seed=seed, budget_seconds=budget, arm=arm, gpu=gpu,
                       panel_size=len(ids), status=point["status"], checkpoint=point.get("checkpoint"),
                       training_seconds=point.get("training_seconds"), joint_step=point.get("joint_step"),
                       ef_step=point.get("ef_step"),
                       energy_RMSE_meV_atom=summary.get(mode, {}).get("energy_RMSE_meV_atom"),
                       force_RMSE_meV_A=summary.get(mode, {}).get("force_RMSE_meV_A"),
                       clamp_minus_proposal_force_RMSE_meV_A=summary.get("clamp_minus_proposal",{}).get("force_RMSE_meV_A"),
                       stationarity_RMS_V=summary.get("stationarity_RMS_V"),
                       charge_RMSE_e=summary.get("newton_panel",{}).get("charge_RMSE_e"),
                       mean_frame_relative_transfer_error_percent=summary.get("newton_panel",{}).get("mean_frame_relative_transfer_error_percent"),
                       dipole_component_RMSE_D=summary.get("dipole",{}).get("component_RMSE_D"))
            rows.append(row)
    if not rows:
        raise ValueError("No matching system/seed")
    rows.sort(key=lambda r: (r["system"], r["seed"], r["budget_seconds"], r["arm"]))
    for r in rows:
        models = cards[r["system"], r["seed"], r["budget_seconds"]]
        r["gpu_models_in_comparison"] = " | ".join(sorted(models))
        r["gpu_comparison"] = "same_model" if len(models) == 1 else "different_models"

    def write(f):
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=args.output.parent, mode="w", newline="", delete=False) as f:
            write(f)
            temp = f.name
        os.replace(temp, args.output)
    else:
        write(sys.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--system", choices=("aumgo", "sasa"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, help="Defaults to CSV on stdout")
    main(parser.parse_args())
