"""Select DPA4C confirmation and test tasks using the original complete validation split."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from tempfile import NamedTemporaryFile


# Fingerprint sorted original ASE source IDs from the registered historical split.
FOLDS = {
    "aumgo": (
        450,
        110,
        "009c932447ef42fca1738853551e7768547def192b6afd6fb303a6cc924e69d1",
    ),
    "sasa": (
        2000,
        14,
        "d1df48628e4fd38e4fa0863c7f41258ec4b94dcf293ad79f766dd4ec2bf66883",
    ),
}
ARMS = {"aumgo": ("A0", "A1", "A2", "A3", "A4"), "sasa": ("B0", "B1", "B2", "B3", "B4")}
# Practical tolerance selected before training starts; it is not an uncertainty estimate.
MARGIN = 0.2  # meV / Angstrom


def read_final(root, system, seed, arm):
    path = root / system / str(seed) / arm / "validation_final.json"
    raw = path.read_bytes()
    doc = json.loads(raw)
    summary, rows = doc["summary"], doc["frames"]
    expected, atoms, signature = FOLDS[system]
    ids = sorted(r["source_index"] for r in rows)
    ids_hash = hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        summary["frames"] != expected
        or ids_hash != signature
        or any(r["atoms"] != atoms for r in rows)
    ):
        raise ValueError(f"Partial or different validation set in {path}")
    pure = arm in {"A0", "B0"}
    mode = "pure" if pure else "proposal"

    def rms(name):
        value = math.sqrt(
            sum(r[name]["force_sse_meV2_A2"] for r in rows) / (3 * atoms * expected)
        )
        if not math.isfinite(value) or not math.isclose(
            value, summary[name]["force_RMSE_meV_A"], abs_tol=1e-6, rel_tol=1e-7
        ):
            raise ValueError(
                f"Nonfinite or inconsistent per-frame force errors in {path}"
            )
        return value

    force = rms(mode)
    energy = summary[mode]["energy_RMSE_meV_atom"]
    if not math.isfinite(energy):
        raise ValueError(f"Nonfinite validation energy in {path}")
    result = dict(
        source=str(path.relative_to(root)),
        sha256=hashlib.sha256(raw).hexdigest(),
        force_RMSE_meV_A=force,
        energy_RMSE_meV_atom=energy,
    )
    if not pure:
        result["clamp_minus_proposal_force_RMSE_meV_A"] = rms("uniform") - force
    return result


def screen(root, system, screen_seed):
    metrics = {arm: read_final(root, system, screen_seed, arm) for arm in ARMS[system]}
    force = {arm: row["force_RMSE_meV_A"] for arm, row in metrics.items()}
    if system == "aumgo":
        retained = ["A0", "A3", "A4"]
        if force["A2"] <= force["A3"] + MARGIN:
            retained.append("A2")
        # A shared kernel that actually ties/beats the best model is a meaningful control.
        if force["A1"] <= min(force[a] for a in ("A2", "A3", "A4")) + MARGIN:
            retained.append("A1")
    else:
        retained = ["B0", "B1", min(("B2", "B3", "B4"), key=force.get)]
    return sorted(retained), metrics


def plan(root, system, screen_seed, confirm_seeds, stage):
    retained, initial = screen(root, system, screen_seed)
    report = dict(
        complete_validation_frames=FOLDS[system][0],
        retained=retained,
        initial=initial,
        confirmation_seeds=confirm_seeds,
    )
    if stage == "screen":
        arms, seeds = retained, confirm_seeds
    else:
        seeds = [screen_seed, *confirm_seeds]
        stats = {}
        for arm in retained:
            items = {
                screen_seed: initial[arm],
                **{s: read_final(root, system, s, arm) for s in confirm_seeds},
            }
            stats[arm] = dict(
                by_seed=items,
                mean_force_RMSE_meV_A=statistics.mean(
                    x["force_RMSE_meV_A"] for x in items.values()
                ),
                mean_energy_RMSE_meV_atom=statistics.mean(
                    x["energy_RMSE_meV_atom"] for x in items.values()
                ),
            )
            if arm != retained[0]:
                stats[arm]["mean_clamp_minus_proposal_force_RMSE_meV_A"] = (
                    statistics.mean(
                        x["clamp_minus_proposal_force_RMSE_meV_A"]
                        for x in items.values()
                    )
                )
        base, *variants = retained
        best = min(stats[a]["mean_force_RMSE_meV_A"] for a in variants)
        # ARMS is ordered by added parameters (AuMgO) or intervention complexity (SA-SA).
        chosen = next(
            a for a in variants if stats[a]["mean_force_RMSE_meV_A"] <= best + MARGIN
        )
        arms = [base, chosen]
        report.update(
            chosen_acvr=chosen,
            retained_validation_results=stats,
            chosen_improvement_vs_pure_meV_A=stats[base]["mean_force_RMSE_meV_A"]
            - stats[chosen]["mean_force_RMSE_meV_A"],
        )
    report["tasks"] = [dict(system=system, seed=s, arm=a) for s in seeds for a in arms]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("screen", "select-test"))
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system", choices=tuple(FOLDS) + ("all",), default="all")
    parser.add_argument("--screen-seed", type=int, default=3407)
    parser.add_argument("--confirm-seeds", type=int, nargs=2, default=(3408, 3409))
    args = parser.parse_args()
    if len(set([args.screen_seed, *args.confirm_seeds])) != 3:
        parser.error("Specify three distinct training seeds")
    root = args.runs.resolve()
    systems = FOLDS if args.system == "all" else (args.system,)
    reports = {
        s: plan(root, s, args.screen_seed, args.confirm_seeds, args.stage)
        for s in systems
    }
    manifest = dict(
        stage=args.stage,
        screen_seed=args.screen_seed,
        force_tolerance_meV_A=MARGIN,
        selection_metric="Complete validation force RMSE before clamping; lower is better.",
        systems=reports,
        tasks=[t for r in reports.values() for t in r["tasks"]],
    )
    payload = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != payload:
        with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
            f.write(payload)
            temp = f.name
        os.replace(temp, path)
    print(
        json.dumps(
            dict(stage=args.stage, tasks=len(manifest["tasks"]), output=str(path))
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
