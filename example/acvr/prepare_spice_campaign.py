"""Snapshot code and prepare the persistent SPICE three-seed experiment DAG."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--physical-epochs", type=int, default=200)
    parser.add_argument("--proposal-epochs", type=int, default=100)
    args = parser.parse_args()
    if min(args.physical_epochs, args.proposal_epochs) < 1:
        parser.error("epoch budgets must be positive")
    repo = Path(__file__).resolve().parents[2]
    campaign = Path(args.output).resolve()
    campaign.mkdir(parents=True, exist_ok=False)
    source = campaign / "source"
    for directory in ("curator", "scripts", "example/acvr"):
        shutil.copytree(repo / directory, source / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    hashes = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(source.rglob("*")) if path.is_file()}
    (campaign / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    for name, command in (("git_diff.patch", ["git", "diff", "--binary"]),
                          ("git_status.txt", ["git", "status", "--short"]),
                          ("git_head.txt", ["git", "rev-parse", "HEAD"])):
        with (campaign / name).open("w") as handle:
            subprocess.run(command, cwd=repo, stdout=handle, check=True)
    python = str(Path(sys.executable).absolute())
    environment = {
        "python": sys.version, "executable": python, "platform": platform.platform(),
        "distributions": sorted([
            {"name": distribution.metadata.get("Name", "unknown"), "version": distribution.version,
             "path": str(distribution.locate_file(""))}
            for distribution in importlib.metadata.distributions()
        ], key=lambda item: (item["name"].lower(), item["path"])),
    }
    (campaign / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    train = [python, "-c", "from curator.commands.hydra_entrypoints import train; train()"]
    jobs, evaluations = [], []
    for seed in (4, 17, 42):
        for experiment, variant, dipole_weight in (("short_range", "short_range", 0.0),
                                                    ("acvr_ef", "acvr", 0.0),
                                                    ("acvr_efd", "acvr", 1.0)):
            identifier = f"{experiment}_seed{seed}"
            run = campaign / "experiments" / experiment / f"seed{seed}"
            common = [f"seed={seed}", f"model.representation.variant={variant}",
                      f"dipole_weight={dipole_weight}", "trainer.accelerator=gpu", "trainer.devices=1"]
            jobs.append({"id": identifier, "kind": "physical_train", "experiment": experiment, "seed": seed,
                         "argv": train + ["--config-name", "acvr_spice", f"run_path={run}",
                                           f"trainer.max_epochs={args.physical_epochs}"] + common,
                         "cwd": str(source), "run_path": str(run), "depends_on": [], "max_retries": 1,
                         "resume_checkpoint": str(run / "model_path/last.ckpt"),
                         "resume_argv": ["model_path={path:'{checkpoint}',mode:resume}"],
                         "success_artifacts": [str(run / "model_path/last.ckpt")]})
            evaluation_id = identifier + "_evaluate"
            jobs.append({"id": evaluation_id, "kind": "evaluate", "experiment": experiment, "seed": seed, "stage": "physical",
                         "argv": [python, str(source / "example/acvr/evaluate_spice_charge.py"),
                                  "--checkpoint", str(run / "model_path"), "--output", str(run / "evaluation"), "--device", "cuda"],
                         "cwd": str(source), "depends_on": [identifier], "max_retries": 0,
                         "success_artifacts": [str(run / "evaluation/metrics.json")]})
            evaluations.append(evaluation_id)
            if variant == "short_range":
                continue
            proposal = run / "proposal"
            proposal_id = identifier + "_proposal"
            jobs.append({"id": proposal_id, "kind": "proposal_train", "experiment": experiment, "seed": seed,
                         "argv": train + ["--config-name", "acvr_spice_proposal", f"run_path={proposal}",
                                           f"model_path.path={run / 'model_path'}",
                                           f"trainer.max_epochs={args.proposal_epochs}"] + common,
                         "cwd": str(source), "run_path": str(proposal), "depends_on": [evaluation_id], "max_retries": 1,
                         "resume_checkpoint": str(proposal / "model_path/last.ckpt"),
                         "resume_argv": ["model_path={path:'{checkpoint}',mode:resume}"],
                         "success_artifacts": [str(proposal / "model_path/last.ckpt")]})
            evaluation_id = proposal_id + "_evaluate"
            jobs.append({"id": evaluation_id, "kind": "evaluate", "experiment": experiment, "seed": seed, "stage": "proposal",
                         "argv": [python, str(source / "example/acvr/evaluate_spice_charge.py"),
                                  "--checkpoint", str(proposal / "model_path"), "--output", str(proposal / "evaluation"),
                                  "--reference-checkpoint", str(run / "model_path"), "--device", "cuda"],
                         "cwd": str(source), "depends_on": [proposal_id], "max_retries": 0,
                         "success_artifacts": [str(proposal / "evaluation/metrics.json")]})
            evaluations.append(evaluation_id)
    jobs.append({"id": "summary", "kind": "summary", "cwd": str(source), "depends_on": evaluations,
                 "argv": [python, str(source / "example/acvr/summarize_spice_campaign.py"), "--campaign", str(campaign)],
                 "max_retries": 0, "success_artifacts": [str(campaign / "summary.json")]})
    manifest = {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "source_repository": str(repo), "source_snapshot": str(source),
                "source_hashes": str(campaign / "source_hashes.json"),
                "environment_manifest": str(campaign / "environment.json"),
                "experiment": {"dataset": "SPICE Ion Pairs", "model_seeds": [4, 17, 42], "data_seed": 17,
                               "physical_epochs": args.physical_epochs, "proposal_epochs": args.proposal_epochs,
                               "scheduler": "none; fixed Adam lr 0.003",
                               "physical_loss_scales": {"energy_eV": 1.0, "force_eV_A": 1.0, "dipole_eA": 1.0},
                               "selection": "Best validation loss within each supervision setting; test never selects a checkpoint"},
                "env": {"PYTHONPATH": str(source), "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, "jobs": jobs}
    (campaign / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"campaign": str(campaign), "jobs": len(jobs), "source_files": len(hashes)}, indent=2))


if __name__ == "__main__":
    main()
