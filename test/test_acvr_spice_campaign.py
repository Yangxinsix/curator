"""CPU-only checks of the persisted SPICE campaign graph and result aggregation."""

from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from hydra import compose, initialize_config_dir
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "example/acvr" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepared_campaign_graph_snapshot_and_resume_arguments(tmp_path):
    campaign = tmp_path / "campaign"
    subprocess.run([sys.executable, str(ROOT / "example/acvr/prepare_spice_campaign.py"),
                    "--output", str(campaign), "--physical-epochs", "2", "--proposal-epochs", "3"],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    manifest = json.loads((campaign / "manifest.json").read_text())
    counts = Counter(job["kind"] for job in manifest["jobs"])
    assert counts == {"physical_train": 9, "proposal_train": 6, "evaluate": 15, "summary": 1}
    by_id = {job["id"]: job for job in manifest["jobs"]}
    assert set(by_id["summary"]["depends_on"]) == {job["id"] for job in manifest["jobs"] if job["kind"] == "evaluate"}
    source = campaign / "source"
    assert manifest["env"]["PYTHONPATH"] == str(source)
    assert manifest["experiment"]["scheduler"] == "none; fixed Adam lr 0.003"
    environment = json.loads((campaign / "environment.json").read_text())
    assert environment["python"] and any(item["name"].lower() == "torch" for item in environment["distributions"])
    checked = subprocess.run([sys.executable, "-c", "import curator; print(curator.__file__)"],
                             cwd=source, env=dict(os.environ, **manifest["env"]),
                             check=True, text=True, capture_output=True)
    assert str(source / "curator/__init__.py") in checked.stdout
    for name in ("acvr_ef_seed4", "acvr_ef_seed4_proposal"):
        job = by_id[name]
        location = job["argv"].index("--config-name")
        resume = [value.replace("{checkpoint}", job["resume_checkpoint"]) for value in job["resume_argv"]]
        with initialize_config_dir(config_dir=str(source / "curator/configs"), version_base=None):
            config = compose(config_name=job["argv"][location + 1], overrides=job["argv"][location + 2:] + resume)
        assert config.model_path.mode == "resume"
        assert config.model_path.path == job["resume_checkpoint"]
        assert config.trainer.devices == 1
    hashes = json.loads((campaign / "source_hashes.json").read_text())
    assert all(hashlib.sha256((source / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())


def test_summary_aggregates_seed_statistics_and_rejects_manifest_mismatch(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "code.py").write_text("# fixed source\n")
    hashes = tmp_path / "source_hashes.json"
    hashes.write_text(json.dumps({"code.py": hashlib.sha256((source / "code.py").read_bytes()).hexdigest()}))
    jobs = []
    for index, seed in enumerate((4, 17, 42), 1):
        path = tmp_path / f"seed{seed}.json"
        metrics = {label: {"rmse": float(index)} for label in ("energy", "forces", "dipole")}
        metrics.update(n_geometries=4, proposal={"projected_proposal_state_rmse_e": index / 10,
            "0.001": {"accepted_projected": index, "n_geometries": 4},
            "by_pair_symmetry": {"homonuclear": {"n_geometries": 1}}})
        path.write_text(json.dumps({"schema": "spice_acvr_evaluation_v1", "model_seed": seed,
            "stage": "physical", "variant": "acvr", "checkpoint": f"{seed}.ckpt", "checkpoint_sha256": "a" * 64,
            "dataset_metadata": {"source_sha256": "b" * 64, "splits": {"train": [0], "val": [1], "test": [2]}},
            "units": {"energy": "eV", "forces": "eV/Angstrom", "dipole": "e*Angstrom"},
            "splits": {"val": metrics, "test": metrics}}))
        jobs.append({"id": str(seed), "kind": "evaluate", "seed": seed, "stage": "physical",
                     "experiment": "acvr_ef", "success_artifacts": [str(path)]})
    (tmp_path / "manifest.json").write_text(json.dumps({"jobs": jobs, "source_hashes": str(hashes),
        "source_snapshot": str(source), "environment_manifest": str(tmp_path / "environment.json"),
        "experiment": {"model_seeds": [4, 17, 42]}}))
    module = load_script("summarize_spice_campaign")
    monkeypatch.setattr(sys, "argv", ["summary", "--campaign", str(tmp_path)])
    module.main()
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["aggregate"][0]["energy_rmse_mean"] == 2
    assert report["aggregate"][0]["energy_rmse_std"] == 1
    assert report["aggregate"][0]["proposal_acceptance_fraction_at_0.001_mean"] == 0.5
    bad_path = tmp_path / "seed42.json"
    bad = json.loads(bad_path.read_text())
    bad["dataset_metadata"]["source_sha256"] = "c" * 64
    bad_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="different dataset source/split"):
        module.main()


def test_proposal_audit_separates_symmetry_and_checks_list_alignment():
    from curator.variational import QuadraticProblem

    module = load_script("evaluate_spice_charge")
    problem = QuadraticProblem(torch.eye(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64),
                               torch.ones(1, 2, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
    report = module.audit_proposal([problem, problem], [torch.ones(2, dtype=torch.float64),
        torch.tensor([0.3, -0.3], dtype=torch.float64)], [{"Z0": 11, "Z1": 11}, {"Z0": 11, "Z1": 9}])
    assert report["by_pair_symmetry"]["homonuclear"]["0.001"]["accepted_projected"] == 1
    assert report["by_pair_symmetry"]["heteronuclear"]["0.001"]["accepted_projected"] == 0
    assert "must not be counted as learned" in report["symmetry_note"]
    with pytest.raises(ValueError, match="equally sized nonempty"):
        module.audit_proposal([problem], [])
