"""Completion gates for the public experiment report; no models or GPUs."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/acvr_public_report.py"
SPEC = importlib.util.spec_from_file_location("acvr_public_report", SCRIPT)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def finished(tmp_path):
    baselines = {}
    for count, name in enumerate(REPORT.EXPERIMENTS, 1):
        directory = tmp_path / name
        metadata_hash = write(directory / "data_metadata.json", {"split_statistics": {"test": {"count": count}}})
        label = "dipole" if name == "BaTiO3" else "generic_label"
        term = {"rmse": 2.0, "unit": "physical_unit", "n_geometries": count,
                "n_conditions": count, "status": "active", "coverage": 1.0}
        baselines[name] = {"num_structures": count, "num_conditions": count,
                           "data_metadata_sha256": metadata_hash, "terms": {label: term}}
        write(directory / "execution/run_result.json", {
            "status": "completed", "epoch": 2, "global_step": 12, "data_metadata_sha256": metadata_hash})
        write(directory / "execution/evaluate_result.json", {
            "status": "completed", "test_structures": count, "data_metadata_sha256": metadata_hash,
            "checkpoint_sha256": name + "hash"})
        write(directory / "evaluate/metrics.json", {
            "num_structures": count, "num_conditions": count,
            "terms": {label: dict(term, rmse=3.0 if name == "BaTiO3" else 1.0)}})
    write(tmp_path / "provenance/heldout_zero_baselines.json", {"experiments": baselines})
    write(tmp_path / "BaTiO3/evaluate/dipole_diagnostic.json", {
        "prediction": {"rmse_eA": 3.0, "zero_rmse_eA": 2.0}, "conclusions": ["branch mismatch observed"]})
    (tmp_path / "BaTiO3/evaluate/dipole_diagnostic.md").write_text("Branch diagnostic\n")
    write(tmp_path / "MgO/phonon/comparison.json", {
        "status": "model_reference_comparison", "n_commensurate_qpoints": 32, "nac": "disabled",
        "model": {"checkpoint_sha256": "MgOhash"},
        "holdout_reference_512": {"n_evaluated_qpoints": 32, "split_provenance": {
            "verified": True, "contains_training_labels": False,
            "reference_force_labels": [{"split": "phonon_holdout"}, {"split": "phonon_holdout"}]}},
        "holdout512_comparison": {"asr": {"qpoint_sorted_frequency_rmse_THz": 0.74}}})
    np.savez(tmp_path / "MgO/phonon/comparison.npz",
             model_gamma_asr_THz=[0, 0, 0, 11.2, 11.2, 11.2],
             holdout512_gamma_asr_THz=[0, 0, 0, 11.75, 11.75, 11.75])
    return tmp_path


def command(root, *args):
    return subprocess.run([sys.executable, str(SCRIPT), "--root", str(root), *args],
                          capture_output=True, text=True, timeout=15)


def test_completed_report_uses_physical_metrics_and_discloses_failure(finished):
    result = command(finished)
    assert result.returncode == 0, result.stderr
    summary = json.loads((finished / "results_summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["completed_experiments"] == 5
    assert summary["experiments"]["Si"]["epochs_completed"] == 3
    dipole = next(row for row in summary["metrics"] if row["experiment"] == "BaTiO3")
    assert dipole["comparison"] == "worse_than_zero"
    assert dipole["model_over_zero"] == 1.5
    markdown = (finished / "RESULTS.md").read_text()
    assert "明确失败结果" in markdown
    assert "BaTiO3/evaluate/dipole_diagnostic.md" in markdown
    assert "11.20000" in markdown and "11.75000" in markdown and "无 NAC" in markdown
    assert "generic_label" in (finished / "metrics.csv").read_text()


@pytest.mark.parametrize("problem", ["missing", "count", "failed", "baseline_hash", "phonon_training_labels"])
def test_incomplete_artifacts_cannot_claim_complete(finished, problem):
    if problem == "missing":
        (finished / "Si/evaluate/metrics.json").unlink()
    else:
        paths = {"count": "Si/evaluate/metrics.json", "failed": "Si/execution/run_result.json",
                 "baseline_hash": "Si/execution/evaluate_result.json", "phonon_training_labels": "MgO/phonon/comparison.json"}
        path = finished / paths[problem]
        value = json.loads(path.read_text())
        if problem == "count":
            value["num_structures"] = 99
        elif problem == "failed":
            value["status"] = "failed"
        elif problem == "baseline_hash":
            value["data_metadata_sha256"] = "wrong"
        else:
            value["holdout_reference_512"]["split_provenance"]["contains_training_labels"] = True
        write(path, value)
    result = command(finished)
    assert result.returncode == 1, result.stderr
    assert not (finished / "RESULTS.md").exists()
    assert json.loads(result.stdout)["status"] == "partial"
    result = command(finished, "--allow-partial")
    assert result.returncode == 0, result.stderr
    assert json.loads((finished / "results_summary.json").read_text())["status"] == "partial"


def test_partial_never_overwrites_a_completed_report(finished):
    assert command(finished).returncode == 0
    before = {name: (finished / name).read_bytes() for name in ("RESULTS.md", "results_summary.json", "metrics.csv")}
    (finished / "Si/execution/evaluate_result.json").unlink()
    result = command(finished, "--allow-partial")
    assert result.returncode == 0
    assert json.loads(result.stdout)["status"] == "partial"
    assert json.loads(result.stdout)["written"] is False
    assert all((finished / name).read_bytes() == contents for name, contents in before.items())
