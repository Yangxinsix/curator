from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import curator_mcp.server as mcp_server
from curator.simulate.backends.ase import AseSimulationBackend
from curator.simulate.backends.base import StressCapabilityRequired
from curator.simulate.backends.lammps import LammpsSimulationBackend
from curator.simulate.backends.torchsim import TorchSimSimulationBackend
from curator.simulate.results import case_result_from_completed
from curator.simulate.simulation import list_simulation_engines, run_simulation
from curator.simulate.spec import SimulationCaseSpec, SimulationRequest
from curator.simulate.validation import validate_simulation_result


def _request(**overrides):
    payload = {
        "task": {},
        "system": {"init_structures": ["init.traj"], "start_indices": [0]},
        "model": {"model_path": ["model.ckpt"], "uncertainty_method": "none"},
        "protocol": {"preset": "short_md_smoke_v1"},
        "backend_policy": {"mode": "ase", "fallback": "none"},
    }
    payload.update(overrides)
    return SimulationRequest.from_inputs(**payload)


def test_short_md_is_only_a_protocol_preset():
    engines = list_simulation_engines()

    assert not hasattr(mcp_server, "run_short_md")
    assert "short_md_smoke_v1" in engines["protocol_presets"]
    assert engines["protocol_presets"]["short_md_smoke_v1"]["task"]["criteria_profile"] == "md_smoke_v1"


def test_direct_use_probe_preset_sets_validation_profile():
    request = _request(protocol={"preset": "md_direct_use_probe_v1", "steps": 5})

    assert request.task_type == "md"
    assert request.criteria_profile == "md_direct_use_validation_v1"
    assert request.protocol["preset"] == "md_direct_use_probe_v1"
    assert request.protocol["steps"] == 5
    assert request.ensembles() == ["nve", "nvt"]
    assert request.temperatures() == [300.0, 600.0, 900.0]


def test_npt_requires_explicit_stress_capability():
    request = _request(protocol={"ensembles": ["npt"], "steps": 1})

    with pytest.raises(StressCapabilityRequired):
        AseSimulationBackend().validate_request(request)


def test_npt_accepts_declared_stress_capability():
    request = _request(
        model={
            "model_path": ["model.ckpt"],
            "uncertainty_method": "none",
            "outputs": ["energy", "forces", "stress"],
        },
        protocol={"ensembles": ["npt"], "steps": 1},
    )

    AseSimulationBackend().validate_request(request)


def test_torchsim_batch_plan_is_first_class(monkeypatch):
    monkeypatch.setattr(TorchSimSimulationBackend, "validate_request", lambda self, request: None)
    request = _request(
        system={"init_structures": ["a.traj", "b.traj"], "start_indices": [0, 1], "batch": True},
        protocol={"ensembles": ["nve"], "temperature_K": [300], "steps": 2},
        backend_policy={"mode": "torchsim", "fallback": "none", "batch": True},
    )

    cases = TorchSimSimulationBackend().plan_cases(request)

    assert len(cases) == 1
    assert cases[0].backend == "torchsim"
    assert cases[0].batch is True
    assert cases[0].init_structures == ["a.traj", "b.traj"]


def test_lammps_backend_is_explicitly_not_implemented():
    request = _request(
        task={"type": "production_md"},
        backend_policy={"mode": "lammps", "fallback": "none"},
    )

    with pytest.raises(NotImplementedError):
        LammpsSimulationBackend().validate_request(request)


def test_mahalanobis_direct_use_guard_blocks_non_independent_reference(tmp_path):
    init = tmp_path / "init.traj"
    init.write_text("", encoding="utf-8")

    result = run_simulation(
        task={"type": "md", "criteria_profile": "md_direct_use_validation_v1"},
        system={"init_structures": [str(init)]},
        model={
            "model_path": ["model.ckpt"],
            "uncertainty_method": "mahalanobis",
            "reference_dataset": str(init),
            "reference_dataset_role": "production_trajectory",
        },
        protocol={"ensembles": ["nve"], "steps": 1},
        backend_policy={"mode": "ase", "fallback": "none"},
        out=str(tmp_path / "run"),
    )

    assert result["ok"] is False
    assert result["error_info"]["type"] == "ReferenceDatasetNotIndependent"
    assert result["decision"]["can_use_directly"] is False


def test_validate_simulation_result_pass_and_fail():
    passed = validate_simulation_result(
        result={
            "uncertainty_evidence": {"method": "none", "status": "not_used"},
            "cases": [
                {
                    "ok": True,
                    "steps_completed": 10,
                    "warning_steps": 0,
                    "outlier_steps": 0,
                    "case": {"label": "nve_300K_s0", "ensemble": "nve"},
                    "drift": {"etot_eV_per_atom_per_ps": 0.001},
                    "force": {"max_force_eV_A": {"max": 1.0}},
                    "structure": {"min_distance_A": {"min": 2.0}},
                }
            ],
        },
        criteria_profile="md_direct_use_validation_v1",
    )
    failed = validate_simulation_result(
        result={
            "uncertainty_evidence": {"method": "none", "status": "not_used"},
            "cases": [
                {
                    "ok": True,
                    "steps_completed": 10,
                    "warning_steps": 0,
                    "outlier_steps": 0,
                    "case": {"label": "nve_300K_s0", "ensemble": "nve"},
                    "drift": {"etot_eV_per_atom_per_ps": 1.0},
                    "force": {"max_force_eV_A": {"max": 1.0}},
                    "structure": {"min_distance_A": {"min": 2.0}},
                }
            ],
        },
        criteria_profile="md_direct_use_validation_v1",
    )

    assert passed["decision"]["can_use_directly"] is True
    assert failed["decision"]["can_use_directly"] is False


def test_case_result_reports_performance_metrics(tmp_path):
    summary_path = tmp_path / "simulation_summary.json"
    summary_path.write_text(
        """
{
  "steps_completed": 10,
  "walltime_sec": 2.0,
  "drift": {"elapsed_ps": 0.005},
  "structure": {"natoms": {"last": 4}},
  "performance": {"simulation_walltime_sec": 2.0, "steps_per_second": 5.0}
}
""".strip(),
        encoding="utf-8",
    )
    case = SimulationCaseSpec(
        case_id=0,
        label="nve_300K_s0",
        backend="ase",
        task_type="md",
        criteria_profile="md_smoke_v1",
        init_structures=["init.traj"],
        start_indices=[0],
        structure_indices=[0],
        ensemble="nve",
        temperature_K=300.0,
        pressure_GPa=None,
        steps=10,
        timestep_fs=0.5,
        integrator="verlet",
        seed=1234,
        run_dir=tmp_path,
        timeout_sec=60,
        model_paths=["model.ckpt"],
        reference_dataset=None,
        uncertainty_method="none",
        device="cpu",
        dtype=None,
    )

    result = case_result_from_completed(
        case=case,
        completed=subprocess.CompletedProcess(args=["simulate"], returncode=0),
        artifacts={
            "summary": str(summary_path),
            "subprocess_walltime_sec": 4.0,
        },
        manifest_filename="simulation_manifest.json",
        tool_name="run_simulation",
    )

    assert result["performance"]["simulation_walltime_sec"] == 2.0
    assert result["performance"]["subprocess_walltime_sec"] == 4.0
    assert result["performance"]["steps_per_second"] == 5.0
    assert result["performance"]["atom_steps"] == 40.0
    assert result["performance"]["atom_steps_per_second"] == 20.0
    assert result["performance"]["subprocess_steps_per_second"] == 2.5
