from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from curator.utils import error_result, path_status, read_json, utc_now, write_json, write_text

from .spec import SimulationCaseSpec


def artifact_status(artifacts: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "trajectory": path_status(artifacts.get("trajectory")),
        "warning_structures": path_status(artifacts.get("warning_structures")),
        "summary": path_status(artifacts.get("summary")),
        "log": path_status(artifacts.get("log")),
    }


def read_summary(artifacts: Mapping[str, Any]) -> Dict[str, Any]:
    summary_path = artifacts.get("summary")
    if summary_path and Path(str(summary_path)).exists():
        return read_json(str(summary_path))
    return {}


def case_performance(case: SimulationCaseSpec, summary: Mapping[str, Any], artifacts: Mapping[str, Any]) -> Dict[str, Any]:
    steps_completed = _optional_float(summary.get("steps_completed"))
    simulation_walltime_sec = _optional_float(
        (summary.get("performance") or {}).get("simulation_walltime_sec")
        if isinstance(summary.get("performance"), Mapping)
        else None
    )
    if simulation_walltime_sec is None:
        simulation_walltime_sec = _optional_float(summary.get("walltime_sec"))
    subprocess_walltime_sec = _optional_float(artifacts.get("subprocess_walltime_sec"))
    natoms = _metric_value(summary.get("structure", {}), "natoms")
    atom_steps = steps_completed * natoms if steps_completed is not None and natoms is not None else None
    simulated_time_ps = _metric_value(summary.get("drift", {}), "elapsed_ps")
    if simulated_time_ps is None and steps_completed is not None:
        simulated_time_ps = steps_completed * float(case.timestep_fs) / 1000.0
    return {
        "simulation_walltime_sec": simulation_walltime_sec,
        "subprocess_walltime_sec": subprocess_walltime_sec,
        "steps_completed": steps_completed,
        "natoms": natoms,
        "atom_steps": atom_steps,
        "simulated_time_ps": simulated_time_ps,
        "steps_per_second": _rate(steps_completed, simulation_walltime_sec),
        "atom_steps_per_second": _rate(atom_steps, simulation_walltime_sec),
        "subprocess_steps_per_second": _rate(steps_completed, subprocess_walltime_sec),
        "subprocess_atom_steps_per_second": _rate(atom_steps, subprocess_walltime_sec),
    }


def classify_simulation_failure(stderr_path: Optional[str], *, backend: str) -> Tuple[str, str]:
    if not stderr_path:
        return "SimulationFailed", f"{backend} simulation subprocess failed."
    try:
        stderr = Path(stderr_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return "SimulationFailed", f"{backend} simulation subprocess failed."
    lowered = stderr.lower()
    if "torch-sim is not installed" in stderr or "no module named 'torch_sim'" in lowered:
        return "BackendDependencyMissing", "TorchSim is not installed in the execution environment."
    if (
        "stress not present in this calculation" in stderr
        or "PropertyNotImplementedError" in stderr and "stress" in stderr
        or "KeyError: 'stress'" in stderr
    ):
        return (
            "StressNotAvailable",
            f"{backend} NPT/stress-dependent simulation case failed because the model did not provide stress.",
        )
    if "nan" in lowered:
        return "NumericalInstability", f"{backend} simulation subprocess failed with NaN/numerical instability evidence."
    return "SimulationFailed", f"{backend} simulation subprocess failed."


def case_result_from_completed(
    *,
    case: SimulationCaseSpec,
    completed: subprocess.CompletedProcess[str],
    artifacts: Dict[str, Any],
    manifest_filename: str,
    tool_name: str,
) -> Dict[str, Any]:
    summary = read_summary(artifacts)
    performance = case_performance(case, summary, artifacts)
    manifest = {
        "created_at": utc_now(),
        "tool": tool_name,
        "backend": case.backend,
        "ok": completed.returncode == 0,
        "status": "completed" if completed.returncode == 0 else "failed",
        "model_path": list(case.model_paths),
        "init_structures": list(case.init_structures),
        "start_indices": list(case.start_indices),
        "reference_dataset": case.reference_dataset,
        "uncertainty_method": case.uncertainty_method,
        "case": case.case_metadata(),
        "command": artifacts.get("command"),
        "returncode": completed.returncode,
        "artifacts": artifacts,
        "summary": summary,
        "performance": performance,
        "artifact_status": artifact_status(artifacts),
    }
    artifacts["manifest"] = str(write_json(case.run_dir / manifest_filename, manifest, sort_keys=True, atomic=True))

    if completed.returncode != 0:
        error_type, message = classify_simulation_failure(artifacts.get("stderr"), backend=case.backend)
        return error_result(
            error_type,
            f"{message} Return code: {completed.returncode}.",
            log_path=artifacts["stderr"],
            artifacts=artifacts,
            backend=case.backend,
            uncertainty_method=case.uncertainty_method,
            returncode=completed.returncode,
            summary=summary,
            performance=performance,
            case=case.case_metadata(),
        )

    return {
        "ok": True,
        "status": "completed",
        "backend": case.backend,
        "uncertainty_method": case.uncertainty_method,
        "steps_requested": int(case.steps),
        "steps_completed": summary.get("steps_completed"),
        "early_stopped": bool(summary.get("early_stop_reason")),
        "early_stop_reason": summary.get("early_stop_reason"),
        "max_uncertainty": summary.get("max_uncertainty", {}),
        "warning_steps": summary.get("warning_steps", 0),
        "outlier_steps": summary.get("outlier_steps", 0),
        "drift": summary.get("drift", {}),
        "thermo": summary.get("thermo", {}),
        "force": summary.get("force", {}),
        "structure": summary.get("structure", {}),
        "performance": performance,
        "summary": summary,
        "case": case.case_metadata(),
        "artifacts": artifacts,
        "returncode": completed.returncode,
    }


def timeout_result(
    *,
    case: SimulationCaseSpec,
    exc: subprocess.TimeoutExpired,
    artifacts: Dict[str, Any],
) -> Dict[str, Any]:
    artifacts["stdout"] = str(write_text(case.run_dir / "simulate_stdout.txt", exc.stdout or ""))
    artifacts["stderr"] = str(write_text(case.run_dir / "simulate_stderr.txt", exc.stderr or ""))
    return error_result(
        "Timeout",
        f"{case.backend} simulation exceeded timeout_sec.",
        log_path=artifacts.get("stderr"),
        artifacts=artifacts,
        backend=case.backend,
        case=case.case_metadata(),
        recoverable=True,
    )


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_value(group: Any, name: str) -> Optional[float]:
    if not isinstance(group, Mapping):
        return None
    item = group.get(name)
    if isinstance(item, Mapping):
        for key in ("last", "max", "first"):
            value = _optional_float(item.get(key))
            if value is not None:
                return value
        return None
    return _optional_float(item)


def _rate(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return float(numerator) / float(denominator)
