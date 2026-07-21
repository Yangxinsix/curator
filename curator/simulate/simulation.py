from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from curator.utils import error_result, utc_now, write_json

from .backends import list_backend_capabilities, select_backend
from .spec import PROTOCOL_PRESETS, SimulationRequest
from .validation import (
    CRITERIA_PROFILES,
    DEFAULT_CRITERIA_PROFILE,
    uncertainty_evidence_guard,
    uncertainty_guard_error,
    validate_simulation_result,
)


def list_simulation_engines() -> Dict[str, Any]:
    """Return task-aware simulation backend capabilities known to Curator."""

    return {
        "ok": True,
        "status": "completed",
        "default_backend": "ase",
        "criteria_profiles": sorted(CRITERIA_PROFILES),
        "criteria_policies": {key: dict(value) for key, value in CRITERIA_PROFILES.items()},
        "protocol_presets": {key: dict(value) for key, value in PROTOCOL_PRESETS.items()},
        "engines": list_backend_capabilities(),
    }


def run_simulation(
    task: Optional[Mapping[str, Any]] = None,
    system: Optional[Mapping[str, Any]] = None,
    model: Optional[Mapping[str, Any]] = None,
    protocol: Optional[Mapping[str, Any]] = None,
    backend_policy: Optional[Mapping[str, Any]] = None,
    decision_policy: Optional[Mapping[str, Any]] = None,
    out: str = "simulation",
    timeout_sec: int = 1800,
) -> Dict[str, Any]:
    """Run task-aware local simulation cases and return validator-ready evidence."""

    artifacts: Dict[str, Any] = {}
    try:
        request = SimulationRequest.from_inputs(
            task=task,
            system=system,
            model=model,
            protocol=protocol,
            backend_policy=backend_policy,
            decision_policy=decision_policy,
            out=out,
            timeout_sec=timeout_sec,
        )
        artifacts = {"run_dir": str(request.run_dir)}

        uncertainty_evidence = uncertainty_evidence_guard(
            criteria_profile=request.criteria_profile,
            uncertainty_method=request.uncertainty_method,
            model_paths=request.model_paths,
            reference_dataset=request.reference_dataset,
            reference_dataset_role=request.reference_dataset_role,
            init_structures=request.init_structures,
        )
        guard_error = uncertainty_guard_error(uncertainty_evidence, request.criteria_profile)
        if guard_error is not None:
            guard_error["artifacts"] = artifacts
            guard_error["decision"]["task_type"] = request.task_type
            return guard_error

        backend = select_backend(request.backend_policy)
        backend.validate_request(request)
        cases = backend.plan_cases(request)
        if not cases:
            return error_result(
                "NoRunnableCases",
                "No supported simulation cases were generated from the requested protocol.",
                artifacts=artifacts,
                backend=backend.name,
                recoverable=True,
            )

        case_results = [backend.run_case(case, tool_name="run_simulation") for case in cases]
        execution_ok = all(bool(case.get("ok", False)) for case in case_results)
        result: Dict[str, Any] = {
            "ok": execution_ok,
            "status": "completed" if execution_ok else "completed_with_failures",
            "backend": backend.name,
            "task": {
                "type": request.task_type,
                "criteria_profile": request.criteria_profile,
                "goal": request.task.get("goal"),
            },
            "protocol": _protocol_summary(request),
            "uncertainty_evidence": uncertainty_evidence,
            "cases": case_results,
            "skipped": [],
            "artifacts": artifacts,
        }
        validation = validate_simulation_result(
            result=result,
            decision_policy=request.decision_policy,
            task_type=request.task_type,
            criteria_profile=request.criteria_profile,
        )
        result["decision"] = validation["decision"]
        result["reliable_for"] = validation["reliable_for"]
        result["not_reliable_for"] = validation["not_reliable_for"]
        result["metrics"] = validation["metrics"]
        result["performance"] = _performance_summary(case_results)
        result["criteria_policy"] = validation["criteria_policy"]

        manifest = {
            "created_at": utc_now(),
            "tool": "run_simulation",
            "ok": result["ok"],
            "status": result["status"],
            "backend": backend.name,
            "task": result["task"],
            "protocol": result["protocol"],
            "uncertainty_evidence": result["uncertainty_evidence"],
            "decision": result["decision"],
            "criteria_policy": result["criteria_policy"],
            "metrics": result["metrics"],
            "performance": result["performance"],
            "skipped": [],
            "cases": case_results,
            "case_manifests": [case.get("artifacts", {}).get("manifest") for case in case_results],
        }
        artifacts["manifest"] = str(write_json(request.run_dir / "simulation_matrix_manifest.json", manifest, sort_keys=True, atomic=True))
        result["artifacts"] = artifacts
        return result
    except Exception as exc:
        return error_result(type(exc).__name__, str(exc), artifacts=artifacts, recoverable=True)

def _protocol_summary(request: SimulationRequest) -> Dict[str, Any]:
    return {
        "preset": request.protocol.get("preset", DEFAULT_CRITERIA_PROFILE),
        "type": request.protocol.get("type", request.task_type),
        "temperatures_K": request.temperatures(),
        "pressures_GPa": request.pressures(),
        "ensembles": request.ensembles(),
        "steps": request.int_protocol("steps", 1000),
        "timestep_fs": request.float_protocol("timestep_fs", 0.5),
        "thermostat": request.protocol.get("thermostat"),
        "barostat": request.protocol.get("barostat"),
        "initialize_velocities": bool(request.protocol.get("initialize_velocities", False)),
        "force_temperature": bool(request.protocol.get("force_temperature", bool(request.protocol.get("initialize_velocities", False)))),
        "remove_translation": bool(request.protocol.get("remove_translation", True)),
        "remove_rotation": bool(request.protocol.get("remove_rotation", True)),
        "summary_interval": request.int_protocol("summary_interval", 1),
        "summary_compute_min_distance": bool(request.protocol.get("summary_compute_min_distance", True)),
        "summary_compute_forces": bool(request.protocol.get("summary_compute_forces", True)),
        "uncertainty_kernel": str(request.model.get("uncertainty_kernel", request.protocol.get("uncertainty_kernel", "local-full-g"))),
        "uncertainty_max_structures": request.model.get(
            "uncertainty_max_structures",
            request.protocol.get("uncertainty_max_structures"),
        ),
    }


def _performance_summary(case_results: list[Mapping[str, Any]]) -> Dict[str, Any]:
    performances = [dict(case.get("performance") or {}) for case in case_results if isinstance(case, Mapping)]
    total_steps = _sum_optional(item.get("steps_completed") for item in performances)
    total_atom_steps = _sum_optional(item.get("atom_steps") for item in performances)
    total_sim_walltime = _sum_optional(item.get("simulation_walltime_sec") for item in performances)
    total_subprocess_walltime = _sum_optional(item.get("subprocess_walltime_sec") for item in performances)
    total_simulated_time_ps = _sum_optional(item.get("simulated_time_ps") for item in performances)
    return {
        "num_cases": len(performances),
        "total_steps_completed": total_steps,
        "total_atom_steps": total_atom_steps,
        "total_simulated_time_ps": total_simulated_time_ps,
        "total_simulation_walltime_sec": total_sim_walltime,
        "total_subprocess_walltime_sec": total_subprocess_walltime,
        "effective_steps_per_second": _rate(total_steps, total_sim_walltime),
        "effective_atom_steps_per_second": _rate(total_atom_steps, total_sim_walltime),
        "effective_subprocess_steps_per_second": _rate(total_steps, total_subprocess_walltime),
        "effective_subprocess_atom_steps_per_second": _rate(total_atom_steps, total_subprocess_walltime),
        "case_steps_per_second": _series_summary(
            item.get("steps_per_second") for item in performances
        ),
        "case_atom_steps_per_second": _series_summary(
            item.get("atom_steps_per_second") for item in performances
        ),
    }


def _sum_optional(values: Any) -> Optional[float]:
    total = 0.0
    seen = False
    for value in values:
        try:
            if value is None:
                continue
            total += float(value)
            seen = True
        except (TypeError, ValueError):
            continue
    return total if seen else None


def _series_summary(values: Any) -> Dict[str, Optional[float]]:
    cleaned = []
    for value in values:
        try:
            if value is not None:
                cleaned.append(float(value))
        except (TypeError, ValueError):
            pass
    if not cleaned:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(cleaned),
        "max": max(cleaned),
        "mean": sum(cleaned) / len(cleaned),
    }


def _rate(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return float(numerator) / float(denominator)
