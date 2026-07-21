from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from curator.utils import read_json


ALLOWED_UNCERTAINTY = {"auto", "none", "ensemble", "mahalanobis"}
DEFAULT_CRITERIA_PROFILE = "md_direct_use_validation_v1"
INDEPENDENT_REFERENCE_ROLES = {"training_reference", "calibration_reference", "dft_audit_reference"}
_DIRECT_USE_VALIDATION_POLICY = {
    "max_uncertainty_violation_rate": 0.0,
    "max_outlier_fraction": 0.0,
    "max_nve_energy_drift_eV_per_atom_ps": 0.01,
    "min_distance_A": 1.0,
    "max_force_eV_A": 20.0,
    "fail_on_early_stop": True,
    "fail_on_failed_case": True,
    "fail_on_skipped": True,
}
CRITERIA_PROFILES: Dict[str, Dict[str, Any]] = {
    "md_direct_use_validation_v1": dict(_DIRECT_USE_VALIDATION_POLICY),
    "md_smoke_v1": {
        "max_uncertainty_violation_rate": 1.0,
        "max_outlier_fraction": 1.0,
        "min_distance_A": 0.7,
        "max_force_eV_A": 100.0,
        "fail_on_early_stop": True,
        "fail_on_failed_case": True,
        "fail_on_skipped": False,
    },
}


def resolve_uncertainty_method(method: str, model_paths: List[str], reference_dataset: Optional[str]) -> str:
    normalized = str(method or "auto").strip().lower()
    if normalized not in ALLOWED_UNCERTAINTY:
        raise ValueError(f"uncertainty_method must be one of {sorted(ALLOWED_UNCERTAINTY)}, got {method!r}.")
    if normalized != "auto":
        return normalized
    if len(model_paths) > 1:
        return "ensemble"
    if reference_dataset:
        return "mahalanobis"
    return "none"


def criteria_policy(criteria_profile: str) -> Dict[str, Any]:
    profile = str(criteria_profile or DEFAULT_CRITERIA_PROFILE)
    if profile not in CRITERIA_PROFILES:
        raise ValueError(f"criteria_profile must be one of {sorted(CRITERIA_PROFILES)}, got {profile!r}.")
    return dict(CRITERIA_PROFILES[profile])


def merge_decision_policy(criteria_profile: str, decision_policy: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    policy = criteria_policy(criteria_profile)
    if decision_policy is None:
        return policy
    if not isinstance(decision_policy, Mapping):
        raise TypeError("decision_policy must be a JSON object.")
    policy.update(dict(decision_policy))
    return policy


def _same_path(left: Any, right: Any) -> bool:
    try:
        return Path(str(left)).expanduser().resolve() == Path(str(right)).expanduser().resolve()
    except Exception:
        return str(left) == str(right)


def uncertainty_evidence_guard(
    *,
    criteria_profile: str,
    uncertainty_method: str,
    model_paths: List[str],
    reference_dataset: Optional[Any],
    reference_dataset_role: str,
    init_structures: Sequence[Any],
) -> Dict[str, Any]:
    try:
        resolved_method = resolve_uncertainty_method(
            uncertainty_method,
            model_paths,
            str(reference_dataset) if reference_dataset else None,
        )
    except Exception:
        resolved_method = str(uncertainty_method or "auto")

    evidence = {
        "method": resolved_method,
        "reference_dataset": str(reference_dataset) if reference_dataset else None,
        "reference_dataset_role": reference_dataset_role,
        "status": "not_used",
        "independent_reference": None,
        "issues": [],
    }
    if resolved_method != "mahalanobis":
        return evidence

    role = str(reference_dataset_role or "unknown").strip().lower()
    same_as_init = bool(reference_dataset) and any(_same_path(reference_dataset, item) for item in init_structures)
    independent = role in INDEPENDENT_REFERENCE_ROLES and not same_as_init
    evidence["reference_dataset_role"] = role
    evidence["independent_reference"] = independent

    if not reference_dataset:
        evidence["status"] = "invalid"
        evidence["issues"].append("mahalanobis uncertainty requires reference_dataset")
    if role not in INDEPENDENT_REFERENCE_ROLES:
        evidence["issues"].append(
            f"reference_dataset_role={role!r} is not an independent role; expected one of {sorted(INDEPENDENT_REFERENCE_ROLES)}"
        )
    if same_as_init:
        evidence["issues"].append("reference_dataset is identical to one of system.init_structures")

    if evidence["issues"]:
        evidence["status"] = "smoke_only" if criteria_profile == "md_smoke_v1" else "invalid"
    else:
        evidence["status"] = "independent_reference"
    return evidence


def uncertainty_guard_error(evidence: Mapping[str, Any], criteria_profile: str) -> Optional[Dict[str, Any]]:
    if evidence.get("method") != "mahalanobis":
        return None
    if criteria_profile == "md_smoke_v1":
        return None
    if evidence.get("status") == "independent_reference":
        return None
    issues = [str(item) for item in evidence.get("issues", [])]
    return {
        "ok": False,
        "status": "blocked",
        "error_info": {
            "type": "ReferenceDatasetNotIndependent",
            "message": "; ".join(issues) or "Mahalanobis reference dataset is not valid for direct-use validation.",
            "recoverable": True,
            "log_path": None,
        },
        "decision": {
            "can_use_directly": False,
            "criteria_profile": criteria_profile,
            "task_type": "md",
            "recommended_next_stage": "provide_independent_reference_dataset",
            "reasons": ["uncertainty reference dataset is not independent"],
            "violations": [
                {
                    "case": {"uncertainty_evidence": dict(evidence)},
                    "label": "uncertainty_reference",
                    "reasons": issues or ["Mahalanobis reference dataset is not valid for direct-use validation."],
                }
            ],
        },
        "uncertainty_evidence": dict(evidence),
    }


def validate_simulation_result(
    result: Optional[Mapping[str, Any]] = None,
    summary_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
    decision_policy: Optional[Mapping[str, Any]] = None,
    task_type: str = "md",
    criteria_profile: str = DEFAULT_CRITERIA_PROFILE,
) -> Dict[str, Any]:
    """Apply hard direct-use rules to structured simulation evidence."""

    policy = merge_decision_policy(criteria_profile, decision_policy)
    evidence: Dict[str, Any] = dict(result or {})

    if manifest_path:
        manifest = read_json(manifest_path)
        evidence = manifest if not evidence else {**manifest, **evidence}
    if summary_path:
        evidence.setdefault("summary", read_json(summary_path))

    case_items = evidence.get("cases", [])
    cases = [dict(item) for item in case_items if isinstance(item, Mapping)]
    skipped = [dict(item) for item in evidence.get("skipped", []) if isinstance(item, Mapping)]
    if not cases and "summary" in evidence:
        cases = [evidence]

    max_warning_fraction = float(
        policy.get("max_warning_fraction", policy.get("max_uncertainty_violation_rate", 0.0))
    )
    max_outlier_fraction = float(policy.get("max_outlier_fraction", 0.0))
    max_uncertainty = dict(policy.get("max_uncertainty", policy.get("max_uncertainty_thresholds", {})) or {})
    max_nve_energy_drift = policy.get("max_nve_energy_drift_eV_per_atom_ps")
    min_distance_threshold = policy.get("min_distance_A")
    max_force_threshold = policy.get("max_force_eV_A")
    max_temperature_deviation_fraction = policy.get("max_temperature_deviation_fraction")
    fail_on_early_stop = bool(policy.get("fail_on_early_stop", True))
    fail_on_failed_case = bool(policy.get("fail_on_failed_case", True))
    fail_on_skipped = bool(policy.get("fail_on_skipped", True))

    violations: List[Dict[str, Any]] = []
    reliable_for: List[str] = []
    not_reliable_for: List[str] = []
    total_steps = 0
    total_warning_steps = 0
    total_outlier_steps = 0
    max_abs_nve_energy_drift: Optional[float] = None
    min_distance_seen: Optional[float] = None
    max_force_seen: Optional[float] = None

    for item in cases:
        case = dict(item.get("case", {}) or {})
        label = str(case.get("label") or case.get("ensemble") or case.get("temperature_K") or "case")
        summary = dict(item.get("summary", {}) or {})
        steps = int(item.get("steps_completed") or summary.get("steps_completed") or 0)
        warnings = int(item.get("warning_steps") or summary.get("warning_steps") or 0)
        outliers = int(item.get("outlier_steps") or summary.get("outlier_steps") or 0)
        max_unc = dict(item.get("max_uncertainty") or summary.get("max_uncertainty") or {})
        total_steps += max(steps, 0)
        total_warning_steps += max(warnings, 0)
        total_outlier_steps += max(outliers, 0)

        case_violations: List[str] = []
        if fail_on_failed_case and not bool(item.get("ok", False)):
            error_info = dict(item.get("error_info") or {})
            if error_info:
                case_violations.append(
                    f"{error_info.get('type', 'SimulationFailed')}: {error_info.get('message', 'simulation case execution failed')}"
                )
            else:
                case_violations.append("simulation case execution failed")
        early_stop_reason = item.get("early_stop_reason") or summary.get("early_stop_reason")
        if fail_on_early_stop and early_stop_reason:
            case_violations.append(f"early stop: {early_stop_reason}")
        warning_fraction = warnings / steps if steps else 0.0
        outlier_fraction = outliers / steps if steps else 0.0
        drift = dict(item.get("drift") or summary.get("drift") or {})
        thermo = dict(item.get("thermo") or summary.get("thermo") or {})
        force = dict(item.get("force") or summary.get("force") or {})
        structure = dict(item.get("structure") or summary.get("structure") or {})
        if warning_fraction > max_warning_fraction:
            case_violations.append(
                f"warning_fraction={warning_fraction:.6g} exceeds {max_warning_fraction:.6g}"
            )
        if outlier_fraction > max_outlier_fraction:
            case_violations.append(
                f"outlier_fraction={outlier_fraction:.6g} exceeds {max_outlier_fraction:.6g}"
            )
        for key, threshold in max_uncertainty.items():
            if key in max_unc and float(max_unc[key]) > float(threshold):
                case_violations.append(f"max_uncertainty[{key}]={float(max_unc[key]):.6g} exceeds {float(threshold):.6g}")
        energy_drift = drift.get("etot_eV_per_atom_per_ps")
        if energy_drift is not None and str(case.get("ensemble", "")).lower() == "nve":
            abs_drift = abs(float(energy_drift))
            max_abs_nve_energy_drift = abs_drift if max_abs_nve_energy_drift is None else max(max_abs_nve_energy_drift, abs_drift)
            if max_nve_energy_drift is not None and abs_drift > float(max_nve_energy_drift):
                case_violations.append(
                    f"abs_nve_energy_drift={abs_drift:.6g} eV/atom/ps exceeds {float(max_nve_energy_drift):.6g}"
                )
        min_distance = (structure.get("min_distance_A") or {}).get("min")
        if min_distance is not None:
            min_distance_seen = float(min_distance) if min_distance_seen is None else min(min_distance_seen, float(min_distance))
            if min_distance_threshold is not None and float(min_distance) < float(min_distance_threshold):
                case_violations.append(
                    f"min_distance_A={float(min_distance):.6g} below {float(min_distance_threshold):.6g}"
                )
        max_force = (force.get("max_force_eV_A") or {}).get("max")
        if max_force is not None:
            max_force_seen = float(max_force) if max_force_seen is None else max(max_force_seen, float(max_force))
            if max_force_threshold is not None and float(max_force) > float(max_force_threshold):
                case_violations.append(
                    f"max_force_eV_A={float(max_force):.6g} exceeds {float(max_force_threshold):.6g}"
                )
        if max_temperature_deviation_fraction is not None and "temperature_K" in thermo and case.get("temperature_K"):
            target_temp = float(case["temperature_K"])
            temp_item = thermo["temperature_K"]
            max_delta = max(
                abs(float(temp_item.get("min", target_temp)) - target_temp),
                abs(float(temp_item.get("max", target_temp)) - target_temp),
            )
            fraction = max_delta / target_temp if target_temp > 0 else 0.0
            if fraction > float(max_temperature_deviation_fraction):
                case_violations.append(
                    f"temperature_deviation_fraction={fraction:.6g} exceeds {float(max_temperature_deviation_fraction):.6g}"
                )

        if case_violations:
            violations.append({"case": case, "label": label, "reasons": case_violations})
            not_reliable_for.append(label)
        else:
            reliable_for.append(label)

    for item in skipped:
        label = str(item.get("label") or item.get("ensemble") or "skipped_case")
        if fail_on_skipped:
            violations.append({"case": item, "label": label, "reasons": [str(item.get("reason") or "simulation case skipped")]})
            not_reliable_for.append(label)

    if not cases:
        violations.append({"case": {}, "label": "simulation", "reasons": ["no completed simulation case evidence"]})

    uncertainty_evidence = dict(evidence.get("uncertainty_evidence") or {})
    if (
        uncertainty_evidence.get("method") == "mahalanobis"
        and criteria_profile != "md_smoke_v1"
        and uncertainty_evidence.get("status") != "independent_reference"
    ):
        reasons = [str(item) for item in uncertainty_evidence.get("issues", [])]
        if not reasons:
            reasons = ["mahalanobis reference dataset is not proven independent"]
        violations.append(
            {
                "case": {"uncertainty_evidence": uncertainty_evidence},
                "label": "uncertainty_reference",
                "reasons": reasons,
            }
        )
        not_reliable_for.append("uncertainty_reference")

    can_use_directly = not violations
    decision = {
        "can_use_directly": can_use_directly,
        "criteria_profile": criteria_profile,
        "task_type": task_type,
        "recommended_next_stage": "deploy_or_production_validation" if can_use_directly else "active_learning",
        "reasons": [] if can_use_directly else ["direct-use criteria violated"],
        "violations": violations,
    }

    return {
        "ok": True,
        "status": "completed",
        "decision": decision,
        "criteria_policy": policy,
        "uncertainty_evidence": uncertainty_evidence,
        "reliable_for": sorted(set(reliable_for)),
        "not_reliable_for": sorted(set(not_reliable_for)),
        "metrics": {
            "num_cases": len(cases),
            "num_skipped": len(skipped),
            "total_steps_completed": total_steps,
            "total_warning_steps": total_warning_steps,
            "total_outlier_steps": total_outlier_steps,
            "warning_fraction": total_warning_steps / total_steps if total_steps else None,
            "outlier_fraction": total_outlier_steps / total_steps if total_steps else None,
            "max_abs_nve_energy_drift_eV_per_atom_ps": max_abs_nve_energy_drift,
            "min_distance_A": min_distance_seen,
            "max_force_eV_A": max_force_seen,
        },
    }
