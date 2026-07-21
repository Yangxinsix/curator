from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from curator.simulate.simulation import (
    list_simulation_engines as core_list_simulation_engines,
    run_simulation as core_run_simulation,
)
from curator.simulate.validation import validate_simulation_result as core_validate_simulation_result


def list_simulation_engines() -> Dict[str, Any]:
    """List task-aware simulation backends and direct-use criteria profiles."""

    return core_list_simulation_engines()


def validate_simulation_result(
    result: Optional[Mapping[str, Any]] = None,
    summary_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
    decision_policy: Optional[Mapping[str, Any]] = None,
    task_type: str = "md",
    criteria_profile: str = "md_direct_use_validation_v1",
) -> Dict[str, Any]:
    """Validate structured simulation evidence against core direct-use criteria."""

    return core_validate_simulation_result(
        result=result,
        summary_path=summary_path,
        manifest_path=manifest_path,
        decision_policy=decision_policy,
        task_type=task_type,
        criteria_profile=criteria_profile,
    )


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
    """Run task-aware simulation cases through the core Curator simulation API."""

    return core_run_simulation(
        task=task,
        system=system,
        model=model,
        protocol=protocol,
        backend_policy=backend_policy,
        decision_policy=decision_policy,
        out=out,
        timeout_sec=timeout_sec,
    )
