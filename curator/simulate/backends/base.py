from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional

from curator.utils import module_available

from ..spec import SimulationCaseSpec, SimulationRequest


@dataclass(frozen=True)
class SimulationBackendCapabilities:
    backend: str
    available: bool
    implemented: bool
    tasks: List[str]
    ensembles: List[str] = field(default_factory=list)
    integrators: List[str] = field(default_factory=list)
    devices: List[str] = field(default_factory=list)
    dtypes: List[str] = field(default_factory=list)
    batching: bool = False
    autobatching: bool = False
    online_uncertainty: bool = False
    trajectory: bool = True
    summary: bool = True
    requires_deployed_model: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "available": self.available,
            "implemented_in_mcp_runner": self.implemented,
            "supports": {
                "task_types": list(self.tasks),
                "ensembles": list(self.ensembles),
                "integrators": list(self.integrators),
                "devices": list(self.devices),
                "dtypes": list(self.dtypes),
                "batching": self.batching,
                "autobatching": self.autobatching,
                "online_uncertainty": self.online_uncertainty,
                "trajectory": self.trajectory,
                "summary": self.summary,
                "requires_deployed_model": self.requires_deployed_model,
            },
            "notes": list(self.notes),
        }


class SimulationBackend(ABC):
    name: str

    @abstractmethod
    def capabilities(self) -> SimulationBackendCapabilities:
        ...

    def available(self) -> bool:
        return self.capabilities().available

    def validate_request(self, request: SimulationRequest) -> None:
        caps = self.capabilities()
        if not caps.implemented:
            raise NotImplementedError(f"Simulation backend {self.name!r} is not implemented.")
        if not caps.available:
            raise RuntimeError(f"Simulation backend {self.name!r} is not available in this environment.")
        if request.task_type not in caps.tasks:
            raise ValueError(
                f"Simulation backend {self.name!r} does not support task.type={request.task_type!r}; "
                f"supported tasks are {caps.tasks!r}."
            )
        if _request_requires_stress(request) and _declared_stress_capability(request.model) is not True:
            raise StressCapabilityRequired(
                "NPT or stress-dependent simulation requires model.stress_capable=True, "
                "model.supports_stress=True, model.capabilities.stress=True, or model.outputs containing 'stress'."
            )

    @abstractmethod
    def plan_cases(self, request: SimulationRequest) -> List[SimulationCaseSpec]:
        ...

    @abstractmethod
    def build_config(self, case: SimulationCaseSpec) -> Dict[str, Any]:
        ...

    @abstractmethod
    def run_case(self, case: SimulationCaseSpec, *, tool_name: str = "run_simulation") -> Dict[str, Any]:
        ...


def dependency_available(module_name: str) -> bool:
    return module_available(module_name)


class StressCapabilityRequired(ValueError):
    """Raised when a stress-dependent simulation is requested without stress-capability provenance."""


def _request_requires_stress(request: SimulationRequest) -> bool:
    if bool(request.protocol.get("requires_stress", False)):
        return True
    return any(str(ensemble).strip().lower() == "npt" for ensemble in request.ensembles())


def _declared_stress_capability(model: Mapping[str, Any]) -> Optional[bool]:
    for key in ("stress_capable", "supports_stress", "stress"):
        if key in model:
            return bool(model[key])

    capabilities = model.get("capabilities", model.get("model_capabilities"))
    if isinstance(capabilities, Mapping):
        for key in ("stress", "stress_capable", "supports_stress"):
            if key in capabilities:
                return bool(capabilities[key])

    outputs = model.get("outputs", model.get("model_outputs"))
    if outputs is None:
        return None
    if isinstance(outputs, str):
        values = {outputs.strip().lower()}
    elif isinstance(outputs, Sequence):
        values = {str(item).strip().lower() for item in outputs}
    else:
        return None
    return bool(values & {"stress", "stresses", "virial", "virials"})
