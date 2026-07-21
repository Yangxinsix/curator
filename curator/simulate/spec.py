from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from curator.utils import as_dict, as_list, normalize_model_paths, normalize_string_list

from .validation import DEFAULT_CRITERIA_PROFILE


ALLOWED_TASK_TYPES = {"md", "md_stability", "relaxation", "static", "production_md", "deployment_equivalence"}
PROTOCOL_PRESETS: Dict[str, Dict[str, Any]] = {
    "md_direct_use_probe_v1": {
        "description": (
            "Default MD direct-use validation matrix. It runs NVE/NVT stability checks "
            "at a temperature ladder and validates with md_direct_use_validation_v1."
        ),
        "task": {
            "type": "md",
            "criteria_profile": DEFAULT_CRITERIA_PROFILE,
        },
        "protocol": {
            "ensembles": ["nve", "nvt"],
            "temperature_K": [300.0, 600.0, 900.0],
            "steps": 1000,
            "timestep_fs": 0.5,
            "integrator": None,
            "friction": 0.01,
            "initialize_velocities": True,
            "force_temperature": True,
            "remove_translation": True,
            "remove_rotation": True,
            "log_interval": 10,
            "trajectory_interval": 10,
            "summary_interval": 10,
            "summary_compute_min_distance": True,
            "summary_compute_forces": True,
        },
    },
    "short_md_smoke_v1": {
        "description": "Short local MD smoke protocol for checking that a model/system can run and emit validator-ready artifacts.",
        "task": {
            "type": "md",
            "criteria_profile": "md_smoke_v1",
        },
        "protocol": {
            "ensembles": ["nvt"],
            "temperature_K": [300.0],
            "steps": 100,
            "timestep_fs": 0.5,
            "integrator": "langevin",
            "friction": 0.01,
            "log_interval": 1,
            "trajectory_interval": 1,
            "summary_interval": 1,
            "summary_compute_min_distance": True,
            "summary_compute_forces": True,
        },
    },
}


def apply_protocol_preset(
    *,
    task: Dict[str, Any],
    protocol: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    preset = protocol.get("preset")
    if preset is None:
        return task, protocol
    key = str(preset).strip()
    if key not in PROTOCOL_PRESETS:
        raise ValueError(f"protocol.preset must be one of {sorted(PROTOCOL_PRESETS)}, got {key!r}.")
    preset_spec = PROTOCOL_PRESETS[key]
    merged_task = dict(preset_spec.get("task", {}))
    merged_task.update(task)
    merged_protocol = dict(preset_spec.get("protocol", {}))
    merged_protocol.update(protocol)
    merged_protocol["preset"] = key
    return merged_task, merged_protocol


@dataclass(frozen=True)
class SimulationRequest:
    task: Dict[str, Any]
    system: Dict[str, Any]
    model: Dict[str, Any]
    protocol: Dict[str, Any]
    backend_policy: Dict[str, Any]
    decision_policy: Dict[str, Any]
    run_dir: Path
    timeout_sec: int

    @classmethod
    def from_inputs(
        cls,
        *,
        task: Optional[Mapping[str, Any]] = None,
        system: Optional[Mapping[str, Any]] = None,
        model: Optional[Mapping[str, Any]] = None,
        protocol: Optional[Mapping[str, Any]] = None,
        backend_policy: Optional[Mapping[str, Any]] = None,
        decision_policy: Optional[Mapping[str, Any]] = None,
        out: str = "simulation",
        timeout_sec: int = 1800,
    ) -> "SimulationRequest":
        task_dict = as_dict(task, name="task")
        protocol_dict = as_dict(protocol, name="protocol")
        task_dict, protocol_dict = apply_protocol_preset(task=task_dict, protocol=protocol_dict)
        run_dir = Path(out).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            task=task_dict,
            system=as_dict(system, name="system"),
            model=as_dict(model, name="model"),
            protocol=protocol_dict,
            backend_policy=as_dict(backend_policy, name="backend_policy"),
            decision_policy=as_dict(decision_policy, name="decision_policy") if decision_policy is not None else {},
            run_dir=run_dir,
            timeout_sec=int(timeout_sec),
        )

    @property
    def task_type(self) -> str:
        value = str(self.task.get("type", self.protocol.get("type", "md"))).strip().lower()
        if value not in ALLOWED_TASK_TYPES:
            raise ValueError(f"task.type must be one of {sorted(ALLOWED_TASK_TYPES)}, got {value!r}.")
        return value

    @property
    def criteria_profile(self) -> str:
        return str(self.task.get("criteria_profile", DEFAULT_CRITERIA_PROFILE))

    @property
    def init_structures(self) -> List[Any]:
        structures = as_list(
            self.system.get("init_structures", self.system.get("init_traj", self.system.get("init_structure")))
        )
        if not structures:
            raise ValueError("system.init_structures or system.init_traj is required.")
        return structures

    @property
    def start_indices(self) -> List[Any]:
        structures = self.init_structures
        indices = as_list(self.system.get("start_indices", self.system.get("start_index", 0)), [0])
        if len(indices) == 1 and len(structures) > 1:
            indices = indices * len(structures)
        if len(indices) != len(structures):
            raise ValueError("system.start_indices length must match system.init_structures length.")
        return indices

    @property
    def model_paths(self) -> List[str]:
        return normalize_model_paths(self.model.get("model_path", self.model.get("model_paths")))

    @property
    def reference_dataset(self) -> Optional[str]:
        value = self.model.get("reference_dataset")
        return str(value) if value else None

    @property
    def reference_dataset_role(self) -> str:
        return str(self.model.get("reference_dataset_role", "unknown")).strip().lower()

    @property
    def uncertainty_method(self) -> str:
        return str(self.model.get("uncertainty_method", "auto")).strip().lower()

    @property
    def device(self) -> str:
        return str(self.backend_policy.get("device", self.model.get("device", "cpu")))

    @property
    def dtype(self) -> Optional[str]:
        value = self.backend_policy.get("dtype", self.model.get("dtype"))
        return str(value) if value is not None else None

    @property
    def structures_as_strings(self) -> List[str]:
        return [str(item) for item in self.init_structures]

    def temperatures(self) -> List[float]:
        return [float(value) for value in as_list(self.protocol.get("temperature_K"), [300.0])]

    def pressures(self) -> List[float]:
        return [float(value) for value in as_list(self.protocol.get("pressure_GPa"), [0.0])]

    def ensembles(self) -> List[str]:
        return normalize_string_list(self.protocol.get("ensembles", self.protocol.get("ensemble")), ["nvt"])

    def bool_protocol(self, key: str, default: bool) -> bool:
        return bool(self.protocol.get(key, default))

    def int_protocol(self, key: str, default: int) -> int:
        return int(self.protocol.get(key, default))

    def float_protocol(self, key: str, default: float) -> float:
        return float(self.protocol.get(key, default))


@dataclass(frozen=True)
class SimulationCaseSpec:
    case_id: int
    label: str
    backend: str
    task_type: str
    criteria_profile: str
    init_structures: List[str]
    start_indices: List[Optional[int]]
    structure_indices: List[int]
    ensemble: Optional[str]
    temperature_K: Optional[float]
    pressure_GPa: Optional[float]
    steps: int
    timestep_fs: float
    integrator: Optional[str]
    seed: int
    run_dir: Path
    timeout_sec: int
    model_paths: List[str]
    reference_dataset: Optional[str]
    uncertainty_method: str
    device: str
    dtype: Optional[str]
    protocol: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    batch: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def init_traj_for_config(self) -> Any:
        return self.init_structures if self.batch else self.init_structures[0]

    @property
    def start_index_for_config(self) -> Any:
        return self.start_indices if self.batch else self.start_indices[0]

    def case_metadata(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "label": self.label,
            "backend": self.backend,
            "task_type": self.task_type,
            "criteria_profile": self.criteria_profile,
            "structure_indices": list(self.structure_indices),
            "init_structures": list(self.init_structures),
            "start_indices": list(self.start_indices),
            "ensemble": self.ensemble,
            "temperature_K": self.temperature_K,
            "pressure_GPa": self.pressure_GPa,
            "steps": self.steps,
            "timestep_fs": self.timestep_fs,
            "integrator": self.integrator,
            "batch": self.batch,
        }
        payload.update(dict(self.metadata))
        return payload
