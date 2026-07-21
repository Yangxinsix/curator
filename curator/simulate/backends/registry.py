from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from curator.utils import normalize_string_list

from .ase import AseSimulationBackend
from .base import SimulationBackend
from .lammps import LammpsSimulationBackend
from .torchsim import TorchSimSimulationBackend


def registered_backends() -> Dict[str, SimulationBackend]:
    return {
        "ase": AseSimulationBackend(),
        "torchsim": TorchSimSimulationBackend(),
        "lammps": LammpsSimulationBackend(),
    }


def list_backend_capabilities() -> List[Dict[str, Any]]:
    return [backend.capabilities().to_dict() for backend in registered_backends().values()]


def get_backend(name: str) -> SimulationBackend:
    backends = registered_backends()
    key = str(name or "auto").strip().lower()
    if key not in backends:
        raise ValueError(f"Unknown simulation backend {key!r}; expected one of {sorted(backends)}.")
    return backends[key]


def _missing_capabilities(backend: SimulationBackend, requirements: Mapping[str, Any]) -> List[str]:
    caps = backend.capabilities()
    available = caps.to_dict()["supports"]
    missing: List[str] = []
    for key, required in dict(requirements or {}).items():
        if key in {"tasks", "task_types"}:
            supported = set(caps.tasks)
            values = set(normalize_string_list(required, []))
            if values and not values.issubset(supported):
                missing.append(f"tasks={sorted(values - supported)}")
            continue
        if key == "ensembles":
            supported = set(caps.ensembles)
            values = set(normalize_string_list(required, []))
            if values and not values.issubset(supported):
                missing.append(f"ensembles={sorted(values - supported)}")
            continue
        if key == "integrators":
            supported = set(caps.integrators)
            values = set(normalize_string_list(required, []))
            if values and not values.issubset(supported):
                missing.append(f"integrators={sorted(values - supported)}")
            continue
        if key in {"batching", "autobatching", "online_uncertainty", "trajectory", "summary"}:
            if bool(required) and not bool(available.get(key)):
                missing.append(key)
            continue
        if key in {"devices", "dtypes"}:
            supported = set(available.get(key, []))
            values = set(normalize_string_list(required, []))
            if values and not values.issubset(supported):
                missing.append(f"{key}={sorted(values - supported)}")
    return missing


def select_backend(backend_policy: Mapping[str, Any]) -> SimulationBackend:
    policy = dict(backend_policy or {})
    mode = str(policy.get("mode", policy.get("backend", "auto"))).strip().lower()
    backends = registered_backends()
    allowed = normalize_string_list(policy.get("allowed"), list(backends))
    fallback = str(policy.get("fallback", "none")).strip().lower()
    requirements = dict(policy.get("require_capabilities") or {})

    if mode != "auto":
        if mode not in backends:
            raise ValueError(f"backend_policy.mode must be one of {sorted(backends) + ['auto']}, got {mode!r}.")
        if mode not in allowed:
            raise ValueError(f"backend {mode!r} is not allowed by backend_policy.allowed={allowed!r}.")
        backend = backends[mode]
        missing = _missing_capabilities(backend, requirements)
        if missing:
            raise ValueError(f"backend {mode!r} is missing required capabilities: {missing!r}.")
        return backend

    candidates: Iterable[str] = allowed or backends.keys()
    blocked: List[str] = []
    for name in candidates:
        backend = backends.get(name)
        if backend is None:
            blocked.append(f"{name}: unknown")
            continue
        caps = backend.capabilities()
        missing = _missing_capabilities(backend, requirements)
        if caps.implemented and caps.available and not missing:
            return backend
        blocked.append(f"{name}: available={caps.available}, implemented={caps.implemented}, missing={missing}")

    if fallback == "none":
        raise RuntimeError(f"No allowed simulation backend satisfies backend_policy; checked {blocked!r}.")
    raise RuntimeError(f"No allowed simulation backend satisfies backend_policy; checked {blocked!r}.")
