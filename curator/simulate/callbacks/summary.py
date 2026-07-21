from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from curator.simulate.core.callbacks import Callback
from curator.simulate.core.context import SimContext


class SimulationSummaryWriter(Callback):
    """Write a compact JSON summary for MCP/runner consumers."""

    def __init__(
        self,
        path: str = "simulation_summary.json",
        interval: int = 1,
        timestep_fs: Optional[float] = None,
        compute_min_distance: bool = True,
        compute_forces: bool = True,
        initial_step_included: bool = True,
    ) -> None:
        self.path = Path(path)
        self.interval = max(1, int(interval))
        self.timestep_fs = float(timestep_fs) if timestep_fs is not None else None
        self.compute_min_distance = bool(compute_min_distance)
        self.compute_forces = bool(compute_forces)
        self.initial_step_included = bool(initial_step_included)
        self.started_at = 0.0
        self.steps_completed = 0
        self.warning_steps = 0
        self.outlier_steps = 0
        self.last_uncertainty: Dict[str, Any] = {}
        self.max_uncertainty: Dict[str, float] = {}
        self.early_stop_reason: Optional[str] = None
        self.sample_count = 0
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self._last_metric_step: Optional[int] = None

    def on_sim_start(self, ctx: SimContext) -> None:
        self.started_at = time.perf_counter()
        self.steps_completed = 0
        self.warning_steps = 0
        self.outlier_steps = 0
        self.last_uncertainty = {}
        self.max_uncertainty = {}
        self.early_stop_reason = None
        self.sample_count = 0
        self.metrics = {}
        self._last_metric_step = None

    def on_step(self, ctx: SimContext) -> None:
        self.steps_completed = int(ctx.step)
        self.early_stop_reason = ctx.state.get("early_stop_reason") or self.early_stop_reason
        self._record_uncertainty(ctx.state.get("uncertainty") or {})
        self._record_metrics(ctx)

    def on_sim_end(self, ctx: SimContext) -> None:
        self.steps_completed = int(ctx.step)
        self.early_stop_reason = ctx.state.get("early_stop_reason") or self.early_stop_reason
        self._record_uncertainty(ctx.state.get("uncertainty") or {})
        self._record_metrics(ctx)
        self._write(status="completed")

    def on_exception(self, ctx: SimContext, exc: BaseException) -> None:
        self.steps_completed = int(ctx.step)
        self.early_stop_reason = ctx.state.get("early_stop_reason") or self.early_stop_reason
        self._record_uncertainty(ctx.state.get("uncertainty") or {})
        self._record_metrics(ctx)
        self._write(status="failed", exception=f"{type(exc).__name__}: {exc}")

    def _record_uncertainty(self, uncertainty: Any) -> None:
        if isinstance(uncertainty, list):
            for item in uncertainty:
                self._record_uncertainty(item)
            return
        if not isinstance(uncertainty, dict) or not uncertainty:
            return
        cleaned = {str(k): self._clean(v) for k, v in uncertainty.items()}
        self.last_uncertainty = cleaned
        if bool(cleaned.get("is_warning")):
            self.warning_steps += 1
        if bool(cleaned.get("is_outlier")):
            self.outlier_steps += 1
        for key, value in cleaned.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            previous = self.max_uncertainty.get(key)
            if previous is None or value > previous:
                self.max_uncertainty[key] = float(value)

    def _record_metrics(self, ctx: SimContext) -> None:
        atoms = ctx.atoms
        sim_state = ctx.state.get("sim_state")
        if atoms is None and sim_state is None:
            return
        step = int(ctx.step)
        if self._last_metric_step == step:
            return
        self._last_metric_step = step
        self.sample_count += 1
        atoms_list = atoms if isinstance(atoms, list) else [atoms]
        atom_metrics = [self._atoms_snapshot(item) for item in atoms_list if item is not None]
        state_metrics = self._state_snapshot(sim_state)
        natoms = int(
            state_metrics.get("natoms")
            if state_metrics.get("natoms") is not None
            else sum(int(item.get("natoms") or 0) for item in atom_metrics)
        )

        self._update_metric("step", float(step))
        self._update_metric("natoms", float(natoms))
        epot_values = [item["epot"] for item in atom_metrics if item.get("epot") is not None]
        ekin_values = [item["ekin"] for item in atom_metrics if item.get("ekin") is not None]
        volume_values = [item["volume"] for item in atom_metrics if item.get("volume") is not None]
        pressure_values = [item["pressure"] for item in atom_metrics if item.get("pressure") is not None]
        force_values = [item["max_force"] for item in atom_metrics if item.get("max_force") is not None]
        distance_values = [item["min_distance"] for item in atom_metrics if item.get("min_distance") is not None]
        epot = state_metrics.get("epot")
        if epot is None:
            epot = sum(epot_values) if epot_values else None
        ekin = state_metrics.get("ekin")
        if ekin is None:
            ekin = sum(ekin_values) if ekin_values else None
        if epot is not None:
            self._update_metric("epot_eV", epot)
            if natoms > 0:
                self._update_metric("epot_eV_per_atom", epot / natoms)
        if ekin is not None:
            self._update_metric("ekin_eV", ekin)
            if natoms > 0:
                self._update_metric("ekin_eV_per_atom", ekin / natoms)
                temp = 2.0 * ekin / (3 * natoms * 8.617333262145e-5)
                self._update_metric("temperature_K", temp)
        if epot is not None and ekin is not None:
            etot = epot + ekin
            self._update_metric("etot_eV", etot)
            if natoms > 0:
                self._update_metric("etot_eV_per_atom", etot / natoms)

        volume = state_metrics.get("volume")
        if volume is None:
            volume = sum(volume_values) if volume_values else None
        if volume is not None:
            self._update_metric("volume_A3", volume)
            if natoms > 0:
                self._update_metric("volume_A3_per_atom", volume / natoms)
        pressure = state_metrics.get("pressure")
        if pressure is None:
            pressure = sum(pressure_values) / len(pressure_values) if pressure_values else None
        if pressure is not None:
            self._update_metric("pressure_eV_A3", pressure)
        if self.compute_forces:
            max_force = state_metrics.get("max_force")
            if max_force is None:
                max_force = max(force_values) if force_values else None
            if max_force is not None:
                self._update_metric("max_force_eV_A", max_force)
        if self.compute_min_distance:
            min_distance = min(distance_values) if distance_values else None
            if min_distance is not None:
                self._update_metric("min_distance_A", min_distance)

    def _atoms_snapshot(self, atoms: Any) -> Dict[str, Any]:
        natoms = 0
        try:
            natoms = int(atoms.get_global_number_of_atoms())
        except Exception:
            natoms = 0
        return {
            "natoms": natoms,
            "epot": self._try_float(lambda: atoms.get_potential_energy()),
            "ekin": self._try_float(lambda: atoms.get_kinetic_energy()),
            "volume": self._try_float(lambda: atoms.get_volume()),
            "pressure": self._pressure(atoms),
            "max_force": self._max_force(atoms) if self.compute_forces else None,
            "min_distance": self._min_distance(atoms) if self.compute_min_distance else None,
        }

    def _state_snapshot(self, state: Any) -> Dict[str, Any]:
        if state is None:
            return {}
        return {
            "natoms": self._state_natoms(state),
            "epot": self._tensor_sum_float(getattr(state, "energy", None)),
            "ekin": self._state_kinetic_energy(state),
            "volume": self._state_volume(state),
            "pressure": self._state_pressure(state),
            "max_force": self._max_force_tensor(getattr(state, "forces", None)) if self.compute_forces else None,
        }

    def _update_metric(self, name: str, value: float) -> None:
        if value != value:
            return
        item = self.metrics.get(name)
        if item is None:
            self.metrics[name] = {
                "first": float(value),
                "last": float(value),
                "min": float(value),
                "max": float(value),
            }
            return
        item["last"] = float(value)
        item["min"] = min(float(item["min"]), float(value))
        item["max"] = max(float(item["max"]), float(value))

    @staticmethod
    def _try_float(fn: Any) -> Optional[float]:
        try:
            return float(fn())
        except Exception:
            return None

    @staticmethod
    def _tensor_sum_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            import torch

            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    return None
                return float(value.detach().sum().cpu())
        except Exception:
            pass
        try:
            import numpy as np

            array = np.asarray(value)
            if array.size == 0:
                return None
            return float(array.sum())
        except Exception:
            return None

    @staticmethod
    def _state_natoms(state: Any) -> Optional[int]:
        for attr in ("n_atoms", "num_atoms"):
            value = getattr(state, attr, None)
            if value is None:
                continue
            try:
                if hasattr(value, "item"):
                    return int(value.item())
                return int(value)
            except Exception:
                pass
        positions = getattr(state, "positions", None)
        if positions is not None:
            try:
                return int(positions.shape[-2])
            except Exception:
                return None
        return None

    @staticmethod
    def _state_kinetic_energy(state: Any) -> Optional[float]:
        momenta = getattr(state, "momenta", None)
        masses = getattr(state, "masses", None)
        if momenta is None or masses is None:
            return None
        try:
            import torch

            p = momenta.detach() if isinstance(momenta, torch.Tensor) else torch.as_tensor(momenta)
            m = masses.detach() if isinstance(masses, torch.Tensor) else torch.as_tensor(masses)
            if m.ndim == 1:
                m = m.unsqueeze(-1)
            return float((p * p / (2 * m)).sum().detach().cpu())
        except Exception:
            return None

    @staticmethod
    def _state_volume(state: Any) -> Optional[float]:
        cell = getattr(state, "cell", None)
        if cell is None:
            return None
        try:
            import torch

            tensor = cell.detach() if isinstance(cell, torch.Tensor) else torch.as_tensor(cell)
            if tensor.ndim == 2:
                return float(torch.det(tensor).abs().detach().cpu())
            if tensor.ndim == 3:
                return float(torch.det(tensor).abs().sum().detach().cpu())
        except Exception:
            return None
        return None

    @staticmethod
    def _state_pressure(state: Any) -> Optional[float]:
        stress = getattr(state, "stress", None)
        if stress is None:
            return None
        try:
            import torch

            tensor = stress.detach() if isinstance(stress, torch.Tensor) else torch.as_tensor(stress)
            if tensor.ndim == 2 and tensor.shape == (3, 3):
                return -float(torch.trace(tensor).detach().cpu()) / 3.0
            if tensor.ndim == 3 and tensor.shape[-2:] == (3, 3):
                traces = tensor.diagonal(dim1=-2, dim2=-1).sum(-1)
                return -float(traces.mean().detach().cpu()) / 3.0
            if tensor.ndim >= 1 and tensor.shape[-1] >= 3:
                return -float(tensor[..., :3].sum(-1).mean().detach().cpu()) / 3.0
        except Exception:
            return None
        return None

    @staticmethod
    def _max_force_tensor(forces: Any) -> Optional[float]:
        if forces is None:
            return None
        try:
            import torch

            tensor = forces.detach() if isinstance(forces, torch.Tensor) else torch.as_tensor(forces)
            if tensor.numel() == 0:
                return None
            return float(torch.linalg.norm(tensor.reshape(-1, 3), dim=1).max().detach().cpu())
        except Exception:
            return None

    @staticmethod
    def _pressure(atoms: Any) -> Optional[float]:
        try:
            stress = atoms.get_stress()
            return -float(stress[0] + stress[1] + stress[2]) / 3.0
        except Exception:
            return None

    @staticmethod
    def _max_force(atoms: Any) -> Optional[float]:
        try:
            import numpy as np

            forces = np.asarray(atoms.get_forces())
            if forces.size == 0:
                return None
            return float(np.linalg.norm(forces.reshape(-1, 3), axis=1).max())
        except Exception:
            return None

    @staticmethod
    def _min_distance(atoms: Any) -> Optional[float]:
        try:
            import numpy as np

            distances = np.asarray(atoms.get_all_distances(mic=True))
            if distances.size == 0:
                return None
            distances[distances <= 0.0] = np.inf
            value = float(distances.min())
            return value if value != float("inf") else None
        except Exception:
            return None

    def _structured_metrics(self, integrated_steps: int) -> Dict[str, Any]:
        thermo_keys = [
            "epot_eV",
            "ekin_eV",
            "etot_eV",
            "epot_eV_per_atom",
            "ekin_eV_per_atom",
            "etot_eV_per_atom",
            "temperature_K",
            "pressure_eV_A3",
        ]
        structure_keys = ["natoms", "volume_A3", "volume_A3_per_atom", "min_distance_A"]
        force_keys = ["max_force_eV_A"]
        thermo = {key: self.metrics[key] for key in thermo_keys if key in self.metrics}
        structure = {key: self.metrics[key] for key in structure_keys if key in self.metrics}
        force = {key: self.metrics[key] for key in force_keys if key in self.metrics}
        drift: Dict[str, Any] = {}
        for key in ("etot_eV", "etot_eV_per_atom", "temperature_K", "volume_A3"):
            item = self.metrics.get(key)
            if item is None:
                continue
            drift[key] = float(item["last"]) - float(item["first"])
        if self.timestep_fs is not None and integrated_steps > 0:
            elapsed_ps = integrated_steps * self.timestep_fs / 1000.0
            drift["elapsed_ps"] = elapsed_ps
            etot_pa = drift.get("etot_eV_per_atom")
            if etot_pa is not None and elapsed_ps > 0:
                drift["etot_eV_per_atom_per_ps"] = float(etot_pa) / elapsed_ps
        return {
            "sample_count": self.sample_count,
            "thermo": thermo,
            "force": force,
            "structure": structure,
            "drift": drift,
        }

    def _write(self, *, status: str, exception: Optional[str] = None) -> None:
        # ASE Dynamics invokes attached callbacks once for the initial state and
        # then after integration steps. For this short-MD callback, report both
        # the raw callback count and the estimated number of integrated steps.
        integrated_steps = max(0, self.steps_completed - 1) if self.initial_step_included else max(0, self.steps_completed)
        walltime_sec = max(0.0, time.perf_counter() - self.started_at) if self.started_at else None
        performance = {
            "simulation_walltime_sec": walltime_sec,
            "steps_per_second": (
                float(integrated_steps) / walltime_sec
                if walltime_sec and walltime_sec > 0.0
                else None
            ),
        }
        payload: Dict[str, Any] = {
            "status": status,
            "steps_completed": integrated_steps,
            "callback_steps": self.steps_completed,
            "warning_steps": self.warning_steps,
            "outlier_steps": self.outlier_steps,
            "early_stop_reason": self.early_stop_reason,
            "last_uncertainty": self.last_uncertainty,
            "max_uncertainty": self.max_uncertainty,
            "walltime_sec": walltime_sec,
            "performance": performance,
        }
        payload.update(self._structured_metrics(integrated_steps))
        if exception is not None:
            payload["exception"] = exception
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)

    @staticmethod
    def _clean(value: Any) -> Any:
        if hasattr(value, "item") and callable(value.item):
            try:
                return value.item()
            except Exception:
                pass
        if isinstance(value, (list, tuple)):
            return [SimulationSummaryWriter._clean(v) for v in value]
        return value
