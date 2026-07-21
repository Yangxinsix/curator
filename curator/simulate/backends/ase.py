from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from curator.utils import error_result, safe_label, single_or_list, utc_now

from ..execution import execute_simulation_config
from ..protocols.md import ASE_MD_INTEGRATORS, SUPPORTED_ASE_MD_ENSEMBLES, md_integrator_for_backend
from ..results import case_result_from_completed, timeout_result
from ..spec import SimulationCaseSpec, SimulationRequest
from ..validation import resolve_uncertainty_method
from .base import SimulationBackend, SimulationBackendCapabilities, dependency_available
from .common import thermo_callback


GPA_PER_EV_A3 = 160.21766208


class AseSimulationBackend(SimulationBackend):
    name = "ase"

    def capabilities(self) -> SimulationBackendCapabilities:
        return SimulationBackendCapabilities(
            backend=self.name,
            available=dependency_available("ase"),
            implemented=True,
            tasks=["md", "md_stability"],
            ensembles=sorted(SUPPORTED_ASE_MD_ENSEMBLES),
            integrators=sorted(ASE_MD_INTEGRATORS),
            devices=["cpu", "cuda"],
            dtypes=[],
            batching=False,
            autobatching=False,
            online_uncertainty=True,
            trajectory=True,
            summary=True,
            requires_deployed_model=False,
            notes=[
                "Uses Curator's ASE MDEngine through the standard Simulator command path.",
                "NPT requires model stress support.",
            ],
        )

    def plan_cases(self, request: SimulationRequest) -> List[SimulationCaseSpec]:
        self.validate_request(request)
        temperatures = request.temperatures()
        pressure_values = request.pressures()
        ensembles = request.ensembles()
        structures = request.structures_as_strings
        start_indices = [None if item is None else int(item) for item in request.start_indices]
        steps = request.int_protocol("steps", 1000)
        if steps <= 0:
            raise ValueError("protocol.steps must be positive.")
        timestep_fs = request.float_protocol("timestep_fs", 0.5)
        requested_integrator = request.protocol.get("integrator")
        seed = request.int_protocol("seed", 1234)

        cases: List[SimulationCaseSpec] = []
        case_index = 0
        for structure_index, init_traj in enumerate(structures):
            start_index = start_indices[structure_index]
            for temperature_K in temperatures:
                for ensemble in ensembles:
                    integrator = md_integrator_for_backend("ase", ensemble, requested_integrator)
                    if ensemble not in SUPPORTED_ASE_MD_ENSEMBLES or integrator is None:
                        continue
                    ensemble_pressures = pressure_values if ensemble == "npt" else [None]
                    for pressure_GPa in ensemble_pressures:
                        pressure_label = f"_{safe_label(pressure_GPa)}GPa" if pressure_GPa is not None else ""
                        label = f"{ensemble}_{safe_label(temperature_K)}K{pressure_label}_s{structure_index}"
                        case_dir = _case_dir(request.run_dir, case_index, label)
                        cases.append(
                            SimulationCaseSpec(
                                case_id=case_index,
                                label=label,
                                backend=self.name,
                                task_type=request.task_type,
                                criteria_profile=request.criteria_profile,
                                init_structures=[init_traj],
                                start_indices=[start_index],
                                structure_indices=[structure_index],
                                ensemble=ensemble,
                                temperature_K=temperature_K,
                                pressure_GPa=pressure_GPa,
                                steps=steps,
                                timestep_fs=timestep_fs,
                                integrator=integrator,
                                seed=seed + case_index,
                                run_dir=case_dir,
                                timeout_sec=request.timeout_sec,
                                model_paths=request.model_paths,
                                reference_dataset=request.reference_dataset,
                                uncertainty_method=request.uncertainty_method,
                                device=request.device,
                                dtype=request.dtype,
                                protocol=dict(request.protocol),
                                model=dict(request.model),
                                batch=False,
                                metadata={
                                    "structure_index": structure_index,
                                    "init_traj": init_traj,
                                    "start_index": start_index,
                                    "initialize_velocities": bool(request.protocol.get("initialize_velocities", False)),
                                    "force_temperature": bool(
                                        request.protocol.get(
                                            "force_temperature",
                                            bool(request.protocol.get("initialize_velocities", False)),
                                        )
                                    ),
                                },
                            )
                        )
                        case_index += 1
        return cases

    def build_config(self, case: SimulationCaseSpec) -> Dict[str, Any]:
        protocol = case.protocol
        model_like = single_or_list(case.model_paths)
        uncertainty_method = case.uncertainty_method
        timestep_fs = float(case.timestep_fs)
        temperature_K = float(case.temperature_K or 300.0)
        friction = float(protocol.get("friction", 0.01))
        taut_fs = float(protocol.get("taut_fs", 100.0 * timestep_fs))
        taup_fs = float(protocol.get("taup_fs", 1000.0 * timestep_fs))
        initialize_velocities = bool(protocol.get("initialize_velocities", False))
        force_temperature = bool(protocol.get("force_temperature", initialize_velocities))
        remove_translation = bool(protocol.get("remove_translation", True))
        remove_rotation = bool(protocol.get("remove_rotation", True))
        log_interval = int(protocol.get("log_interval", 1))
        trajectory_interval = int(protocol.get("trajectory_interval", 1))
        summary_interval = int(protocol.get("summary_interval", 1))
        summary_compute_min_distance = bool(protocol.get("summary_compute_min_distance", True))
        summary_compute_forces = bool(protocol.get("summary_compute_forces", True))
        low_threshold = protocol.get("low_threshold")
        high_threshold = protocol.get("high_threshold")
        uncertain_count = protocol.get("uncertain_count")
        uncertainty_kernel = str(case.model.get("uncertainty_kernel", protocol.get("uncertainty_kernel", "local-full-g")))
        uncertainty_max_structures = case.model.get("uncertainty_max_structures", protocol.get("uncertainty_max_structures"))

        callbacks: List[Dict[str, Any]] = [
            {
                "_target_": "curator.simulate.callbacks.calc.CalculatorAssign",
                "calculator": model_like,
                "warmup": True,
                "require_forces": True,
                "device": case.device,
            }
        ]
        if initialize_velocities:
            callbacks.append(
                {
                    "_target_": "curator.simulate.callbacks.velocity.VelocityInitializer",
                    "temperature_K": temperature_K,
                    "force": force_temperature,
                    "remove_translation": remove_translation,
                    "remove_rotation": remove_rotation,
                }
            )

        callbacks.append(
            thermo_callback(
                logger_target="curator.simulate.callbacks.thermo.MDThermoLogger"
                if uncertainty_method == "none"
                else "curator.simulate.callbacks.thermo_uncertainty.ThermoWithUncertainty",
                uncertainty_method=uncertainty_method,
                model_like=model_like,
                reference_dataset=case.reference_dataset,
                device=case.device,
                run_dir=case.run_dir,
                log_interval=log_interval,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                uncertain_count=uncertain_count,
                uncertainty_kernel=uncertainty_kernel,
                uncertainty_max_structures=uncertainty_max_structures,
            )
        )
        callbacks.extend(
            [
                {
                    "_target_": "curator.simulate.callbacks.trajectory.TrajectoryWriter",
                    "path": str(case.run_dir / "trajectory.traj"),
                    "interval": int(trajectory_interval),
                },
                {
                    "_target_": "curator.simulate.callbacks.summary.SimulationSummaryWriter",
                    "path": str(case.run_dir / "simulation_summary.json"),
                    "interval": int(summary_interval),
                    "timestep_fs": timestep_fs,
                    "compute_min_distance": summary_compute_min_distance,
                    "compute_forces": summary_compute_forces,
                    "initial_step_included": True,
                },
                {"_target_": "curator.simulate.callbacks.early_stop.EarlyStop"},
            ]
        )

        return {
            "cfg": None,
            "seed": int(case.seed),
            "run_path": str(case.run_dir),
            "model_path": model_like,
            "dataset": case.reference_dataset,
            "device": case.device,
            "deploy": None,
            "calculator": model_like,
            "simulator": {
                "_target_": "curator.simulate.core.simulator.Simulator",
                "init_traj": case.init_traj_for_config,
                "start_index": case.start_index_for_config,
                "engine": _ase_md_engine_config(
                    str(case.integrator),
                    timestep_fs,
                    temperature_K,
                    friction,
                    case.pressure_GPa,
                    taut_fs,
                    taup_fs,
                ),
                "run_kwargs": {"steps": int(case.steps)},
                "callbacks": callbacks,
            },
            "runner_metadata": {
                "created_at": utc_now(),
                "backend": self.name,
                "uncertainty_method": uncertainty_method,
            },
        }

    def run_case(self, case: SimulationCaseSpec, *, tool_name: str = "run_simulation") -> Dict[str, Any]:
        artifacts: Dict[str, Any] = {"run_dir": str(case.run_dir)}
        try:
            resolved_method = resolve_uncertainty_method(case.uncertainty_method, case.model_paths, case.reference_dataset)
            if resolved_method == "mahalanobis" and not case.reference_dataset:
                raise ValueError("uncertainty_method=mahalanobis requires reference_dataset.")
            if resolved_method == "ensemble" and len(case.model_paths) < 2:
                raise ValueError("uncertainty_method=ensemble requires at least two model_path entries.")
            resolved_case = replace(case, uncertainty_method=resolved_method)
            config = self.build_config(resolved_case)
            completed, artifacts = execute_simulation_config(config, resolved_case.run_dir, resolved_case.timeout_sec)
            return case_result_from_completed(
                case=resolved_case,
                completed=completed,
                artifacts=artifacts,
                manifest_filename="simulation_manifest.json",
                tool_name=tool_name,
            )
        except subprocess.TimeoutExpired as exc:
            return timeout_result(case=case, exc=exc, artifacts=artifacts)
        except Exception as exc:
            return error_result(
                type(exc).__name__,
                str(exc),
                artifacts=artifacts,
                backend=self.name,
                case=case.case_metadata(),
                recoverable=True,
            )


def _case_dir(run_dir: Path, case_index: int, label: str) -> Path:
    case_dir = run_dir / f"case_{case_index:03d}_{safe_label(label)}"
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def _ase_md_engine_config(
    integrator: str,
    timestep_fs: float,
    temperature_K: float,
    friction: float,
    pressure_GPa: Optional[float],
    taut_fs: float,
    taup_fs: float,
) -> Dict[str, Any]:
    if integrator == "verlet":
        return {
            "_target_": "curator.simulate.engines.ase_md.MDEngine",
            "dynamics_cls": "ase.md.verlet.VelocityVerlet",
            "timestep": float(timestep_fs),
        }
    if integrator == "npt_berendsen":
        pressure_au = 0.0 if pressure_GPa is None else float(pressure_GPa) / GPA_PER_EV_A3
        return {
            "_target_": "curator.simulate.engines.ase_md.MDEngine",
            "dynamics_cls": "ase.md.nptberendsen.NPTBerendsen",
            "timestep": float(timestep_fs),
            "temperature_K": float(temperature_K),
            "pressure_au": pressure_au,
            "taut": float(taut_fs),
            "taup": float(taup_fs),
        }
    return {
        "_target_": "curator.simulate.engines.ase_md.MDEngine",
        "dynamics_cls": "ase.md.langevin.Langevin",
        "timestep": float(timestep_fs),
        "temperature_K": float(temperature_K),
        "friction": float(friction),
    }
