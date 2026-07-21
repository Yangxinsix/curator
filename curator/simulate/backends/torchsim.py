from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

from curator.utils import error_result, safe_label, single_or_list, utc_now

from ..execution import execute_simulation_config
from ..protocols.md import (
    SUPPORTED_TORCHSIM_MD_ENSEMBLES,
    TORCHSIM_MD_INTEGRATORS,
    md_integrator_for_backend,
)
from ..results import case_result_from_completed, timeout_result
from ..spec import SimulationCaseSpec, SimulationRequest
from ..validation import resolve_uncertainty_method
from .base import SimulationBackend, SimulationBackendCapabilities, dependency_available
from .common import thermo_callback


class TorchSimSimulationBackend(SimulationBackend):
    name = "torchsim"

    def capabilities(self) -> SimulationBackendCapabilities:
        return SimulationBackendCapabilities(
            backend=self.name,
            available=dependency_available("torch_sim"),
            implemented=True,
            tasks=["md", "md_stability"],
            ensembles=sorted(SUPPORTED_TORCHSIM_MD_ENSEMBLES),
            integrators=sorted(TORCHSIM_MD_INTEGRATORS),
            devices=["cpu", "cuda"],
            dtypes=["float32", "float64", "float16", "bfloat16"],
            batching=True,
            autobatching=True,
            online_uncertainty=True,
            trajectory=True,
            summary=True,
            requires_deployed_model=False,
            notes=[
                "Uses CuratorTorchSimAdapter and TorchSimEngine through the standard Simulator command path.",
                "TorchSim batch execution is represented in SimulationCaseSpec.batch and aggregate summary metrics.",
                "NPT integrators require stress-capable model outputs.",
                "Relaxation/static tasks are part of the backend contract but not declared until implemented with conformance tests.",
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
        thermostat = request.protocol.get("thermostat")
        barostat = request.protocol.get("barostat")
        seed = request.int_protocol("seed", 1234)
        batch_requested = bool(request.system.get("batch", request.backend_policy.get("batch", len(structures) > 1)))

        cases: List[SimulationCaseSpec] = []
        case_index = 0
        for temperature_K in temperatures:
            for ensemble in ensembles:
                integrator = md_integrator_for_backend(
                    "torchsim",
                    ensemble,
                    requested_integrator,
                    thermostat=thermostat,
                    barostat=barostat,
                )
                if ensemble not in SUPPORTED_TORCHSIM_MD_ENSEMBLES or integrator is None:
                    continue
                ensemble_pressures = pressure_values if ensemble == "npt" else [None]
                for pressure_GPa in ensemble_pressures:
                    if batch_requested:
                        pressure_label = f"_{safe_label(pressure_GPa)}GPa" if pressure_GPa is not None else ""
                        label = f"{ensemble}_{safe_label(temperature_K)}K{pressure_label}_batch"
                        cases.append(
                            self._case(
                                request,
                                case_index=case_index,
                                label=label,
                                structures=structures,
                                start_indices=start_indices,
                                structure_indices=list(range(len(structures))),
                                ensemble=ensemble,
                                temperature_K=temperature_K,
                                pressure_GPa=pressure_GPa,
                                steps=steps,
                                timestep_fs=timestep_fs,
                                integrator=integrator,
                                seed=seed + case_index,
                                batch=True,
                            )
                        )
                        case_index += 1
                        continue

                    for structure_index, init_traj in enumerate(structures):
                        pressure_label = f"_{safe_label(pressure_GPa)}GPa" if pressure_GPa is not None else ""
                        label = f"{ensemble}_{safe_label(temperature_K)}K{pressure_label}_s{structure_index}"
                        cases.append(
                            self._case(
                                request,
                                case_index=case_index,
                                label=label,
                                structures=[init_traj],
                                start_indices=[start_indices[structure_index]],
                                structure_indices=[structure_index],
                                ensemble=ensemble,
                                temperature_K=temperature_K,
                                pressure_GPa=pressure_GPa,
                                steps=steps,
                                timestep_fs=timestep_fs,
                                integrator=integrator,
                                seed=seed + case_index,
                                batch=False,
                            )
                        )
                        case_index += 1
        return cases

    def build_config(self, case: SimulationCaseSpec) -> Dict[str, Any]:
        protocol = case.protocol
        model_like = single_or_list(case.model_paths)
        timestep_ps = float(case.timestep_fs) / 1000.0
        temperature_K = float(case.temperature_K or 300.0)
        log_interval = int(protocol.get("log_interval", 10))
        trajectory_interval = int(protocol.get("trajectory_interval", 1))
        summary_interval = int(protocol.get("summary_interval", 1))
        summary_compute_min_distance = bool(protocol.get("summary_compute_min_distance", True))
        summary_compute_forces = bool(protocol.get("summary_compute_forces", True))
        low_threshold = protocol.get("low_threshold")
        high_threshold = protocol.get("high_threshold")
        uncertain_count = protocol.get("uncertain_count")
        uncertainty_kernel = str(case.model.get("uncertainty_kernel", protocol.get("uncertainty_kernel", "local-full-g")))
        uncertainty_max_structures = case.model.get("uncertainty_max_structures", protocol.get("uncertainty_max_structures"))
        integrator_kwargs = dict(protocol.get("integrator_kwargs") or protocol.get("torchsim_integrator_kwargs") or {})
        integrator_kwargs.setdefault("seed", int(case.seed))
        if str(case.integrator or "").startswith("npt_"):
            integrator_kwargs.setdefault("external_pressure_GPa", float(case.pressure_GPa or 0.0))
        if str(case.integrator) == "nvt_langevin" and "gamma" not in integrator_kwargs and "friction" in protocol:
            integrator_kwargs["gamma"] = float(protocol["friction"])

        callbacks: List[Dict[str, Any]] = [
            thermo_callback(
                logger_target="curator.simulate.callbacks.torchsim_logger.TorchSimThermoLogger",
                uncertainty_method=case.uncertainty_method,
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
            ),
            {
                "_target_": "curator.simulate.callbacks.trajectory.TrajectoryWriter",
                "path": str(case.run_dir / ("trajectory_sys{i}.traj" if case.batch else "trajectory.traj")),
                "interval": int(trajectory_interval),
                "per_system": bool(case.batch),
            },
            {
                "_target_": "curator.simulate.callbacks.summary.SimulationSummaryWriter",
                "path": str(case.run_dir / "simulation_summary.json"),
                "interval": int(summary_interval),
                "timestep_fs": float(case.timestep_fs),
                "compute_min_distance": summary_compute_min_distance,
                "compute_forces": summary_compute_forces,
                "initial_step_included": False,
            },
            {"_target_": "curator.simulate.callbacks.early_stop.EarlyStop"},
        ]

        adapter: Dict[str, Any] = {
            "_target_": "curator.interface.torchsim.CuratorTorchSimAdapter",
            "model": model_like,
            "cutoff": case.model.get("cutoff"),
            "compute_neighbor_list": bool(case.model.get("compute_neighbor_list", True)),
            "transforms": case.model.get("transforms"),
            "device": case.device,
            "load_compiled": bool(case.model.get("load_compiled", False)),
            "load_weights_only": bool(case.model.get("load_weights_only", False)),
            "return_cell_displacements": bool(case.model.get("return_cell_displacements", False)),
            "outputs": case.model.get("outputs"),
            "energy_scale": float(case.model.get("energy_scale", 1.0)),
            "forces_scale": float(case.model.get("forces_scale", 1.0)),
            "stress_scale": float(case.model.get("stress_scale", 1.0)),
            "detach": bool(case.model.get("detach", True)),
            "dtype": case.dtype,
        }

        return {
            "cfg": None,
            "seed": int(case.seed),
            "run_path": str(case.run_dir),
            "model_path": model_like,
            "dataset": case.reference_dataset,
            "device": case.device,
            "deploy": None,
            "simulator": {
                "_target_": "curator.simulate.core.simulator.Simulator",
                "init_traj": case.init_traj_for_config,
                "start_index": case.start_index_for_config,
                "batch": bool(case.batch),
                "engine": {
                    "_target_": "curator.simulate.engines.torchsim.TorchSimEngine",
                    "model": adapter,
                    "integrator": str(case.integrator),
                    "temperature": temperature_K,
                    "timestep": timestep_ps,
                    "integrator_kwargs": integrator_kwargs,
                },
                "run_kwargs": {"steps": int(case.steps)},
                "callbacks": callbacks,
            },
            "runner_metadata": {
                "created_at": utc_now(),
                "backend": self.name,
                "uncertainty_method": case.uncertainty_method,
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
            if resolved_case.batch:
                artifacts["trajectory"] = [
                    str(path)
                    for path in sorted(resolved_case.run_dir.glob("trajectory_sys*.traj"))
                ]
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

    def _case(
        self,
        request: SimulationRequest,
        *,
        case_index: int,
        label: str,
        structures: List[str],
        start_indices: List[int | None],
        structure_indices: List[int],
        ensemble: str,
        temperature_K: float,
        pressure_GPa: float | None,
        steps: int,
        timestep_fs: float,
        integrator: str,
        seed: int,
        batch: bool,
    ) -> SimulationCaseSpec:
        case_dir = _case_dir(request.run_dir, case_index, label)
        return SimulationCaseSpec(
            case_id=case_index,
            label=label,
            backend=self.name,
            task_type=request.task_type,
            criteria_profile=request.criteria_profile,
            init_structures=structures,
            start_indices=start_indices,
            structure_indices=structure_indices,
            ensemble=ensemble,
            temperature_K=temperature_K,
            pressure_GPa=pressure_GPa,
            steps=steps,
            timestep_fs=timestep_fs,
            integrator=integrator,
            seed=seed,
            run_dir=case_dir,
            timeout_sec=request.timeout_sec,
            model_paths=request.model_paths,
            reference_dataset=request.reference_dataset,
            uncertainty_method=request.uncertainty_method,
            device=request.device,
            dtype=request.dtype,
            protocol=dict(request.protocol),
            model=dict(request.model),
            batch=batch,
        )


def _case_dir(run_dir: Path, case_index: int, label: str) -> Path:
    case_dir = run_dir / f"case_{case_index:03d}_{safe_label(label)}"
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir
