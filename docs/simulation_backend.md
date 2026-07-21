# Simulation Backend Architecture

## Non-Negotiable Rule

Do not add short-term compatibility code to make a simulation backend appear supported.

Every backend must enter through the same request, backend, result, validation, and artifact contract. Unsupported features must fail explicitly. Silent fallback to another backend is not allowed.

## Public MCP Contract

The MCP-facing simulation API is intentionally small:

```python
list_simulation_engines()
run_simulation(task, system, model, protocol, backend_policy, decision_policy)
validate_simulation_result(result=None, summary_path=None, manifest_path=None, decision_policy=None)
```

There is no standalone `run_short_md` tool. Short MD is represented by `protocol.preset="short_md_smoke_v1"` and still enters through `run_simulation`.

## Current Flow

```text
curator_mcp/server.py
  -> curator_mcp/runners/simulation.py
  -> curator/simulate/simulation.py
  -> curator/simulate/backends/{ase,torchsim,lammps}.py
  -> curator/simulate/execution/subprocess.py
  -> curator/commands/simulate.py
  -> curator.simulate.core.simulator.Simulator
```

`curator/simulate/simulation.py` is the orchestration layer only. It normalizes the request, applies uncertainty provenance guardrails, selects a backend, executes backend-planned cases, aggregates results, calls backend-independent validation, and writes `simulation_matrix_manifest.json`.

Backend-specific details belong in backend modules:

- `curator/simulate/backends/ase.py`
- `curator/simulate/backends/torchsim.py`
- `curator/simulate/backends/lammps.py`

Shared result parsing belongs in `curator/simulate/results.py`. Shared subprocess execution belongs in `curator/simulate/execution/subprocess.py`.

## Request Objects

`SimulationRequest` is created from JSON-like MCP input:

```python
SimulationRequest(
    task={...},
    system={...},
    model={...},
    protocol={...},
    backend_policy={...},
    decision_policy={...},
    run_dir=...,
    timeout_sec=...,
)
```

Required user-facing fields:

- `task.type`: currently `md` or `md_stability` for implemented local backends.
- `task.criteria_profile`: validation rules, usually `md_direct_use_validation_v1` or `md_smoke_v1`.
- `system.init_structures`: ASE-readable structure path or list of paths.
- `system.start_indices`: optional structure indices.
- `model.model_path`: model checkpoint path or list of model paths.
- `model.uncertainty_method`: `auto`, `none`, `ensemble`, or `mahalanobis`.
- `protocol`: physical simulation intent.
- `backend_policy.mode`: `auto`, `ase`, `torchsim`, or `lammps`.

## Protocol Presets

Presets are versioned protocol defaults. User-supplied fields override preset defaults.

### `short_md_smoke_v1`

Purpose: quick local smoke check that the model/system can run and emit validator-ready artifacts.

Defaults:

- `task.type = md`
- `task.criteria_profile = md_smoke_v1`
- `protocol.ensembles = ["nvt"]`
- `protocol.temperature_K = [300.0]`
- `protocol.steps = 100`
- `protocol.timestep_fs = 0.5`
- `protocol.integrator = "langevin"`

### `md_direct_use_probe_v1`

Purpose: direct-use MD validation probe.

Defaults:

- `task.type = md`
- `task.criteria_profile = md_direct_use_validation_v1`
- `protocol.ensembles = ["nve", "nvt"]`
- `protocol.temperature_K = [300.0, 600.0, 900.0]`
- `protocol.steps = 1000`
- `protocol.timestep_fs = 0.5`
- velocity initialization enabled

`protocol.preset` names the simulation matrix. `task.criteria_profile` names the validator policy. Do not use `md_direct_use_probe_v1` as a criteria profile.

## Backend Contract

Every backend implements:

```python
class SimulationBackend:
    name: str

    def capabilities(self) -> SimulationBackendCapabilities: ...
    def validate_request(self, request: SimulationRequest) -> None: ...
    def plan_cases(self, request: SimulationRequest) -> list[SimulationCaseSpec]: ...
    def build_config(self, case: SimulationCaseSpec) -> dict: ...
    def run_case(self, case: SimulationCaseSpec, *, tool_name: str = "run_simulation") -> dict: ...
```

The common `validate_request` contract enforces:

- backend is implemented
- backend is available
- task type is supported
- NPT or stress-dependent protocols have explicit stress capability provenance

Stress provenance is explicit. For NPT, the model input must include one of:

```json
{"stress_capable": true}
{"supports_stress": true}
{"capabilities": {"stress": true}}
{"outputs": ["energy", "forces", "stress"]}
```

If stress capability is unknown, the request fails before execution with `StressCapabilityRequired`.

## Implemented Backends

### ASE

Implemented for local `md` and `md_stability`.

Uses Curator's ASE `MDEngine` through the standard `Simulator` command path. It supports NVE, NVT, and NPT, but NPT requires explicit stress capability provenance.

### TorchSim

Implemented for local `md` and `md_stability`.

Uses `CuratorTorchSimAdapter` and `TorchSimEngine` through the standard `Simulator` command path. Batch execution is represented directly in `SimulationCaseSpec.batch` and writes aggregate summary metrics plus per-system trajectories.

### LAMMPS

Registered but not implemented for local MCP `run_simulation`.

LAMMPS should enter through deployment-equivalence and scheduler-aware submit/status/collect support. It must not be approximated with ASE or TorchSim fallback.

## Result Schema

Every backend returns the same case result schema:

```json
{
  "ok": true,
  "status": "completed",
  "backend": "ase",
  "uncertainty_method": "none",
  "steps_requested": 1000,
  "steps_completed": 1000,
  "early_stopped": false,
  "early_stop_reason": null,
  "max_uncertainty": {},
  "warning_steps": 0,
  "outlier_steps": 0,
  "drift": {},
  "thermo": {},
  "force": {},
  "structure": {},
  "performance": {
    "simulation_walltime_sec": 1.23,
    "subprocess_walltime_sec": 1.75,
    "steps_completed": 1000,
    "natoms": 128,
    "atom_steps": 128000,
    "simulated_time_ps": 0.5,
    "steps_per_second": 813.0,
    "atom_steps_per_second": 104064.0,
    "subprocess_steps_per_second": 571.4,
    "subprocess_atom_steps_per_second": 73142.9
  },
  "summary": {},
  "case": {},
  "artifacts": {},
  "returncode": 0
}
```

The matrix result returned by `run_simulation` is:

```json
{
  "ok": true,
  "status": "completed",
  "backend": "ase",
  "task": {},
  "protocol": {},
  "uncertainty_evidence": {},
  "cases": [],
  "skipped": [],
  "decision": {},
  "reliable_for": [],
  "not_reliable_for": [],
  "metrics": {},
  "performance": {
    "num_cases": 6,
    "total_steps_completed": 6000,
    "total_atom_steps": 768000,
    "total_simulated_time_ps": 3.0,
    "total_simulation_walltime_sec": 7.4,
    "total_subprocess_walltime_sec": 10.8,
    "effective_steps_per_second": 810.8,
    "effective_atom_steps_per_second": 103783.8,
    "effective_subprocess_steps_per_second": 555.6,
    "effective_subprocess_atom_steps_per_second": 71111.1
  },
  "criteria_policy": {},
  "artifacts": {
    "run_dir": "runs/...",
    "manifest": "runs/.../simulation_matrix_manifest.json"
  }
}
```

The validator must read only this backend-neutral schema. It must not branch on `backend == "ase"` or `backend == "torchsim"`.

## Validation Profiles

`md_smoke_v1` is permissive and intended to prove the tool path can execute.

`md_direct_use_validation_v1` is strict and intended for direct-use decisions. It checks:

- failed cases
- skipped cases
- early stops
- uncertainty warning/outlier fractions
- NVE total-energy drift
- minimum interatomic distance
- maximum force
- Mahalanobis reference independence

Mahalanobis direct-use validation requires independent reference provenance:

- `training_reference`
- `calibration_reference`
- `dft_audit_reference`

Production trajectories, initial structures, smoke references, and unknown roles cannot pass direct-use validation.

## Artifact Contract

Each case writes:

- `input_config.yaml`
- `simulate_stdout.txt`
- `simulate_stderr.txt`
- `simulation.log`
- `simulation_summary.json`
- `simulation_manifest.json`
- trajectory artifact(s)

Each case summary/result must expose runtime information through `performance`. `simulation_walltime_sec` measures the simulator callback runtime inside the subprocess. `subprocess_walltime_sec` measures the full command wall time including Python startup, config loading, model loading, and output writing. Model/backend choice, distillation decisions, and deployment conversion decisions should use these values rather than parsing logs.

The matrix run writes:

- `simulation_matrix_manifest.json`

Artifacts must be absolute or run-directory-resolvable paths and should be reported through the `artifacts` field.

## Required Conformance Tests

Every backend addition or contract change must keep tests for:

- no standalone `run_short_md` MCP tool
- `short_md_smoke_v1` preset
- `md_direct_use_probe_v1` preset
- ASE request validation
- TorchSim batch case planning
- LAMMPS explicit not-implemented behavior
- Mahalanobis direct-use guard
- direct-use validator pass/fail
- NPT stress capability preflight

Backend-specific tests may add real execution smoke tests, but the contract tests must stay fast and deterministic.
