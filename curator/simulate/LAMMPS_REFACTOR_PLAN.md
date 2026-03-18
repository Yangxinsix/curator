# LAMMPS Refactor Plan

## Goal

Refactor CURATOR's simulation stack so LAMMPS integrates as a first-class backend without forcing it into the old simulator abstraction.

The target architecture is:

- `workflow.py` orchestrates jobs and iteration hand-offs only.
- `curator.simulate.core.simulator.Simulator` remains the generic orchestrator.
- Each backend is implemented as an engine.
- Workflow hand-offs use declared output artifacts instead of backend-specific fields such as `simulator.out_traj`.

## Why This Refactor

The current codebase mixes two generations of simulation design:

- the newer `Simulator + BaseEngine + callbacks` stack
- the older backend-specific simulator contract assumed by `workflow.py`

This causes several issues:

- the workflow reads backend internals directly
- the default ASE path already writes outputs through callbacks, while the workflow still expects `simulator.out_traj`
- the LAMMPS config points to a legacy target that does not exist anymore

For LAMMPS, this mismatch is especially uncomfortable because LAMMPS is better treated as a shell-style execution backend with explicit artifacts, not as a Python-step-driven simulator.

## Target Boundaries

### Workflow

Workflow should only care about:

- model inputs
- initial structure inputs
- declared simulation outputs
- downstream files consumed by select and label

Workflow should not care about:

- how a backend writes trajectories
- whether uncertainty is computed via callbacks or external logs
- whether restart data is a trajectory, restart file, or something backend-specific

### Simulation Stack

- `Simulator` loads context, sets up the engine, and dispatches callbacks.
- `BaseEngine` defines backend execution.
- Backends publish standard artifacts through `outputs.*`.

### LAMMPS Stack

- `lammps_mliap_interface.py` stays as the model bridge.
- a future `LammpsEngine` handles execution
- I/O adapters convert LAMMPS data, dump, log, and restart files into CURATOR artifacts

## Standard Artifact Contract

Every simulation config should expose:

- `outputs.pool_set`: trajectory data used by active learning selection
- `outputs.uncertain_set`: optional trajectory of warning or outlier structures
- `outputs.restart_source`: optional source used to continue the next iteration
- `outputs.thermo_log`: human-readable thermo output
- `outputs.raw_dir`: backend-owned scratch or raw artifacts directory

The workflow should consume only these fields.

## Phase Plan

| Phase | Goal | Main Changes | Exit Criteria |
| --- | --- | --- | --- |
| 1 | Replace workflow's legacy simulate hand-off with `outputs.*` | Add top-level `outputs` config, wire ASE callbacks to it, update workflow to read declared outputs, keep deprecated fallback | Workflow no longer depends on `simulator.out_traj` for default configs |
| 2 | Introduce `LammpsEngine` as a shell-style backend | Add engine module, Hydra config, input writer, output parser, executable wiring | `curator-simulate simulator=lammps ...` produces `outputs.pool_set` |
| 3 | Move uncertainty and restart semantics to backend-owned artifacts | Standardize uncertain and restart outputs for LAMMPS | LAMMPS active-learning loop can feed select without ad hoc file handling |
| 4 | Remove legacy simulator-specific assumptions | Delete or rewrite old LAMMPS simulator config and deprecations | No runtime path depends on legacy simulator-only fields |

## Detailed Work Breakdown

### Phase 1

Scope:

- add top-level `outputs` to `configs/simulate.yaml`
- point default callback outputs to `outputs.*`
- update `workflow.py` to consume `outputs.*`
- preserve compatibility with old `simulator.out_traj`-style configs with warnings

Notes:

- restart hand-off should prefer `outputs.restart_source`
- if restart output is not declared, fallback to `outputs.pool_set`
- old keys such as `simulator.out_traj` and old uncertainty output fields should still work temporarily

### Phase 2

Scope:

- add `curator/simulate/engines/lammps/engine.py`
- add `curator/configs/simulator/engine/lammps.yaml`
- add LAMMPS input/output helpers under a dedicated I/O module
- keep the first version artifact-driven, not callback-driven

Minimum supported path:

- one system
- one model or deployed artifact
- run LAMMPS
- write `outputs.pool_set`
- optionally write `outputs.uncertain_set`

### Phase 3

Scope:

- formalize backend-native restart outputs
- extract uncertainty-tagged frames from LAMMPS artifacts
- make select consume the standardized files without backend-specific logic

### Phase 4

Scope:

- remove legacy `LammpsSimulator` assumptions from configs and docs
- document the difference between step-aware engines and shell engines
- make `workflow.py` backend-agnostic for simulation outputs

## Risks And Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Old user configs still depend on `simulator.out_traj` | Workflow breakage for existing users | Keep fallback logic with deprecation warnings during Phase 1 |
| Default trajectory writer emits per-system files while workflow expects one path | Select may read the wrong file | Standardize the default hand-off on a single combined file at `outputs.pool_set=${run_path}/trajectory.traj` |
| Restart semantics differ across backends | Iteration continuation becomes ambiguous | Make `outputs.restart_source` explicit and allow backend-specific override |
| LAMMPS shell execution has no per-step callback contract | Existing callback expectations may not transfer | Treat LAMMPS as artifact-driven first; add richer integration later only if justified |

## Acceptance Checklist

### Phase 1

- `simulate.yaml` defines top-level `outputs`
- default ASE callback configs resolve paths through `outputs.*`
- workflow uses `outputs.pool_set` and `outputs.uncertain_set`
- workflow restart hand-off prefers `outputs.restart_source`
- legacy output keys still work with warnings

### Phase 2

- a LAMMPS engine can run from `curator-simulate`
- generated outputs satisfy the same workflow contract as ASE

### Phase 3

- uncertainty outputs from LAMMPS can be fed directly into select
- restart continuation is explicit rather than inferred from backend internals

### Phase 4

- no maintained code path requires a backend-specific simulator contract
