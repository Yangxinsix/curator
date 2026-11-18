# Training callback boundaries (opt-in convergence protocol v2)

This refactor changes training infrastructure, not model architectures, energy,
forces, losses, datasets, or charge supervision. Default Curator configurations
are unchanged. The existing `convergence_all_20261010` campaign remains on its
frozen v1 protocol; queued jobs import its byte-identical controller from
`compat/convergence_v1.py`. New protocol settings must be selected explicitly.

## Responsibilities

| Component | Sole responsibility | Does not own |
|---|---|---|
| PyTorch `ReduceLROnPlateau` | Change optimizer learning rates from validation loss | Stopping, file I/O, physical metrics |
| `LearningRateGatedEarlyStopping` | Decide whether ordinary plateau stopping is allowed | LR changes, checkpoint writes, metrics-file parsing |
| Existing physical monitors | Compute/log energy, force and applicable stationarity diagnostics | Scheduling or stopping |
| Native `ModelCheckpoint` | Select/save the best validation model | Full continuation archive |
| `ResumeCheckpoint` | Persist latest full training state after scheduler updates | Model selection or stopping policy |
| `ResumeTrainingState` | Verify restored optimizer/step/scheduler and normalize completed-epoch loop state | LR policy, metric interpretation |
| Campaign driver / Trainer | Resource limits, manifests, evaluation and reports | Model physics |

The early-stopping class subclasses Lightning EarlyStopping. Its ordinary
plateau stop is gated by LR floor, observations at that floor, and physical
metric plateaus. Native nonfinite/explicit-threshold stopping remains active;
another callback's stop request is preserved. Invalid physical metrics raise an
error. Monitors must log their metrics every validation. These checks do not
detect stale entries left in `trainer.callback_metrics` by a broken monitor.

**A plateau is not scientific acceptance.** A high stationarity residual that
stops improving is still high. Report its magnitude separately; do not interpret
an early stop as evidence that ACVR has solved its variational problem.

## Opt-in configuration

Existing `LitNNP.configure_optimizers` already supports native scheduler factories;
no model or optimizer setup rewrite is needed. Added presets:

- `task/scheduler/convergence_plateau.yaml`
- `trainer/callbacks/lr_gated_early_stopping.yaml`
- `trainer/callbacks/resume_training_state.yaml`
- `trainer/callbacks/resume_checkpoint.yaml`

Set `task.scheduler_monitor` explicitly to the objective already used for model
selection. Use the same monitor for scheduler, early stopping, and ModelCheckpoint.
Replace any existing EarlyStopping callback; do not append a second stopping
policy. Use native ModelCheckpoint with `save_last: false`; place the physical
monitor before stopping, and recovery before resume checkpoint. For configurations
whose callbacks are a list, instantiate/add these preset contents as list entries.
For Hydra's default dictionary configuration, select them in the callback groups.

Example effective configuration (merge into the existing model/data configuration):

```yaml
task:
  scheduler:
    _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
    _partial_: true
    mode: min
    factor: 0.5
    patience: 7
    threshold: 0.005
    threshold_mode: rel
    cooldown: 3
    min_lr: 1.0e-6
  scheduler_monitor: val_physical_total_loss
trainer:
  # Retain the model's physical monitor first.
  # Add stationarity to physical_metrics for response models.
  callbacks:
    physical:
      _target_: curator.train.ef_monitor.EFPhysicalMonitor
      output_path: ${run_path}/physical_metrics.jsonl
    restore:
      _target_: curator.train.resume_callbacks.ResumeTrainingState
      output_path: ${run_path}/resume/restoration.json
    stopping:
      _target_: curator.train.convergence.LearningRateGatedEarlyStopping
      monitor: ${task.scheduler_monitor}
      mode: min
      patience: 8
      physical_metrics:
        - val_physical_energy_rmse_meV_per_atom
        - val_physical_force_component_rmse_eV_per_Angstrom
    resume:
      _target_: curator.train.resume_callbacks.ResumeCheckpoint
      dirpath: ${run_path}/resume
    best:
      _target_: pytorch_lightning.callbacks.ModelCheckpoint
      dirpath: ${run_path}/model_path
      monitor: ${task.scheduler_monitor}
      mode: min
      save_top_k: 1
      save_last: false
```

For ACVR use its existing ACVRPhysicalMonitor and objective (including residual
loss), and include `val_physical_stationarity_rms_V` in the stopping metrics.
For legacy models keep the existing LegacyBoundaryMonitor/ConservativeQEqMonitor
and `val_legacy_total_loss`. Do not replace their objectives with this EF example.
Scheduler `patience=7` means reduction after **eight** unsuccessful checks because
PyTorch compares `num_bad_epochs > patience`. The default stopping gate needs
12 validations at the same floor LR and compares two six-validation median
windows; >0.5% improvement or >2% deterioration keeps the gate closed.

## Checkpoints and migration

Lightning steps epoch-based ReduceLROnPlateau after epoch-end callbacks. Therefore
an ordinary best checkpoint can precede that LR update. `ResumeCheckpoint` saves
at the next epoch start and at fit end, tagging the archive
`resume_after_scheduler`; ordinary best saves are tagged
`evaluation_before_scheduler`. Use `resume/last.ckpt` for full continuation.
Best remains usable for evaluation, test, prediction, or explicit weights-only
initialization; it is rejected for full fit restoration by the recovery callback.

A legacy checkpoint without a scheduler restores model, Adam moments, parameter
groups/LR and global step. Scheduler and stopping histories start fresh. This is
an explicit protocol transition, not an exact recreation of v1's ad-hoc controller
history. Existing jobs are not automatically migrated. Compatibility code that
repairs completed-epoch restart state uses Lightning internals and is covered by
real training-loop tests; rerun these tests when changing Lightning versions.

Ordinary DataLoader checkpoints do not preserve iterator/sample position or RNG.
Mid-epoch recovery preserves optimizer updates, but bitwise data/RNG replay is
not claimed. Saving once per epoch does not guarantee recovery of the newest
updates after a hard kill inside an epoch.

## Regression validation

`tests/test_acvr_convergence.py` uses tiny real Lightning training loops, not
mocked callback calls. Coverage includes native LR trajectory, floor/physical
stopping gates, invalid metrics, external stop preservation, complete Adam and
scheduler state restoration, old epoch-end and new epoch-start/fit-end archives,
mid-epoch continuation, and best checkpoint evaluation versus fit restoration.
Runtime-specific results and source backups are stored outside Git under
`experiments/callback_refactor_20261010`.
