# Controlled GNN / proposal optimization

`curator.train.variational_optimization.VariationalOptimizationTask` extends
`LitNNP`. It changes update scheduling only; it preserves the model, physical
loss definitions, force derivatives, validation, metrics, and full checkpoints.
It is an opt-in task, not a replacement for the default trainer.

| Protocol | `update_rule` | `inner_steps` | Initializer LR multiplier |
| --- | --- | --- | --- |
| C: joint control | `joint` | 0 | none |
| D: faster proposal | `joint` | 0 | `proposal: 4.0` |
| A0: residual steps then joint | `alternating_joint` | 2 | none |
| A1: residual steps then physical parameters | `alternating_ef` | 2 | none |

An outer batch contains `accumulation_steps=3` pristine minibatches. Each inner
step uses the original weighted stationarity loss on that same batch, and only
the named `proposal` optimizer group receives gradients. A0 then performs the
original joint update. A1 updates every non-proposal group with the original
weighted energy and force losses. The proposal's dependence on coordinates and
GNN features remains differentiable during A1; no stop-gradient force or
surrogate energy is introduced. One Adam instance retains every group's state.
Inactive parameters have `grad=None`, so their moments and weights do not move.

## Configuration and initialization

Keep the existing model, outputs, optimizer, datasets and monitor callbacks.
Select the task above and set:

```yaml
task:
  update_rule: alternating_ef
  inner_steps: 2
  accumulation_steps: 3
  gradient_clip_norm: 10.0
  proposal_group: proposal
  scheduler: null
  warmup_steps: 0
trainer:
  accumulate_grad_batches: 1
  gradient_clip_val: null
  precision: 32-true
  devices: 1
```

Fixed learning rates are supported for controlled pilots. Convergence runs may
use native `torch.optim.lr_scheduler.ReduceLROnPlateau` and
`task.scheduler_monitor=val_joint_predicted_loss`. Other schedulers, warmup,
mixed precision, multiple devices, combined dataloaders, and losses other than
one `VariationalModelOutput` with energy, force and stationarity terms are
rejected. Alternating runs also reject running BatchNorm and stochastic
dropout, whose extra forward passes would change the comparison.

Start a **new branch** by strictly loading the source model `state_dict`, and
attach `curator.train.optimizer_initialization.OptimizerStateInitializer` with
`state_path` and optional `lr_multipliers`. The file is created with `torch.save`:

```python
{
    "optimizer_state": source_checkpoint["optimizer_states"][0],
    "parameter_names": [["model.<name>", ...], ...],
    "provenance": {"source_checkpoint": "...", "sha256": "..."},
}
```

`parameter_names` must list source parameter names in exact optimizer-group and
parameter order. `optimizer_parameter_names(task, optimizer)` supplies this list
when inspecting the source task. The initializer checks names, groups, tensor
shapes, and every restored Adam state value. LR multipliers affect group metadata
only. The initializer records a source-file SHA-256 and LR audit in its callback
state; it does not train, schedule learning rates, save models, or stop training.

A new branch resets epoch/data-stream history. It must not be described as a
complete continuation of the source Lightning loop, RNG, or scheduler state.
A later **full resume of the branch** uses the normal Lightning checkpoint path,
the same task protocol and initializer callback. Restored callback state prevents
reinitializing Adam or multiplying LR again, even if the source file is gone.

## Native plateau scheduling and convergence callbacks

For a convergence run, replace `scheduler: null` with the existing native
ReduceLROnPlateau factory. Retain `JointValidationLoss`, the physical monitor,
`LearningRateGatedEarlyStopping`, `ResumeTrainingState`, and `ResumeCheckpoint`.
The task steps the scheduler in `on_train_epoch_end`, after callbacks and after
this epoch's validation loss has been aggregated. It skips epochs without a new
validation and never steps twice for the same epoch. The scheduler state remains
owned and serialized by Lightning; no convergence controller is introduced.
Epoch-end validation (`val_check_interval=1.0`) is required with scheduling.

With Hydra, convert OmegaConf containers before calling the native PyTorch
scheduler. For the D branch use:

```yaml
task:
  scheduler_monitor: val_joint_predicted_loss
  scheduler:
    _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
    _partial_: true
    _convert_: all
    mode: min
    factor: 0.5
    patience: 7
    threshold: 0.005
    threshold_mode: rel
    cooldown: 3
    min_lr: [1.0e-6, 1.0e-6, 1.0e-6, 4.0e-6]
    eps: 0.0
```

`_convert_: all` is required for a sequence of floors: PyTorch recognizes Python
lists and tuples, but treats an unconverted OmegaConf `ListConfig` as a scalar
and repeats that container instead of receiving four numeric group floors.

Supply proportional LR floors to preserve group ratios. For ordered groups
`backbone, baseline, functional, proposal` and proposal LR ×4, use
`min_lr: [1e-6, 1e-6, 1e-6, 4e-6]`; C/A can use scalar `1e-6`. The task rejects
floors that would change the ratios. `eps: 0` avoids a native per-group epsilon
cutoff changing ratios for very small updates. Configure the existing stopping
callback's `min_lr` as the largest group floor (D: `4e-6`; C/A: `1e-6`); because
the ratios are fixed, that gate opens exactly when all groups reach their floors.
Use identical physical plateau tolerances and patience policies for every arm.

`ResumeCheckpoint` writes post-scheduler state at the next epoch start and fit
end. Best checkpoints saved earlier remain evaluation checkpoints; use
`resume/last.ckpt` with `ResumeTrainingState` for full continuation. The task also
persists its last scheduled epoch alongside update counters.

## Accumulation, checkpoints and measurements

The final incomplete accumulation window is flushed, dividing by the configured
accumulation factor as native Lightning does. For five minibatches with factor
three, the last step sums two losses divided by three. Checkpoints must occur at
completed outer-update boundaries; pending windows are rejected, not silently
lost. Validation may run at epoch end or an integer interval divisible by the
accumulation factor.

Lightning `global_step` counts **all** optimizer steps. A0/A1 with two inner steps
advance it three times per outer update; C/D advance it once. Compare exposure
using `optimization/outer_steps`, `ef_structures`, `residual_only_structures`,
`inner_steps`, and `forward_minibatches`, which are also persisted under the
checkpoint's `variational_optimization` field. Do not use an equal `max_steps` to
claim equal physical-model training. Phase-specific pre-clipping gradient norms
and clipping factors are logged. Inner-only forwards do not enter ordinary
training loss averages; full validation remains inherited from `LitNNP`.

These short fixed-LR branches test mechanisms, not final convergence. Compare
energy, force, stationarity and tail stability at both matched outer exposure and
matched elapsed time, then advance candidates to a separately documented
convergence protocol before claiming a general training recommendation.
