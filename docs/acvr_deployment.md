# ACVR deployment output policy

The default model and old checkpoints retain the existing training and evaluation
contract. The following switches are evaluation-only and change no parameters or
energy definitions.

```python
model.eval()
head = next(m for m in model.output_modules if getattr(m, "joint_variational_output", False))
head.configure_deployment(
    compute_residuals=False,
    reuse_constraint_projection=True,
)
prediction = model(batch)
# prediction["variational_prediction"]["residuals"] == {}
# metadata["residuals_computed"] is False
head.reset_deployment()  # required before training again
```

This skips the second, detached-feature proposal and residual branch. The first
proposal, hard constraints, full energy and conservative coordinate derivative
remain connected. It never replaces missing stationarity measurements by zeros
or labels a predicted state as converged.

Projection reuse validates the live equality tensors once per condition, then
reuses the small constraint Gram matrix for the feasible origin, initial state,
final state and tangent field. A one-row system uses exact scalar division.
Multiple independent equalities retain a small linear solve. Nothing is cached
across forward calls, and derivatives through geometry-dependent constraints and
live boundary targets are preserved. Redundant and inconsistent constraints are
still rejected. Projection reuse requires residual diagnostics to be disabled.

For energy-only deployment, add `energy_only=True`. This omits built-in forces,
configured observables and field responses without mutating the saved training
specifications. Use the analytic Taylor backend to eliminate nested charge
autodifferentiation where exact derivatives of the energy are implemented.
With that backend, the complete model also honors `torch.no_grad()` when
energy-only deployment disables residuals:

```python
head.internal_energy.derivative_backend = "analytic"
head.configure_deployment(energy_only=True, reuse_constraint_projection=True)
with torch.no_grad():
    prediction = model(batch)
```

The model checks an explicit `allows_no_grad_evaluation` capability on every
module requiring energy gradients. Unknown derivative modules retain the usual
gradient-enabled behavior. Force inference still requires coordinate gradients;
`torch.inference_mode()` is unsupported.

Tests are in `test/test_acvr_deployment.py`. Benchmark the same checkpoint,
device, dtype, batch and outputs; compare diagnostics-on and diagnostics-off
timings separately.


For batched conservative force inference, models whose head already declares
`batch_observables=True` can also opt into `batch_forces=True`. That existing
contract asserts independence between geometries throughout the backbone and
energy graph; merely sharing trainable weights is fine.

```python
head.configure_deployment(
    compute_residuals=False,
    reuse_constraint_projection=True,
    batch_forces=True,
)
```

When every condition belongs to a distinct geometry, one derivative of the sum
of their live energies replaces repeated backwards through the batched backbone.
Force slices remain in condition order, including reordered subsets and variable
atom counts. Repeated conditions on the same geometry automatically use separate
derivatives, preserving distinct boundary-condition forces. The output metadata
`batched_force_derivative` records which branch actually ran. Energy-only calls
do not use this option. The option resets to false on a subsequent
`configure_deployment()` call unless explicitly requested again.


## Periodic Ewald neighbor search

For derivative + Ewald, an independent opt-in replaces Python ASE neighbor
enumeration with Curator's existing compiled matscipy dependency:

```python
head.internal_energy.ewald.neighbor_backend = "matscipy"
```

The default remains `"ase"`, including old checkpoints. Explicit `"matscipy"`
requires that optional package; `"auto"` uses matscipy when installed and falls
back to ASE otherwise. Every forward constructs fresh atom pairs and integer
periodic shifts, including multiple images when the real-space cutoff exceeds
half the cell length. Live displacements remain `R[j] - R[i] + S @ cell`, so
coordinate and cell derivatives are preserved. No positions, cells or neighbor
topology are cached; changing geometry, cutoff or periodicity is supported.
Partial periodicity remains explicitly unsupported, as before.

This selection is independent of `derivative_backend`. The analytic Taylor
backend is mathematically equivalent but float32 operation ordering can change
roundoff: a frozen-checkpoint GPU Ewald check observed force RMS difference of
about `1.1e-4 eV/Angstrom` from the original derivative path. Applications that
need the original derivative numerics can keep
`head.internal_energy.derivative_backend = "autograd"` while still disabling
residual diagnostics, reusing validated projections, batching force derivatives
and using matscipy. Compare frozen-checkpoint energies and forces at the intended
dtype before choosing a deployment configuration; float64 equivalence alone
does not establish float32 equivalence at every checkpoint.

