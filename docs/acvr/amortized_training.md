# Backend-independent amortized variational training

The amortization contract is intentionally separate from every physical
backend.  A physical plugin owns the internal variable, functional, boundary
features, and proposal architecture.  The shared layer receives only:

1. one boundary-compiled `QuadraticProblem` per condition;
2. one flat proposal per condition;
3. an optional positive metric used only for hard projection.

Neither `prepare_amortized_training_batch` nor `AmortizedVariationalLoss`
imports or assumes charge, Ewald, dipoles, or any particular Green function.
Charge plus Ewald is one plugin that can produce a quadratic problem; it is not
the definition of ACVR.

## Two distinct training objectives

The physical functional and the proposal solve different learning problems.
They must not be allowed to hide each other's errors.

### Functional fitting

The physical plugin produces `F_theta(R, x)` and solves its compiled boundary
exactly.  Energy, force, response, and other physical labels update `theta`
through the exact/implicit solution.

### Proposal distillation

The proposal model produces `x_hat_phi(R, boundary)`.  The shared training
layer computes

```text
x_hat -> hard projection x_0 -> pre-correction KKT residual
                              -> exact detached teacher x_star
```

and trains `phi` with state distillation and the KKT residual measured at
`x_0`.  By default the physical problem tensors are detached inside this loss.
Thus a bad proposal must move toward the functional; the functional cannot
reduce the amortization loss by moving toward a bad proposal.

The exact teacher can be recomputed every call while the functional is being
updated.  Once the functional is frozen for proposal distillation,
`build_amortized_teacher_batch` can cache the exact states.  A cached teacher
must be rebuilt after any functional coefficient or boundary changes.

## Python contract

```python
from curator.train import (
    AmortizedTrainingModelOutput,
    AmortizedVariationalLoss,
)
from curator.variational import (
    build_amortized_teacher_batch,
    prepare_amortized_training_batch,
)

# The plugin constructs these.  They may describe charge, induced dipoles,
# another scalar/vector internal variable, or a synthetic analytic oracle.
problems = plugin.compiled_problems(data, conditions)
proposal = plugin.proposal_model(data, conditions)  # flat ragged tensor

# Cache only during a frozen-functional proposal-distillation stage.
teacher = build_amortized_teacher_batch(problems)
training = prepare_amortized_training_batch(
    problems,
    proposal,
    teacher=teacher,
    projection_metrics=plugin.projection_metrics(data, conditions),
)

prediction = {"amortized_training": training}
output = AmortizedTrainingModelOutput(
    AmortizedVariationalLoss(
        state_weight=1.0,
        kkt_weight=1.0,
        state_scale=plugin.state_scale,
    )
)
loss, n_geometries = output.calculate_loss(prediction, batch)
```

`batch["variational"]` supplies `condition_geometry_index` and optional
condition weights.  Reduction is geometry-balanced by default, so a geometry
with more control conditions is not automatically overweighted.

## What a plugin must provide

The shared code does not define a universal proposal network.  A plugin must
provide a proposal with the correct symmetry and boundary information:

- scalar atomic charge: scalar, permutation-equivariant output conditioned on
  charge/potential groups;
- induced dipole: vector-equivariant output conditioned on electric fields;
- constrained spin: manifold-aware output and projection;
- other variables: their own representation and physical controls.

The hard projector guarantees declared linear equalities.  It does not make an
inappropriate variable, functional, or boundary physically valid.

## Inference acceptance

Training success is measured by the inference policy, not only by loss:

```text
proposal -> projection -> KKT certificate
```

The primary metrics are zero-correction acceptance rate, average correction
count, fallback rate, and operator matvec count.  A `corrected` dense quadratic
result is numerically useful but is not evidence of amortization speedup; the
speedup occurs when the projected proposal is accepted without a solve.

The regression in `test/test_acvr_amortized_training.py` trains a
boundary-conditioned linear proposal for a two-component internal-variable
oracle through Curator's ordinary `run_step`.  The oracle contains no Ewald or
charge terms, and all trained conditions are subsequently accepted with zero
correction at the declared certification tolerance.
