# Calibrated two-scale convex q-EFA

`TwoScaleQConditionedEFAEnergy` reuses the original convex energy's single EFA
attention, geometry readout, charge-gated Values, positive output weights and
local hardness. It keeps the full declared backbone feature tensor.

With geometry-only offset `b`, affine charge response `t = A(R)q`, positive
output weights `c`, and `s(x) = softplus(x)`, its correction is

```
L = sum(c * sigmoid(b) * t)
D = sum(c * (s(b + t) - s(b) - sigmoid(b) * t))
E = energy_scale * geometry_readout
    + driving_scale * L + curvature_scale * D + hardness * sum(q²) / 2
```

The driving scale is signed and independently trainable. The curvature scale
is a softplus-parameterized positive scalar. For fixed geometry,
`H_q = curvature_scale * Aᵀ diag(c * s''(b+t)) A + hardness * I`, so the same
strict positive curvature lower bound holds for every driving scale. Both
scales equal to the original `energy_scale` recover the old convex energy.
At `q=0` the charge correction and its geometry derivatives vanish exactly.

Use the `efa_two_scale_charge` model transform on a plain trained EFA adapter
before constructing the optimizer. It preserves the learned geometry branch
and uses the existing `VariationalEnergyOutput`, charge proposal, Fukui
constraint correction and boundary interfaces. The original
`efa_convex_charge` transform and checkpoint schema are unchanged. The Hydra
entry point is `acvr_efa_charge_two_scale`; its default scales preserve the
old mathematical form and are not a claimed successful calibration.

For diagnostics, `response_terms(context, state)` returns differentiable
`geometry_energy`, `linear`, `curvature`, `offset` and `response` tensors from
one attention evaluation. The geometry energy is already scaled; the linear
and curvature terms include output weights but exclude their scalar scales.
`set_scales(driving_scale=..., curvature_scale=...)` changes existing parameter
values in place. `from_convex(energy, **scale_options)` copies every parameter
from a bound original convex energy. Ordinary checkpoint loading must use a
matching schema; conversion is explicit.

Calibrate charge input derivative scales and curvature on training data, then
select driving strength by held-out equilibrium energy/force performance.
Freeze geometry EFA and shared Q/K along with the backbone during charge
warmup. The new transform defaults to `proposal_only` residual gradients:
stationarity trains the proposal while energy/force labels train the energy.
The calibrated values and warmup schedule belong to the experiment config.

Validation must include old-form equivalence, charge Hessian bounds, mixed
geometry/charge/parameter derivatives, and checkpoint integration. A small
stationarity RMS is insufficient: compare equilibrium energies and forces
with both the geometry model and the same charge model evaluated at uniform
charge, on a fixed held-out split. Do not use charge labels or choose a target
charge magnitude just to make the response larger.
