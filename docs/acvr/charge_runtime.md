# ACVR charge runtime

`ACVRChargeRuntime` is the new boundary-conditioned charge path.  It does not
change the serialized behavior of the legacy `ChargeEquilibration` layer.

For condition $k$, the runtime constructs

$$
F_k(q)=\chi^\mathrm{T}q
+\frac12 q^\mathrm{T}[\operatorname{diag}(\eta)+K_\mathrm{Ewald}]q
-h_k^\mathrm{T}B_kq
+\frac12(C_kq-y_k)^\mathrm{T}K_k(C_kq-y_k),
\qquad A_kq=b_k,
$$

directly from a `CompiledBoundary`.  The returned `thermodynamic_energy` is the
minimum of this complete potential.  `base_energy` is the onsite plus Ewald
part before intensive controls and Robin environments.  The two names are not
interchangeable under fixed potential.

## Geometry and condition axes

The runtime accepts one geometry batch and an independent flat list of
conditions.  `condition_geometry_index[k]` selects which geometry condition
`k` uses.  Thus eight voltage conditions can reuse one local-model evaluation
and one Ewald kernel instead of storing or evaluating the coordinates eight
times.  Charge states, multipliers, constraints, and forces remain ragged and
carry explicit pointers.

## Solver and force modes

- `solver="implicit"` performs the exact dense KKT reference solve with an
  adjoint backward.
- `solver="amortized"` applies proposal, hardness-metric hard projection,
  residual certification, KKT correction, and optional exact fallback.
- `force_mode="envelope"` differentiates the complete potential at fixed
  state/multipliers.  At an exact KKT solution this is the relaxed force.
- `force_mode="full_differentiable"` differentiates through the supplied state
  path and is strictly the derivative of the reported finite-residual energy.
- `force_mode="adjoint_corrected"` provides a first-order correction for an
  approximate feasible state and is currently a first-derivative-only mode.

Unlike the legacy diagnostic Ewald force, the new force contains derivatives
of learned electronegativity, learned hardness, the Ewald kernel, intensive
controls, Robin terms, and geometry-dependent constraints.  Strain derivatives
produce virial and stress from the same thermodynamic potential.

## Training output

`ChargeRuntimeResult.as_variational_prediction()` emits physical-unit charge,
force, and normalized KKT-certificate fields for `VariationalJointLoss`.
Energy is omitted unless the caller explicitly chooses `condition_energy` or
`condition_free_energy`; the runtime cannot infer the dataset's Legendre and
reference convention from a tensor name.  This explicit choice prevents a
fixed-electron-number DFT energy from being silently treated as a fixed-voltage
grand potential.

The current implementation uses differentiable dense Torch kernels and is the
reference/training backend.  Large-scale production requires the separate
LAMMPS KSpace site-potential/matvec backend to satisfy the same operator parity
contract.
