# Convex charge EFA

`ConvexQConditionedEFAEnergy` is an opt-in explicit-state energy in
`curator/layer/efa/convex.py`. The ordinary geometry EFA readout remains
unrestricted. It uses the full representation feature specification.

At fixed geometry, one linear-attention call carries concatenated Values
`[V(h), q * B(h)]`. Q and K depend only on geometry. Split the attention output
into `g` and `z(q)`, and use

```
a(q) = Linear([g, z(q), q])
c = softplus(raw_output_weights)
eta = curvature_floor + softplus(raw_hardness)
U = U_baseline + energy_scale * sum(readout(g))
    + energy_scale * sum(c * (softplus(a(q)) - softplus(a(0))))
    + eta / 2 * sum(q**2)
```

The same affine map supplies `a(0)` without another attention pass. This is an
energy reference, not an equilibrium constraint. At `q=0`, the energy and
geometry derivatives equal the original geometry EFA branch.

Writing `a(q)=Aq+d`, the charge Hessian is
`energy_scale * A.T @ diag(c * sigmoid(a) * (1-sigmoid(a))) @ A + eta*I`.
It is bounded below by `eta*I`, on both the full space and any linear
fixed-charge constraint subspace. This holds for every finite q, irrespective
of signs in attention or affine weights. Positive hardness is outside
`energy_scale`. Charges are in e, energies in eV, hardness in eV/e².
The floor 0.01 and initial hardness 0.1 are explicit model settings, not
measurements of NaCl hardness. Positive definiteness does not establish
prediction accuracy or a small proposal residual.

## Framework integration

`VariationalEnergyOutput` adds the baseline and boundary terms once, uses the
existing `EFAChargeProposal` and Fukui projection, and computes forces through
the full proposal graph. Existing fixed-charge/fixed-potential data interfaces
remain applicable. Partial PBC remains unsupported by EFA.

`residual_gradient_policy="joint"` trains the energy, representation and
proposal from the same projected energy-gradient residual. It reuses the live
proposal state. The default `"proposal_only"` preserves the previous isolated
gradient behavior and checkpoint contract. Unknown policies are rejected.
No inner equilibrium solve is added to ordinary training.

For new models use `model/output_modules/efa_charge_convex.yaml`. For trained
plain EFA models, apply the native `efa_convex_charge` model transform before
optimizer construction, or use `acvr_efa_charge_convex.yaml` and supply paths
and boundary data. Conversion copies Q/K/V, attention and geometry readout;
charge parameters and proposal are initialized anew. Old nonconvex charge
parameters are not silently reinterpreted.

`test/test_efa_convex.py` checks the Hessian, finite differences and higher
derivatives, charge conservation, force/residual gradient routing, batching,
geometry transfer, symmetries and serialization. The NaCl experiment protocol,
optimizer grouping and results are separate scientific artifacts.
