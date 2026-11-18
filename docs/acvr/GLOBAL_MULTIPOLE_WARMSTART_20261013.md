# Global nonlinear state energy: finite-system warmstart prototype

The 2026-10-13 implementation adds an independent global energy and proposal to
an already fitted Curator MACE model. Training labels are energy and forces only.
Neither atomic charges nor atomic/molecular dipoles are supervised. Total charge
is a boundary constraint, not an atomic-charge target. The old quadratic, Ewald,
LES and EFA implementations are retained for comparisons.

## Energy and state

For each atom the variational state is x_i=(q_i,p_ix,p_iy,p_iz), with charge in e
and dipole in e Angstrom. Its representation is 1x0e + 1x1o. The internal energy is

    U(R,x) = E_fitted_MACE(R) + Delta_U_theta(R, h_MACE(R), x).

Delta_U is an unrestricted nonlinear invariant neural energy, not a Taylor
polynomial, prescribed Coulomb kernel, or separately predicted Hessian. Derivatives
and HVPs are derivatives of this same scalar. Two global equivariant attention
blocks process 64x0e + 16x1o + 8x2e hidden features. Query/key weights depend on
geometry features; nonlinear messages also depend on the current state. All atom
pairs participate. Learned radial tensor-product weights use smooth radial basis
functions; `max_length` is a basis scale (32 Angstrom for NaCl, 82 for SA-BS),
not a hard neighbor cutoff. Complexity
is quadratic in atom count. This prototype explicitly rejects periodic inputs.

The MACE internals stay unchanged. `response_feature_spec` binds full equivariant
features to an e3nn linear adapter. Other backbones can use that same interface if
they supply compatible invariant/equivariant features; this campaign validates
MACE only. State dimensions do not depend on the backbone feature width. This does
not make inferred states mathematically independent of the chosen backbone.

The proposal has independent parameters and two blocks of the same family. It
reads geometry features, a feasible origin and projected conjugate driving fields,
then predicts a state correction. A final exact projection imposes A x=b. Projection
solves a small constraint Gram system, not an atom-by-atom Hessian inverse.
The first proposal reads the origin and projected field, but does not read the
constraint membership matrix itself. Different regional partitions with identical
origins/fields are thus indistinguishable before projection. Feasibility is enforced
for them, but universal regional-boundary amortization is not implemented or
validated; this campaign only uses a single total-charge constraint per system.

## Boundary convention

At fixed geometry the quantity minimized is

    G(R,x) = U(R,x) + sum_i q_i phi_i
             - E_uniform . sum_i(q_i R_i + p_i)
             - sum_i E_local_i . p_i - v_control^T B q,
    subject to A x = b.

All controls are explicit external work; the intrinsic neural energy does not read
the boundary context. Region charge constraints and voltage controls reuse the
existing compiler. q occupies every fourth state component; dipoles have no total-
charge constraint. Spatially varying externally supplied phi/E tensors must retain
their coordinate graph when their gradients are required. Robin environments and
multiple conditions per geometry are rejected in this first adapter.

The stationary residual is the tangent projection of dG/dx. q residuals are in V,
and dipole residuals in V/Angstrom. They have separate reported RMS and separate
fixed normalization scales; they must not be combined as if their units matched.
Residual zero alone does not establish a minimum. Unconstrained neural energy is
not guaranteed convex, coercive, fragment additive, or to have physical Coulomb
asymptotics. These are validation questions, not properties established by this
implementation. Atomic q/p are latent physically coupled states, not a claim of
agreement with a particular DFT partitioning scheme.

## Training and gradient ownership

The physical forward uses G(R,proposal(R,boundary)) and differentiates the full
coordinate path to compute forces. Energy/force losses update all unfrozen energy,
proposal and MACE parameters. A second deterministic state evaluation uses detached
geometry/backbone features and detached energy parameters. Only proposal parameters
receive gradients from stationarity penalties. This prevents residual-only training
from flattening the internal energy. The physical and residual proposal states must
agree. There is no charge-label or dipole-label loss.

Use `augment_multipole_model(fitted_model, adapter)` to compose the model. It retains
physical-unit energy rescaling and the original fitted energy reference. Optimizer
groups `state_energy` and `proposal` are separate from existing MACE groups. Full
coordinate grad mode is enabled even in validation; inference_mode is rejected.
The wrapper is opt-in and existing serialized model classes are unchanged, so no
legacy checkpoint migration is required.

The first campaign uses NaCl (Q=+1e) and ACDB SA-BS (Q=-1e), with the original fixed
train/validation/test splits and immutable copies of fitted pure-MACE checkpoints.
Each system has pure-MACE continuation, fitted-MACE+EFA and fitted-MACE+new-ACVR.
Adapters warm up for three epochs with a frozen backbone, then all parameters train
jointly. MACE lr=1e-5; new modules lr=1e-4. Energy and force weights are 200 each;
normalized q/p residual weights are 1 each. Stage changes are explicit weights-only
transitions; interruption recovery within a stage restores optimizer and scheduler.
Validation plateau and learning-rate-floor callbacks determine early stopping;
300 additional epochs is a safety cap, not evidence of convergence.

## Required checks and interpretation

Tests cover O(3), translations, permutation, derivative finite differences, HVPs,
higher derivatives, boundary work, nonzero total charges, gradient ownership,
serialization, and full-model eval/optimizer integration. Training smoke checks
exercise warmup, joint updates and optimizer/scheduler restoration. Formal GPU runs
must depend on successful GPU smoke jobs. CPU validation cannot satisfy GPU gates.

Report E/F errors together with physical q/p stationarity, charge feasibility,
state amplitudes, timing and GPU memory. Improvement in E/F alone does not prove
the state is useful: compare learned proposal with feasible-origin states and a
small same-functional reference solve. That solve is a diagnostic dense nonlinear
KKT root solver with a reduced-curvature check, not a production HVP solver nor a
proof of global minimality. Record failures and saddles explicitly. Validation on
fixed-Q, zero-field data does not establish accuracy at unseen voltages/fields.

Experiment artifacts and live status:
`/data1/home/hzxy05/work/yangxin/acvr/experiments/acvr_global_state_warmstart_20261013`.
