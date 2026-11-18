# Geometry-aware shared scalar energy for the long-range benchmark

This is an explicit opt-in addition. Existing NeuralChargeEnergy,
NeuralChargeStateEnergy, QuadraticTaylorEnergy, and old checkpoints are unchanged.

## Why it is needed

The old scalar energy uses local backbone features h_i and pooled charge modes.
If two disconnected local neighbor graphs are rigidly separated further, their
features do not change. Their pooled charge energy therefore cannot know the
new inter-fragment separation. HVP accurately differentiates that old energy but
does not supply missing geometry. The new energy reads all-pairs geometry.

The existing BoundaryConditionedProposal(communication='attention') already reads
all-pairs relative geometry and is selected for BOTH energy arms. The proposal
architecture is not changed in this patch. The initial charge q0 still comes
from the same local invariant feature readout; proposal learns its correction.

## Scalar energy, not an independently predicted charge Hessian

Let U_old be the existing convex scalar energy and let q_i denote internal
charge. A shared projection defines a_ic = tanh(W h_i + b)_c. Pair descriptors
contain a_i+a_j, a_i*a_j, and radial Gaussians exp(-r_ij^2/l_c^2), with length
scales l_c logarithmically spaced from 0.5 to 32 Angstrom. A shared MLP predicts
b_ijc and raw v_ijc from those symmetric descriptors. Its output is independent
of q, while the scalar energy itself explicitly depends on q:

    U(R,h,q) = U_old(h,q)
      + (1/C) sum_{i<j,c} s_ij softplus(v_ijc)
          [softplus(b_ijc + a_ic q_i + a_jc q_j) - softplus(b_ijc)].

All chi/curvature values are DERIVATIVES of this U, not separately trained
coefficient heads. Ewald and HVP use identical U architecture/parameter shapes.
The pair network is an energy parametrization, not a separately produced
coupling matrix. Production code never constructs the neural charge Hessian.

The pair normalization is

    e_ij = exp(-r_ij^2 / L^2), d_i = 1 + sum_{j != i} e_ij,
    s_ij = e_ij / sqrt(d_i*d_j), L = 32 Angstrom.

This keeps pair contributions from trivially scaling with the number of nearby
atoms. It has no hard cutoff; disconnected graphs remain coupled. Unlike
normalization by total atom count, infinitely separated subsystems recover
independent normalizations. The Gaussian descriptor/envelope is a geometric
network choice, not an Ewald kernel or a fixed Coulomb law. Max length is a
recorded hyperparameter, not an unreported extension of the MACE cutoff.

Positive pair weights and softplus of affine charge expressions preserve
convexity in q. The inherited curvature floor remains fixed at 20 eV/e^2 in the
initial matched experiments. That floor and convex architecture restrict the
model; they must not be described as an unrestricted functional. Convex neural
U does NOT by itself guarantee positive curvature after adding an Ewald kernel:
Ewald's self term can change definiteness, which must be checked separately.

## The two quadratic responses

At the initial state q0, let d=q-q0, g0=partial_q U(q0), and H0=partial_q^2 U(q0).
Both arms use the same scalar neural energy and initial/proposal networks:

    full_hvp: U(q0)-U(0) + g0.d + 0.5*d.(H0*d)
    diagonal_ewald: U(q0)-U(0) + g0.d
                   + 0.5*sum_i (H0)_ii*d_i^2 + 0.5*q.J_ewald.q.

The surrounding MACE backbone supplies its baseline energy. H0*d is computed
by autodiff HVP. J_ewald is used only by the second arm. The exact diagonal for
the pair term is obtained by the chain rule:

    (H_pair)_ii = (1/C) sum_{j != i,c}
       s_ij softplus(v_ijc) sigmoid(t_ijc) [1-sigmoid(t_ijc)] a_ic^2,
    t_ijc = b_ijc + a_ic q_i + a_jc q_j.

This is exactly the diagonal of the SAME U, not an approximation or predicted
hardness. It avoids one separate Hessian-row backward per atom. Geometry and
parameter graphs remain live through normalization, diagonal and HVP, including
the derivatives required by force training.

## Symmetries and limits

Nonperiodic geometry uses Euclidean squared distances. Periodic geometry uses
the existing attention convention: smooth sine/cosine lattice-phase metric,
weighted by cell-vector squared lengths. It is rotation/translation/permutation
invariant and invariant under translating one atom by a lattice vector. It is
NOT a converged periodic image sum and is not guaranteed invariant under an
arbitrary change of lattice basis or cell replication. Initial fixed-cell Au/MgO
comparisons fit this scope; supercell transfer requires further design/checks.

Pair arithmetic is O(N^2 C); it does not depend on the local MACE neighbor graph.
This is feasible for the proposed ~31-110 atom structures, not claimed linearly
scaling. The new model is backbone independent and needs invariant scalar
features plus positions/cell/pbc, not a MACE-specific embedding contract.

## Verification

`test/test_geometric_neural_charge_state_energy.py`: 8 tests passed on the HHU
runtime (6.70 seconds). Dense Hessians are used only as tiny test references.
Checks cover exact diagonal and mixed parameter/position/state derivatives;
convexity floor; rigid rotation/translation, permutation and periodic shifts;
nonzero remote geometric forces checked by finite differences; and finite
energy+force+residual backward in BOTH Taylor modes without a dense Hessian.

A800 deterministic higher-derivative smoke job: 239105. Artifacts are under
`experiments/acvr_global_geometry_20261009` in the separate ACVR experiment root.
This scalar-only timing excludes the backbone, proposal, Ewald construction,
and data loading and must not be presented as end-to-end training performance.

A800 scalar-only smoke completed all four 31/110-atom, HVP/diagonal cases with
finite parameter gradients and four optimizer steps each. Added parameter count:
6480 (total scalar head 31760 for 256 input channels). At 110 atoms, warm timings
were approximately 0.030 s/step HVP and 0.038 s/step diagonal, excluding MACE,
proposal, physical kernel construction and data loading.
