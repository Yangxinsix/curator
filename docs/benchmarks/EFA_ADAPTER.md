# Curator EFA adapter

The implementation now lives in `curator/layer/efa/`. Historical private module
imports and checkpoints remain supported. This document describes the unchanged
geometry-only additive adapter; see [state-conditioned EFA](EFA_STATE.md) for
charge gating, equivariant tensor integration and ACVR integration.

This is a trainable PyTorch adapter after an existing Curator backbone. The
backbone remains trainable. It adds a global energy term before force
differentiation. It does **not** load the authors' JAX checkpoint, reproduce
their complete message-passing backbone, or claim their published accuracy.

## What is implemented

`curator.layer.efa.EFAAdapter` consumes the backbone's
`response_feature_spec()`, so both `IrrepsFeatureSpec` (MACE/NequIP-style)
and `ScalarVectorFeatureSpec` (scalar/vector representations) work without
modifying those backbones. This is an interface compatibility claim: it does
not mean MACE, NequIP and PaiNN have all completed integration tests. The
current real pretrained-backbone integration audit uses MACE; unit tests cover
both feature specification formats. It binds through Curator's existing
representation callback or explicit `adapter.bind(spec)`; create the optimizer after binding.

For input atom features h_i, equivariant linear maps produce Q_i and K_i.
By default every declared input irrep contributes; `use_equivariant_queries=False`
restricts Q/K to 0e. V_i is a learned invariant linear map. For each integration
direction u_m, pairs of feature channels are rotated by angle
omega_c (r_i dot u_m). Denote those rotated features Q_im and K_im:

    S_m = sum_j K_jm outer V_j
    beta_i = sum_m w_m Q_im^T S_m / sqrt(D_qk)
    Delta E = energy_scale * sum_i MLP(beta_i)
    E_total = E_backbone + Delta E
    F = -d E_total / d R

This is Euclidean rotary position encoding (ERoPE), followed by the genuine
linear-attention contraction. It is **not** ordinary softmax attention.
Self terms are included, as in the official implementation; there is no
division by atom count and no q-label supervision or variational residual.

S_m is computed first, so no N-by-N attention matrix or pair neighbor list is
constructed. Cost at fixed angular content/width/grid is linear in atom count:
O(N M D_qk D_v), with O(M D_qk D_v) global intermediate storage per structure.
Each graph is evaluated separately so batched structures cannot communicate.

This is the EFA specialization with no tensor-product integration,
equivariant Q/K and invariant V. Angular Q/K components are contracted to a
scalar attention coefficient. Thus it retains angular backbone information
while producing an invariant energy. It is not the full tensor-integration
EFA architecture; do not label results simply as reproduction of paper EFA.

## Integration

Use the common benchmark assembler rather than editing a representation or
the existing ACVR energy. Direct construction:

```python
from curator.layer.efa import EFAAdapter

adapter = EFAAdapter(
    feature_spec=backbone.response_feature_spec(),
    hidden_features=32,       # must be even (ERoPE channel pairs)
    num_directions=110,       # FFF Lebedev grid: 50, 110 or 194; TTT uses 3 lattice rows
    max_frequency=6.283185307179586,
    max_length=32.0,          # frequency scale; FFF intended maximum diameter in Angstrom
    energy_scale=0.01,        # fixed physical eV multiplier of learned readout
    use_equivariant_queries=True,
)
```

The separate preset `curator/configs/model/output_modules/efa_adapter_periodic.yaml`
selects the official AuMgO positional frequencies while retaining this adapter's
32-channel width and readout:

```yaml
num_directions: null
max_frequency: 0.7853981633974483  # pi/4
max_length: 1.0
```

`num_directions=None` omits the unused Lebedev grid for a periodic-only model;
FFF input then raises rather than choosing a fallback quadrature. Keep a supported
Lebedev grid configured when the same model must accept both FFF and TTT graphs.
This preset aligns the positional operator, not the authors' complete neural
architecture or training hyperparameters.

Insert the adapter after any energy rescale/offset to physical units and before
force differentiation. **Forces must differentiate total energy with respect
to positions**; a derivative only with respect to short-range edge vectors
misses the EFA term's direct global positional dependence. The shared benchmark
assembler handles this integration. EF training differentiates through the
backbone, Q/K/V maps, ERoPE and readout entirely in PyTorch, including the
mixed derivatives needed to train forces. No JAX runtime is required.

The output contains total `energy` and the additive `efa_energy` diagnostic.
Atomic charge labels, boundary total charge and proposal networks are not
used by this baseline. Consequently it cannot distinguish identical geometry
with different electronic charge/spin; charged-only datasets can be fit but
mixed charge states require a separately designed conditioning interface.

## Model replacement and compatibility

The common assembler deep-copies the baseline and binds a copied adapter before
optimizer construction. Curator's subsequent callback binds it again; binding
an equal feature specification is idempotent and preserves trained parameters.

`clone_with_representation()` copies output modules and runs their callbacks.
For EFA, a changed feature key, scalar/vector layout or irreps specification
raises an error rather than silently keeping incompatible matrices. A new
adapter is required when intentionally changing those interfaces. Equal
specifications allow weight reuse, but do not prove two different backbones'
embedding channels have the same semantics. For cross-backbone comparisons,
construct a fresh adapter instead of transferring trained adapter weights.

This canonical revision has no `curator.utils.update_model` function; that
name in older workflow notes is stale. The current relevant mechanisms are
`clone_with_representation`, model conversion and checkpoint loading. This
adapter does not promise automatic adaptation to arbitrary checkpoint/model
architecture changes, multi-domain conversion, TorchScript deployment or stress.

## Boundary and quadrature limitations

Explicit boundary flags select the integration directions independently for
each structure. Isolated FFF structures retain the Lebedev grid. Fully periodic
TTT structures use the **three raw, row-wise lattice vectors**, with weights
1/3, as in the authors' periodic branch. Partial periodicity (such as TTF) is
not implemented and raises `NotImplementedError`; missing boundary metadata or
invalid periodic cells raise rather than silently selecting an isolated model.
An FFF/TTT mixed batch is valid when a Lebedev grid is configured and each
periodic structure has its own cell. Cell tensors use Curator's flattened
`(3 * B, 3)` layout or explicit `(B, 3, 3)` layout, with lattice vectors in rows
and the graph index matching `n_atoms`/`image_idx`.

The periodic route changes only directions and weights in ERoPE. It does not
normalize lattice vectors, construct reciprocal vectors, wrap atoms, enumerate
periodic images, apply a minimum-image replacement, or add an Ewald energy.
It uses the same fixed frequency construction as the isolated route. With
row-wise lattice vectors a_m its phase is

    theta_imc = omega_c * dot(r_i, a_m),  m = 1, 2, 3.

The cell is part of the input geometry, not a fixed laboratory coordinate frame.
Rotating atom coordinates **and the cell** together therefore preserves the
periodic scalar attention algebraically. The position-centering translation
gauge used in the PyTorch core cancels in query/key inner products; it does not
wrap independent atoms or change the authors' relative-position kernel.
Cell and position tensors remain in the autograd graph. Forces are coordinate
derivatives at fixed cell; stress is outside the benchmark assembler contract.

This follows the official lattice-direction model, not a guarantee of strict
invariance under choosing a different periodic image for an individual atom.
Changing r_i to r_i + a_j adds omega_c * dot(a_j, a_m) to its phase, which need
not be a multiple of 2*pi. Training and evaluation must therefore preserve a
consistent cell basis and coordinate-image convention. Independent atom
wrapping, arbitrary cell-basis changes, and unit-cell replication need separate
physical validation; they are not claimed invariances of this implementation.
The limitation is recorded explicitly in the numerical checks rather than
hidden by wrapping coordinates. A commensurate reciprocal-frequency extension
would be a separately named model change, not this official-branch port.

For periodic runs `max_length` remains a divisor in the frequency formula;
its isolated-molecule interpretation as a maximum diameter does not transfer to
raw lattice projections. The authors' AuMgO configuration uses max_frequency =
pi/4 and max_length = 1.0. These are model hyperparameters, not an Ewald cutoff
or a guarantee of periodic image invariance. `num_directions` only selects the
fallback FFF grid, or disables it when set to None; a TTT evaluation always
contracts three directions.

For FFF, the finite Lebedev rule approximates continuous rotational integration.
Translation and permutation invariance are algebraic; rotation invariance and
force covariance have quadrature error. Supported maximum frequencies obey the
official rule-of-thumb limits (50: pi, 110: 2.5 pi, 194: 4 pi). The limit assumes
interatomic distances no larger than `max_length`. Choose this length from
geometry coverage, and validate held-out scan diameters and rotated energies/
forces when changing frequencies. Frequencies are fixed, not learned, to avoid
training silently invalidating quadrature accuracy. The adapter does not apply
a distance cutoff; `max_length` sets frequencies, not interaction range.

EFA is a learned global representation rather than an electrostatic solver.
It does not enforce Coulomb asymptotics, charge conservation or additivity of
infinitely separated fragments. Those properties should be evaluated in the
long-range scans, not assumed from the module name.

## Verification

`test/test_efa_adapter.py` and `test/test_efa_periodic.py` check:

- Lebedev normalization and isotropic second moment for all three grids;
- linear contraction against a dense, tiny-system ERoPE oracle;
- an independent exact spherical sinc integral and its position derivative
  near the configured maximum molecular diameter;
- noninteracting batches, atom permutation and translation invariance;
- energy invariance and force covariance under a general rotation;
- force finite difference and finite nonzero EF-loss gradients into Q/K/V,
  readout and input backbone features;
- idempotent binding, scalar/vector features, serialization and late dtype
  conversion;
- periodic lattice contractions against an independent dense reference,
  total coordinate force derivatives and mixed energy/force-loss gradients;
- per-structure FFF/TTT routing, joint cell/position rotations and graph isolation;
- explicit refusal of partial periodicity, missing boundary metadata and invalid
  periodic cells;
- a diagnostic documenting the official lattice kernel's lack of general
  independent atom-image invariance, without claiming that limitation is fixed.

Real pretrained-MACE integration and energy/force optimizer smoke checks are
performed by the benchmark assembler's tests/audit. Passing these is functional
validation, not a claim of improved benchmark accuracy.

## Provenance

Algorithm: J. Thorben Frank, Stefan Chmiela, Klaus-Robert Mueller and Oliver T.
Unke, *Machine Learning Global Atomic Representations with Euclidean Fast
Attention*, Nature Machine Intelligence (2026);
https://doi.org/10.1038/s42256-026-01195-y and
https://arxiv.org/abs/2412.08541.

Reference implementation:
https://github.com/thorben-frank/euclidean_fast_attention , pinned revision
`a7f28cabe56464b92bb0d9bd82793b8d44deb277` for the periodic extension.
The periodic `pbc_bool` branch in `euclidean_fast_attention/fast_attention.py`
selects raw lattice rows and 1/3 weights; `rope.py` contracts row-wise projections.
The separately defined `frequency_init_fn_pbc` is **not** used by that default
branch. The reference AuMgO settings are in
`configs/model/4ghdnnp_base_model_pbc.py`; the model design is described in
[paper Appendix C](https://arxiv.org/html/2412.08541v2#A3).
Its `pyproject.toml` declares MIT and its backend is JAX/Flax/e3x.
This independently written PyTorch implementation follows the ERoPE equations,
query scaling, channel pairing, frequency spacing and key/value-first
contraction. It does not copy the authors' training code.

Numerical standard Lebedev orbit constants were cross-checked against e3x
(Apache-2.0), https://github.com/google-research/e3x ,
`e3x/so3/_lebedev_grids.npz`, Git blob
`eb7a3ed138f89cbfe5664df89b72e40092d0ab37`.
The table is expanded from symmetry orbits at initialization, and stored with
each model checkpoint. No runtime external download is needed.

