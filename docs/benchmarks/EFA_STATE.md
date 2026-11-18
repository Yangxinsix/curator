# State-conditioned EFA

The public package is `curator.layer.efa`. It contains the existing Curator
EFA adapter and state-dependent variants. The state variant is an architectural
extension, not a claim to reproduce the paper's complete backbone or accuracy.

```text
curator/layer/efa/
    core.py      EuclideanFastAttention: ERoPE and linear attention contraction
    geometry.py  quadrature/cell selection and positional encoding
    adapter.py   EFAAdapter: unchanged additive geometry-only energy
    blocks.py    EquivariantEFA: equivariant Q/K/V and directional integration
    state.py     charge encoders, residual fusion, StateEFAEnergy
    readout.py   InvariantEnergyReadout: invariant tensor couplings and MLP
```

The old `_efa_adapter` and `_efa_geometry` modules are compatibility imports.
The old adapter's defaults, parameter names, state-dict structure and numerical
operator are preserved. Historical whole-model pickles still resolve their
original module paths. Existing experiments are not silently changed to the
new state energy.

## Charge and direction

Atomic charge q is a continuous even scalar (0e). Geometry features and optional
dipoles supply direction. The gate changes their amplitudes without changing
their transformation rules. Let h be the cached backbone feature tensor and
b = Linear_eq(h [+ p^(1o)]). The charge encoders share the invariant conditioner

    a = MLP_SiLU([invariants(b), q / charge_scale])
    z^(0e) = Linear(a)

For every non-scalar channel, the default `GatedChargeEncoder` uses

    z^(l,p) = 2 sigmoid(Linear_gate(a)) * b^(l,p)

with one scalar gate per complete irrep channel, shared across its magnetic
components. `ScalarChargeEncoder` instead sets z^(l,p) = b^(l,p). Thus its
*initial non-scalar features* are independent of q. Subsequent EFA/fusion layers
can still communicate q into directional features; this is an injection
ablation, not a wholly scalar network. Neither encoder discards backbone
channels by taking a scalar-only feature view. Irrep types absent from the
source cannot be invented by the encoder's linear map; directional tensor
integration can create them later.

## Global features and energy

Each `EquivariantEFA` block computes Q, K and V from z, with separate learned
equivariant linear maps. ERoPE frequencies depend on geometry and fixed model
configuration, not on atom-specific q. The order of contractions is

    S_m = sum_j rotated_K_jm outer V_j
    beta_im = rotated_Q_im^T S_m / sqrt(D_qk)
    g_i = sum_m w_m TP(beta_im, Y(u_m))

`tensor_integration=False` removes the last directional tensor product and
integrates beta directly. This switch is independent of the charge encoder.
All atoms including self participate. No softmax or atom-count normalization is
introduced. At fixed widths and quadrature size this energy branch is linear
in atom count; intermediate storage includes direction-resolved features and
the global K/V contraction, not an N-by-N attention matrix.

Local and global features enter a smooth, gated equivariant residual update.
The final `InvariantEnergyReadout` takes both features, applies two independent
equivariant channel projections, contracts their tensor product into 0e, and
feeds these invariants together with direct scalar features into a SiLU MLP:

    t = z_final [+] g_final
    c = TP_0e(Linear_eq_A(t), Linear_eq_B(t))
    epsilon_i = MLP_SiLU([t^(0e), c])
    U_state = energy_scale * sum_i epsilon_i

The brackets [+] denote concatenation with declared irreps. Cross contractions
preserve relative directional information, such as local/global vector dot
products, that separate squared norms would lose. The final MLP output is an
unrestricted scalar. No fixed quadratic charge model, separate potential head,
or convexity constraint is added. Chemical potentials and curvature are
derivatives of this same U_state. `energy_scale=1.0` is the new model default;
the legacy adapter retains its existing 0.01 default. Both remain configurable.

## ACVR use

For the existing joint charge/dipole pipeline:

```python
from curator.layer.efa import StateEFAEnergy
from curator.layer._multipole_variational_output import MultipoleVariationalOutput
from curator.model.multipole import augment_multipole_model

energy = StateEFAEnergy(encoder="gated", include_dipoles=True)
adapter = MultipoleVariationalOutput(energy_model=energy, proposal=proposal)
model = augment_multipole_model(base_model, adapter)
# Assembly binds all modules before constructing the optimizer.
```

Use the two presets in `curator/configs/model/output_modules/`:

- `multipole_efa_gated.yaml`: default q-gated encoder.
- `multipole_efa_scalar.yaml`: identical energy/proposal settings, scalar injection.

Both presets keep the existing `MultipoleStateProposal`, which uses dense
attention. Consequently the complete model still contains an N² proposal even
though this energy branch is linear. Replacing the proposal is a separate
architectural decision. These presets preserve current charge projection and
residual training. E/F losses differentiate the complete geometry/state path;
stationarity residual replay detaches backbone features and energy parameters
and updates only the proposal. Boundary controls and external work are handled
by the existing ACVR boundary layer, not fed into the internal energy directly.

For a charge-only functional, construct
`StateEFAEnergy(include_dipoles=False)` and bind with `components=1,
state_kind="scalar"`. Its input is `[N,1]`; the joint version takes `[N,4]` in
`[q,px,py,pz]` order. Both implement `bind`, `initial_state`, and
`forward(context, state) -> scalar`. Context supplies flattened `features`,
`positions`, and boundary flags. State units are e and e*Angstrom; the model
scales inputs continuously. The energy uses no charge labels.

## Contracts and validation

State energy parameters are independent of proposal parameters. Bind an equal
feature specification repeatedly without replacing parameters; changing the
specification raises. State checkpoints record feature layout, encoder choice,
energy/state scales, irreps, quadrature and readout configuration, and reject
incompatible contracts. Moving the module to a dtype/device before binding is
supported. Cached backbone features retain their position gradient.

New state energy and directional tensor integration support finite systems.
They explicitly reject PBC. The existing geometry-only adapter's raw-lattice
periodic branch is unchanged, including its documented image-invariance
limitation. Spherical quadrature gives approximate rotational covariance;
the numerical error depends on frequencies, size and quadrature order.

The tests cover an independent tiny dense relative-phase oracle, output and
parameter/input derivatives, O(3)/translation/permutation behavior, q-gating
paths, directional cross terms in the readout, charge Hessian-vector products,
cross-atom response, force finite differences, checkpoint compatibility, and
end-to-end proposal-only residual updates. These checks establish implementation
contracts; predictive accuracy and energy-scale choices require the scientific
training ablations.

```bash
python -m pytest -q test/test_efa_adapter.py test/test_efa_periodic.py \
  test/test_efa_features.py test/test_efa_state.py
```
