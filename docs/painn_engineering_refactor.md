# Reusable PaiNN Engineering Components

## Goal

Bring the useful numerical and graph engineering used by production PaiNN
implementations into Curator without adding a second FairChem-specific backbone.
The components are reusable by other scalar/vector models, while the existing
`Painn` class remains responsible only for composing them at valid points in its
computation graph.

Hidden-feature normalization is intentionally separate from physical-output
rescaling. None of the components in this document reads energy/force/Hessian
output metadata or interacts with `GlobalRescaleShift`.

## Status

| Task | Status | Result |
|---|---|---|
| Write the implementation plan | Complete | This document records the design, implementation, and verification status. |
| Add `ScaledSiLU` | Complete | Reusable fixed-scale SiLU activation. |
| Add `safe_norm` | Complete | Stable first- and second-order derivatives at zero. |
| Add `ResidualAdd` | Complete | Explicit residual policy with an optional variance-preserving scale. |
| Add `VarianceScale` | Complete | Non-trainable, observable hidden-feature scaling with persistent fitted state. |
| Add a generic variance fitter | Complete | Discovers and fits scales in actual forward execution order without checking model type. |
| Add `reset_linear` | Complete | Reusable Xavier-uniform weight and zero-bias initializer. |
| Add `GatedEquivariantBlock` | Complete | Public scalar/vector block reused by the architecture-independent direct-force head. |
| Add bidirectional-edge handling | Complete | Deduplicates edges, supplies missing reverse edges, and preserves edge-aligned fields per structure. |
| Share PaiNN radial encoding | Complete | The radial basis and cutoff are evaluated once per forward pass. |
| Make PaiNN blocks return deltas | Complete | The representation loop owns residual composition. Standalone blocks retain their residual-compatible default API. |
| Compose the components in PaiNN | Complete | Components are constructor-injected; there is no `fairchem_mode` or second backbone. |
| Add focused unit tests | Complete | Covers formulas, derivatives, state dicts, fitting, equivariance, edge fields, and PaiNN integration. |
| Run relevant regression tests | Complete | 70 direct-force, Hessian, dynamic-PHL, normalization, graph, and integration tests pass. |

## Implemented Components

### Numerical primitives

The reusable primitives live in `curator.layer`:

- `ScaledSiLU`:

  \[
  y = c\,\operatorname{SiLU}(x), \qquad c_{\mathrm{default}} = \frac{1}{0.6}.
  \]

- `safe_norm`:

  \[
  \lVert x\rVert_\epsilon =
  \sqrt{\sum_i x_i^2 + \epsilon}.
  \]

  A positive \(\epsilon\) keeps both first- and second-order derivatives finite
  at a zero vector, which matters for force and Hessian training.

- `ResidualAdd`:

  \[
  y = c(x+\Delta x).
  \]

  Ordinary residual addition uses \(c=1\). A common variance-preserving choice is
  \(c=1/\sqrt{2}\). Its fixed scale is constructor configuration and therefore
  does not add a new persistent key to legacy checkpoints.

- `VarianceScale`:

  \[
  y = sx,\qquad
  s =
  \begin{cases}
  1/\sqrt{\operatorname{Var}(x)}, & \text{unit target variance},\\
  \sqrt{\operatorname{Var}(x_{\mathrm{ref}})/
  \operatorname{Var}(x)}, & \text{reference variance}.
  \end{cases}
  \]

  Observation uses streaming sums and squared sums. The fitted `scale` and
  `fitted` flag are ordinary buffers and round-trip through `state_dict`.

- `reset_linear` applies Xavier-uniform initialization to a linear-like
  module's weights and zeros its bias when present.

### Model-independent variance fitting

`fit_variance_scales(model, batches, ...)`:

1. finds all `VarianceScale` instances through `named_modules()`;
2. records actual execution order with one dry forward pass;
3. fits the modules sequentially, so every earlier fitted scale is active while
   a later scale is observed;
4. restores the model's original train/eval state;
5. reports unexecuted or unfitted modules explicitly.

The utility accepts a caller-provided `forward(batch)` adapter for device transfer
or nonstandard input signatures. It contains no PaiNN, MACE, trainer, or output
rescale branch.

Example:

```python
from curator.train import fit_variance_scales

scales = fit_variance_scales(
    model,
    calibration_batches,
    num_batches=16,
    forward=lambda batch: model(move_to_device(batch)),
)
```

### Gated scalar/vector block

`GatedEquivariantBlock` consumes invariant scalar channels \(s\) and equivariant
vector channels \(v\). With two independent channel projections,

\[
v_{\mathrm{inv}} = W_{\mathrm{inv}}v,\qquad
v_{\mathrm{out}} = W_{\mathrm{out}}v,
\]

it constructs rotation-invariant inputs

\[
h = [s,\lVert v_{\mathrm{inv}}\rVert_\epsilon]
\]

and predicts new scalars and scalar vector gates,

\[
(s',g) = \operatorname{MLP}(h),\qquad
v' = g\odot v_{\mathrm{out}}.
\]

Only scalar gates multiply vectors, so the vector output remains equivariant.
The block supports injected activations, scalar-output activations, dimensions,
and linear initialization. `ScalarVectorForceHead` uses this public block rather
than a PaiNN-private implementation.

### Bidirectional edges

`EnsureBidirectionalEdges` is a data transform, not model-forward logic. For
every directed periodic edge

\[
(i,j,\Delta r,T)
\]

it ensures the presence of

\[
(j,i,-\Delta r,-T).
\]

The transform:

- deduplicates exact directed edge/image duplicates;
- retains the payload of an already-present reverse edge;
- copies all other registered edge fields to a generated reverse edge;
- negates displacement-like fields;
- processes each `n_pairs` segment independently;
- preserves empty batches and updates per-structure edge counts.

## PaiNN Integration

Curator still has one native `Painn` backbone. The integration changes its
internal ownership boundaries:

1. `RadialBasisEdgeEncoding` computes the basis and cutoff once per forward pass.
2. `PainnMessage` returns message deltas when composed by the representation.
3. The representation applies the selected message residual rule.
4. `PainnUpdate` returns intra-atomic update deltas.
5. The representation adds the update and applies an optional per-layer
   `VarianceScale`.

In compact form, interaction layer \(l\) is:

\[
(\Delta s_m,\Delta v_m)=M_l(s_l,v_l,e),
\]

\[
s_m=c_{\mathrm{res}}(s_l+\Delta s_m),\qquad
v_m=v_l+\Delta v_m,
\]

\[
(\Delta s_u,\Delta v_u)=U_l(s_m,v_m),
\]

\[
s_{l+1}=S_l(s_m+\Delta s_u),\qquad
v_{l+1}=v_m+\Delta v_u.
\]

The message and update layers also accept optional engineering scales for state
vectors, vector messages, invariant inner products, and scalar updates. These are
plain constructor parameters rather than a named architecture mode.

An enhanced composition can be assembled explicitly:

```python
import math
from torch import nn

from curator.layer import ScaledSiLU, reset_linear
from curator.model import Painn

representation = Painn(
    num_interactions=3,
    num_features=128,
    cutoff=5.0,
    activation=ScaledSiLU,
    scalar_norm=nn.LayerNorm,
    message_residual_scale=1 / math.sqrt(2),
    state_vector_scale=1 / math.sqrt(3),
    message_vector_scale=1 / math.sqrt(128),
    inner_product_scale=1 / math.sqrt(128),
    scalar_update_scale=1 / math.sqrt(2),
    norm_eps=1e-8,
    vector_bias=False,
    layer_scales=[None, None, None],
    linear_initializer=reset_linear,
)
```

`None` inside an explicit `layer_scales` sequence creates an unfitted
`VarianceScale`; a number creates a fixed fitted scale; a module is copied and
used directly. Omitting `layer_scales` keeps identity scaling.

The ordinary Curator choices remain individually available: regular `SiLU`,
identity scalar normalization, unscaled residuals, and identity layer scaling.
Two correctness fixes are now the PaiNN defaults:

- `safe_norm(..., eps=1e-8)` prevents undefined Hessians at zero vector norm;
- vector-channel projections have no bias, because a learned Cartesian bias is
  not rotation equivariant.

The existing checkpoint upgrader removes the two legacy vector biases when it
loads an older serialized PaiNN model.

For exact component-level comparisons, PaiNN additionally exposes generic
`num_elements`, `atomic_number_offset`, and `update_scalar_first` parameters.
The direct-force output accepts two `block_kwargs` mappings, so invariant widths,
hidden widths, activations, scalar activations, and norm epsilon can be selected
without introducing an architecture mode.

## FairChem Paper Parity

The paper configuration was reconstructed with:

- 4 interactions, 128 channels, and 128 Gaussian radial functions;
- a 12 Å fifth-order polynomial cutoff;
- 83 embeddings addressed by `Z - 1`;
- scalar-first update inputs;
- the paper repository's four fitted scale weights;
- the FairChem energy and gated direct-force readout dimensions.

Using the paper-pinned FairChem commit
`13014274282ded9c3cbbba3fb156ad5f11fb3edd`, identical network weights, and an
identical directed graph, the implementations have the same 1,029,635 trainable
parameters. On a 12-atom SPICE monomer, energy matched exactly and direct forces
had a maximum absolute difference of `5.59e-8`; all intermediate comparisons
passed an absolute tolerance of `2e-5`.

The parity configuration, script, JSON output, and detailed report live under
`experiments/energy_hessian_paper_reproduction/{configs/parity,painn_parity}`.

## Files

| Area | Files |
|---|---|
| Numerical components | `curator/layer/activation.py`, `norm.py`, `residual.py`, `variance_scale.py`, `initialization.py` |
| Equivariant head component | `curator/layer/gated_equivariant.py`, `curator/layer/_force_output.py` |
| Graph transform | `curator/data/_transform.py` |
| PaiNN composition | `curator/layer/_painn_message.py`, `_painn_update.py`, `curator/model/painn.py` |
| Fitting utility | `curator/train/variance_scaling.py` |
| Focused tests | `test/test_numerical_components.py`, `test/test_variance_scaling.py`, `test/test_bidirectional_edges.py`, `test/test_painn_engineering.py`, direct-force tests |

## Verification

The final regression command covered the new components plus the affected
direct-force, Hessian, dynamic projected-Hessian, distillation normalization, and
neighbor-list behavior:

```text
70 passed, 15 warnings in 10.29s
```

The warnings are pre-existing optional native-neighbor-list and TorchScript type
annotation warnings; there were no test failures.

## Non-goals

- No `FairChemPainn` class and no `fairchem_mode`.
- No second model-specific training path.
- No PaiNN-only variance-fitting callback.
- No hidden-feature state inside physical output rescale layers.
- No automatic choice of experimental scale factors.
- No change to experiment YAML files as part of this component refactor.
