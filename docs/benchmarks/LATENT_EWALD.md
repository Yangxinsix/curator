# Curator adapters for official Latent Ewald Summation

Implementation: `curator/layer/_latent_ewald.py`.
These are opt-in output modules. Existing MACE, ACVR, QEq and Ewald classes
are unchanged. This is a trainable Curator integration of the official LES
energy kernels; it is **not** a claim to reproduce a paper's full architecture,
training schedule, checkpoints or published scores.

## Which methods are implemented?

- `LatentEwaldEnergy`: short-range neural energy plus monopole LES.
  Invariant backbone features predict one latent scalar per atom. No charge
  target, QEq solve, ACVR proposal, stationarity loss or total-charge projection
  is applied. The latent scalar is not asserted to equal a DFT atomic charge.
- `PolarizableLatentEwaldEnergy`: the same official backend, with permanent
  polar-vector dipoles and induced dipoles. This is the **rank-1 subset** of
  extended LES; it contains no quadrupoles. The official induced response is
  alpha times the field from permanent sources, with the corresponding energy
  correction; it is not an iterative self-consistent polarizable QEq model.

Both reuse Curator's typed response feature specification. MACE, NequIP and
PaiNN can supply invariant scalar features. Permanent dipoles additionally
require polar-vector (1o) features. There is no dependency on a particular
backbone class.

## External code and reproducibility

The energy operator is `les.Les` from
[the authors' PyTorch repository](https://github.com/ChengUCB/les),
pinned to commit `35b971a650cf30adb84fef7138761e81aa47b3a2`.
Its license is **CC BY-NC 4.0**, including attribution and noncommercial terms.
It is an optional dependency and has not been copied into the Curator package.
The canonical runtime image is unchanged.

HHU official source:
`/data1/home/hzxy05/work/yangxin/acvr/runtime/les-35b971a/src`.
The downloaded official archive, README, pyproject and source are kept alongside
that directory. The upstream package declares PyTorch >= 2.5.0.

The Curator launcher resets PYTHONPATH. Add this directory **inside** its
Python process, or use a task-specific wrapper which appends it inside the
container; do not assume a host PYTHONPATH override survives the launcher.
The root benchmark launcher is responsible for this optional dependency.

References:
- [Original LES paper (2025)](https://www.nature.com/articles/s41524-025-01577-7).
- [LES theory and boundary conventions](https://les.readthedocs.io/en/latest/theory.html).
- [Extended multipolar LES preprint](https://arxiv.org/abs/2605.05746).
- [Published-method training examples](https://github.com/ChengUCB/extended_les_fit).

## What is actually calculated?

For isolated structures (FFF), the official monopole term is

    E_LES = C sum_{i<j} z_i z_j erf(r_ij / (sqrt(2) sigma)) / r_ij,

with C = 90.4756 / (2 pi) eV Angstrom and default sigma = 1 Angstrom.
This includes every pair, independently of the backbone's neighbor cutoff.
It is Gaussian-smoothed Coulomb, **not** the full point-charge qJq interaction.

For periodic structures (TTT), the same official backend uses the reciprocal
lattice sum with Gaussian attenuation exp(-sigma^2 k^2 / 2), the k=0 convention
and optional self-energy subtraction of upstream LES. Default dl=2 Angstrom
controls reciprocal-space resolution; it is independent of the backbone cutoff.
No extra direct-space point-charge term is added.
The default remove_self_interaction=True follows the official robust setting.

The adapter checks explicit pbc. A molecular storage box with FFF is sent as a
zero cell to upstream, preventing a nonzero box from silently selecting periodic
physics. Mixed TTT/FFF structure batches work. Partially periodic axes (TTF etc.)
are rejected. Periodic cells must be finite, right-handed and nonsingular.
The adapter does not override upstream reciprocal grid truncation or add a
separate background correction. Periodic latent charges are not constrained to
the physical total charge; this follows latent LES, and must not be described
as a physical fixed-Q variational model.

The monopole readout is the official Atomwise MLP with hidden widths 32,16,
SiLU, a parallel linear readout, and output scale 0.1. It is materialized during
bind, before constructing the optimizer. No parameter is introduced lazily
during a training forward.

For the rank-1 extension, permanent dipoles are a bias-free channel-linear
readout of the backbone's Cartesian vector features, scaled by output_scale.
The isotropic polarizability is predicted from scalar features with a sigmoid,
bounded between 0 and polarizability_scale (default 0.01 in the backend's
units). This bound is a Curator stability choice, not an assertion that it
matches an authors' fitted configuration. Both the bound and readout widths
are experiment hyperparameters; the bound is not a proof of global energy stability.
All monopole/dipole/induced interaction
energies and fields themselves are evaluated by the official code.

## Composition and training

Insert the module after the physical-unit short-range energy rescaling and
before total-energy force differentiation:

    MACE / NequIP / PaiNN -> physical E_short
                         -> LatentEwaldEnergy -> E_short + E_LES
                         -> position gradient -> forces

Use `.bind(representation.response_feature_spec())` before optimizer creation.
The usual representation callback also binds it; repeated binding to the same
specification is idempotent, and a different specification raises an error.

Total forces must differentiate positions, including both the direct LES
geometry dependence and the charge/dipole dependence through backbone
features. An edge-only force head misses the direct all-pairs/reciprocal term.
The adapter intentionally does not compute forces itself.

Outputs include `les_energy`, `les_latent_charges`, and for the rank-1 adapter,
`les_permanent_dipoles`, `les_polarizability`, `les_total_dipoles`.
Only the total energy and total forces should receive data supervision for
the user's EF-only benchmark. These models do not have an ACVR stationarity
residual to train or compare.

The current adapter exposes a single energy channel and does not claim
TorchScript export, compiled performance, stress validation, quadrupoles,
or physical charge conservation. Those require separate validation.

## Verification

`test/test_latent_ewald.py` exercises the real optional official backend.
As of 2026-10-10, ten tests passed on the canonical runtime CPU:
FFF/TTT energy-force finite differences, energy-plus-force parameter backward
for both adapters, explicit isolated smoothed-kernel equality, artificial-box
dispatch, mixed structure batching, rejection of mixed axes, symmetry,
parameter registration, strict checkpoint state round-trip, and geometry-dependent
descriptor force derivatives/backpropagation to mock backbone parameters.

Audit:
`/data1/home/hzxy05/work/yangxin/acvr/experiments/longrange_baselines_20261010/les_tests.log`.

This validates operator integration and differentiability, not benchmark
accuracy. No new formal LES training has been launched by this adapter task.

