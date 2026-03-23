# CURATOR LAMMPS Interface

This directory contains the CURATOR-owned files that are patched into LAMMPS:

- `pair_curator.cpp/.h`: `pair_style curator`
- `compute_uncertainty.cpp/.h`: `compute uncertainty <key>`
- `compute_uncertainty_atom.cpp/.h`: `compute uncertainty/atom <key>`
- `ML-IAP/mliap_data.h/.cpp`: ML-IAP data plumbing needed by the CURATOR `mliap` bridge
- `ML-IAP/mliap_unified_couple.pyx`: Python bridge for `pair_style mliap unified`

The intended design is:

- keep CURATOR-specific LAMMPS code here
- prefer patching these files instead of editing official LAMMPS source files
- keep the `mliap` uncertainty path as a CURATOR-side interface change, not a `pair_mliap_kokkos.cpp` fork

## Deploy Uncertainty Maintenance Rule

Deploy-time uncertainty must stay centralized on the CURATOR Python side.

- use `curator.simulate.uncertainty._deploy` as the single deploy uncertainty entrypoint
- keep method semantics inside the method modules under `curator.simulate.uncertainty`
- do not add one-off deploy-only files such as `latent_*` or method-specific uncertainty shims
- prefer generic output-key plumbing in LAMMPS-facing code; LAMMPS should consume keys, not methods

This is a long-term maintainability rule for the curator pipeline: new uncertainty methods
should extend the registry/builder in `curator.simulate.uncertainty` instead of scattering
deploy logic across unrelated modules.

The deploy config contract is:

```yaml
deploy:
  uncertainty:
    method: none | ensemble | mahalanobis
    dataset: null
    output_keys: null
    maha:
      kernel: local-full-g
      max_structures: null
      regularization: 1e-6
      streaming: false
```

Additional maintenance constraints:

- `method: mahalanobis` must reuse the existing `FeatureCalculator` semantics from `curator.layer._feature`
- covariance / precision fitting belongs to deploy preparation, not to LAMMPS C++
- do not add a deploy-only fallback such as `_latent_mahalanobis.py` just to satisfy one runtime
- if TorchScript support for hook-based Mahalanobis needs more work, fix the shared feature path itself instead of adding a second Mahalanobis implementation

## Variables Used Below

Replace these placeholders with paths on your own machine:

- `${CURATOR_ROOT}`: path to this CURATOR repo
- `${SPACK_ROOT}`: path to your Spack installation
- `${SPACK_REPO}`: path to your local Spack repo that contains `packages/lammps/package.py`
- `${LAMMPS_SRC}`: path to an unpacked LAMMPS source tree
- `${LAMMPS_TARBALL}`: cached or downloaded LAMMPS source tarball
- `${PYTHON_EXE}`: Python executable that already has `torch`, `numpy`, and `cython`

If you do not already have a Python environment, create one first and install:

- `torch`
- `numpy`
- `cython`

## Recommended Path: Install With Spack

This is the cleanest path if you want a reproducible install.

### 1. Add your local Spack repo

```bash
source "${SPACK_ROOT}/share/spack/setup-env.sh"
spack repo add "${SPACK_REPO}"
```

Your local repo should contain a LAMMPS package file here:

```text
${SPACK_REPO}/packages/lammps/package.py
```

### 2. Create a dedicated Spack environment

Use a dedicated environment instead of global Spack config.

```bash
PY_VER=$("${PYTHON_EXE}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
ENV_DIR="${PWD}/spack-env-lammps-curator"
mkdir -p "${ENV_DIR}"

cat > "${ENV_DIR}/spack.yaml" <<EOF
spack:
  repos:
  - ${SPACK_REPO}
  packages:
    python:
      externals:
      - spec: python@${PY_VER}
        prefix: $(dirname "$(dirname "${PYTHON_EXE}")")
      buildable: false
    kokkos:
      version: [4.6.02]
  specs:
  - lammps +kokkos +python +plumed +ml-iap ^kokkos@4.6.02 ^python@${PY_VER}
EOF
```

### 3. Rebuild the CURATOR patch and update the sha256

Use the user-facing helper in this directory:

```bash
bash "${CURATOR_ROOT}/interface/spack/rebuild_curator_pytorch_patch.sh" \
  --lammps-tarball "${LAMMPS_TARBALL}" \
  --spack-package-dir "${SPACK_REPO}/packages/lammps" \
  --python-exe "${PYTHON_EXE}"
```

This script:

- rebuilds `curator_pytorch.patch`
- dry-runs the patch against the LAMMPS tarball
- updates the `sha256` in `package.py` when it finds the expected patch entry

### 4. Concretize and install

```bash
spack -e "${ENV_DIR}" concretize -f
spack -e "${ENV_DIR}" install
```

### 5. Validate

```bash
spack -e "${ENV_DIR}" load lammps
lmp -h | rg -i "curator|ml-iap|python|plumed|kokkos"
```

You should see the packages you enabled.

## Fast Rebuild After Editing Interface Files

If you edit files under `interface/` and only want a new LAMMPS binary:

1. rebuild the patch
2. keep the same dependency stack
3. rebuild only the root `lammps` package

The patch rebuild step stays the same:

```bash
bash "${CURATOR_ROOT}/interface/spack/rebuild_curator_pytorch_patch.sh" \
  --lammps-tarball "${LAMMPS_TARBALL}" \
  --spack-package-dir "${SPACK_REPO}/packages/lammps" \
  --python-exe "${PYTHON_EXE}"
```

Then run your usual root-only Spack rebuild.

## What Actually Gets Patched Into LAMMPS

The generated patch injects these files:

- `src/pair_curator.cpp`
- `src/pair_curator.h`
- `src/compute_uncertainty.cpp`
- `src/compute_uncertainty.h`
- `src/compute_uncertainty_atom.cpp`
- `src/compute_uncertainty_atom.h`
- `src/ML-IAP/mliap_data.cpp`
- `src/ML-IAP/mliap_data.h`
- `src/ML-IAP/mliap_unified_couple.pyx`

The canonical CURATOR source for the ML-IAP bridge is:

- `interface/ML-IAP/mliap_data.cpp`
- `interface/ML-IAP/mliap_data.h`
- `interface/ML-IAP/mliap_unified_couple.pyx`

These files are intentionally stored under `interface/` so users can find and edit all CURATOR-owned LAMMPS-facing code in one place.

## Install Without Spack

If you are not using Spack, do this manually.

### 1. Prerequisites

You need:

- a LAMMPS source tree
- a Python with `torch`, `numpy`, `cython`
- a C++ compiler
- optionally MPI, Kokkos, PLUMED

### 2. Use a compatible LAMMPS release

This repo is currently patched against:

- `22 Jul 2025 - Update 3`

If you use a different LAMMPS release, expect patch conflicts.

### 3. Apply the patch

Preferred route:

```bash
patch -p1 -d "${LAMMPS_SRC}" < "${SPACK_REPO}/packages/lammps/curator_pytorch.patch"
```

If you are not maintaining a Spack repo, you can still generate the patch from this directory using the helper under `interface/spack/`.

### 4. Manual copy route if you do not want to apply the full patch

Copy these files:

```bash
cp "${CURATOR_ROOT}/interface/pair_curator.cpp" "${LAMMPS_SRC}/src/"
cp "${CURATOR_ROOT}/interface/pair_curator.h" "${LAMMPS_SRC}/src/"
cp "${CURATOR_ROOT}/interface/compute_uncertainty.cpp" "${LAMMPS_SRC}/src/"
cp "${CURATOR_ROOT}/interface/compute_uncertainty.h" "${LAMMPS_SRC}/src/"
cp "${CURATOR_ROOT}/interface/compute_uncertainty_atom.cpp" "${LAMMPS_SRC}/src/"
cp "${CURATOR_ROOT}/interface/compute_uncertainty_atom.h" "${LAMMPS_SRC}/src/"
mkdir -p "${LAMMPS_SRC}/src/ML-IAP"
cp "${CURATOR_ROOT}/interface/ML-IAP/mliap_data.cpp" "${LAMMPS_SRC}/src/ML-IAP/"
cp "${CURATOR_ROOT}/interface/ML-IAP/mliap_data.h" "${LAMMPS_SRC}/src/ML-IAP/"
cp "${CURATOR_ROOT}/interface/ML-IAP/mliap_unified_couple.pyx" "${LAMMPS_SRC}/src/ML-IAP/"
```

If you choose this route, you must also carry over the CMake changes that:

- add `PKG_CURATOR`
- compile the CURATOR source files
- link Torch

The generated patch remains the authoritative reference.

### 5. Configure and build manually

Example:

```bash
cmake -S "${LAMMPS_SRC}/cmake" -B build \
  -C "${LAMMPS_SRC}/cmake/presets/basic.cmake" \
  -D BUILD_MPI=on \
  -D BUILD_OMP=on \
  -D PKG_PYTHON=on \
  -D PKG_ML-IAP=on \
  -D PKG_KOKKOS=on \
  -D PKG_PLUMED=on \
  -D PKG_CURATOR=on \
  -D CMAKE_PREFIX_PATH="$("${PYTHON_EXE}" - <<'PY'\nimport torch\nprint(torch.utils.cmake_prefix_path)\nPY)" \
  -D Python_EXECUTABLE="${PYTHON_EXE}"

cmake --build build -j
cmake --install build
```

Adjust the package flags to your actual needs.

### 6. Make Python imports visible at runtime

LAMMPS must be able to import:

- its own Python package
- CURATOR

Example:

```bash
export PYTHONPATH="/path/to/lammps-install/lib/pythonX.Y/site-packages:${CURATOR_ROOT}:${PYTHONPATH}"
```

Replace `pythonX.Y` with your actual Python version.

### 7. Validate

```bash
/path/to/lammps-install/bin/lmp -h | rg -i "curator|ml-iap|python|plumed|kokkos"
```

Then run a minimal input using:

- `pair_style curator`
- or `pair_style mliap unified ...`

## Runtime Usage In LAMMPS

The two CURATOR-backed LAMMPS paths are:

- `pair_style curator`: loads a TorchScript model saved by normal CURATOR deploy
- `pair_style mliap unified`: loads the Python-backed `LAMMPS_MLIAP` object saved by `--mliap` deploy

The uncertainty rule is simple:

- LAMMPS can only read uncertainty keys that are already present in the exported model
- scalar uncertainty is read with `compute uncertainty <key>`
- per-atom uncertainty is read with `compute uncertainty/atom <key>`

### 1. Export Models

`pair_style curator` uses a normal TorchScript export:

```bash
python "${CURATOR_ROOT}/curator/deploy.py" \
  "${CKPT_OR_CKPTS}" \
  --target_path compiled_model.pt
```

`pair_style mliap unified` uses the LAMMPS-specific export:

```bash
python "${CURATOR_ROOT}/curator/deploy.py" \
  "${CKPT_OR_CKPTS}" \
  --target_path mliap_model.pt \
  --mliap \
  --element-types Fe Li O P
```

Convenient uncertainty presets:

```bash
# ensemble deploy without a config file
python "${CURATOR_ROOT}/curator/deploy.py" \
  ckpt1.ckpt ckpt2.ckpt ckpt3.ckpt \
  --uncertainty ensemble \
  --target_path compiled_ensemble.pt

# Mahalanobis deploy for pair_style curator
python "${CURATOR_ROOT}/curator/deploy.py" \
  model.ckpt \
  --uncertainty mahalanobis \
  --dataset reference.traj \
  --target_path compiled_maha.pt

# Mahalanobis deploy for mliap
python "${CURATOR_ROOT}/curator/deploy.py" \
  model.ckpt \
  --mliap \
  --element-types Fe Li O P \
  --uncertainty mahalanobis \
  --dataset reference.traj \
  --target_path mliap_model.pt
```

Notes:

- pass multiple checkpoints if you want an `EnsembleModel`
- `--uncertainty ensemble` and `--uncertainty mahalanobis` are the only convenience presets exposed on the CLI
- `deploy.uncertainty` in the config controls whether the exported model carries uncertainty outputs
- CLI only exposes `method` and `dataset`; advanced settings such as `output_keys`, `maha.kernel`, `max_structures`, `regularization`, and `streaming` belong in `cfg_path`
- `pair_style curator` Mahalanobis is TorchScript-safe only for `kernel: gnn` or `kernel: local-gnn`
- `pair_style mliap unified` can also use hook-based Mahalanobis kernels such as `full-g` and `local-full-g`
- `max_structures: null` means use the full reference dataset; set an integer only if you explicitly want to cap fitting cost
- ensemble deploy normally does not need a config file; use `cfg_path` only if you want to customize exported uncertainty keys or other advanced deploy settings

Advanced Mahalanobis tuning stays in `cfg_path`, for example:

```yaml
deploy:
  uncertainty:
    method: mahalanobis
    dataset: reference.traj
    output_keys:
      - maha_dist
      - maha_dist_per_atom
    maha:
      kernel: local-full-g
      max_structures: null
      regularization: 1e-6
      streaming: false
```

### 2. Use `pair_style curator`

`pair_style curator` requires `newton off`, and `pair_coeff` expects atomic numbers in LAMMPS type order.

Minimal example:

```lammps
units metal
atom_style atomic
atom_modify map yes
boundary p p p
newton off
read_data system.data

mass 1 55.845
mass 2 6.94
mass 3 15.999
mass 4 30.973761998

pair_style curator
pair_coeff * * compiled_model.pt 26 3 8 15

neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

thermo_style custom step pe
thermo 1
run 0
```

If you want uncertainty from `pair_style curator`, the requested keys must be listed on the `pair_style` line:

```lammps
pair_style curator uncertainty force_sd force_sd_per_atom
pair_coeff * * compiled_model.pt 26 3 8 15

compute fsd all uncertainty force_sd
compute fsd_atom all uncertainty/atom force_sd_per_atom

thermo_style custom step pe c_fsd
dump d1 all custom 1 dump.curator id type x y z fx fy fz c_fsd_atom
run 0
```

For Mahalanobis on `pair_style curator`:

```lammps
pair_style curator uncertainty maha_dist maha_dist_per_atom
pair_coeff * * compiled_model.pt 26 3 8 15

compute umaha all uncertainty maha_dist
compute umaha_atom all uncertainty/atom maha_dist_per_atom
```

Use `maha_dist_per_atom` only if the exported Mahalanobis kernel is local, for example `local-gnn`.

### 3. Use `pair_style mliap unified`

`pair_style mliap unified` loads a Python object, so LAMMPS must be able to import both the LAMMPS Python package and CURATOR:

```bash
export PYTHONPATH="/path/to/lammps-install/lib/pythonX.Y/site-packages:${CURATOR_ROOT}:${PYTHONPATH}"
```

Minimal example:

```lammps
units metal
atom_style atomic
boundary p p p
newton on
read_data system.data

mass 1 55.845
mass 2 6.94
mass 3 15.999
mass 4 30.973761998

pair_style mliap unified mliap_model.pt 0
pair_coeff * * Fe Li O P

neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

thermo_style custom step pe
thermo 1
run 0
```

Unlike `pair_style curator`, `mliap unified` does not take uncertainty keys on the `pair_style` line. It exposes whatever keys are already carried by the exported model.

Scalar uncertainty example:

```lammps
compute esd all uncertainty energy_sd
compute fsd all uncertainty force_sd
thermo_style custom step pe c_esd c_fsd
```

Per-atom uncertainty example:

```lammps
compute fsd_atom all uncertainty/atom force_sd_per_atom
compute aesd_atom all uncertainty/atom atomic_energy_sd
dump d1 all custom 1 dump.mliap id type x y z fx fy fz c_fsd_atom c_aesd_atom
```

Mahalanobis example:

```lammps
compute umaha all uncertainty maha_dist
compute umaha_atom all uncertainty/atom maha_dist_per_atom
```

Use `maha_dist_per_atom` only when the exported Mahalanobis kernel is local, for example `local-full-g`, `local-ll-g`, or `local-gnn`.

### 4. Supported Uncertainty Keys

Current exported-model behavior is:

| Deploy method | `pair_style curator` | `pair_style mliap unified` |
| --- | --- | --- |
| `none` | energy and forces only | atomic energy and edge forces are written back as total energy and forces |
| `ensemble` | scalar: `energy_max`, `energy_min`, `energy_var`, `energy_sd`, `force_var`, `force_sd`; per-atom: `force_sd_per_atom` | scalar: `energy_max`, `energy_min`, `energy_var`, `energy_sd`, `force_var`, `force_sd`; per-atom: `force_sd_per_atom`, `atomic_energy_sd` |
| `mahalanobis` | scalar: `maha_dist`; per-atom: `maha_dist_per_atom` only for local TorchScript-safe kernels such as `local-gnn` | scalar: `maha_dist`; per-atom: `maha_dist_per_atom` for local kernels |

Practical rules:

- only request keys that are actually present in the exported model
- use `compute uncertainty <key>` for scalar outputs
- use `compute uncertainty/atom <key>` for per-atom outputs
- per-atom outputs can be sent to `dump custom` exactly like any other per-atom compute

### 5. Kokkos / `mliap/kk`

The same exported `mliap_model.pt` is used for `mliap/kk`.

Typical launch pattern:

```bash
lmp -k on g 1 -sf kk -pk kokkos neigh half newton on -in in.mliap
```

If `mliap/kk` is used, keep the same uncertainty commands:

- `compute uncertainty <key>`
- `compute uncertainty/atom <key>`

## Notes On mliap Uncertainty

The `mliap` uncertainty path in this repo is intentionally kept minimal.

It depends on:

- `interface/compute_uncertainty.cpp/.h`
- `interface/ML-IAP/mliap_data.h/.cpp`
- `interface/ML-IAP/mliap_unified_couple.pyx`
- `curator/simulate/lammps_mliap_interface.py`

## Notes On `data.tags`

The current `mliap` fallback path in `curator/simulate/lammps_mliap_interface.py` uses `data.tags` to map ghost neighbors back to locally owned atoms.

That requires all three ML-IAP files to be patched together:

- `interface/ML-IAP/mliap_data.h`
- `interface/ML-IAP/mliap_data.cpp`
- `interface/ML-IAP/mliap_unified_couple.pyx`

What each file does:

- `mliap_data.h`: declares the `tags` field in `MLIAPData`
- `mliap_data.cpp`: fills that field from `atom->tag`
- `mliap_unified_couple.pyx`: exposes the field to Python as `data.tags`

If only the `.pyx` file is updated but `mliap_data.h/.cpp` are not, the install is incomplete and another machine will fail when the Python interface tries to access `data.tags`.

Current design rule:

- prefer changing CURATOR-owned interface files first
- avoid changing official `pair_mliap_kokkos.cpp` unless there is no cleaner option

## Interface/Spack Helpers

User-facing Spack patch helpers now live here:

- `interface/spack/rebuild_curator_pytorch_patch.sh`
- `interface/spack/README.md`

These are the files a user should look at first if they want to maintain the CURATOR patch in their own Spack repo.

## Legacy Helper

There is also an older helper:

- `interface/patch_lammps.sh`

Treat it as legacy. It is still useful for quick experiments, but the patch-generation path above is the intended workflow.
