# CURATOR LAMMPS Interface

This directory contains the CURATOR-owned files that are patched into LAMMPS:

- `pair_curator.cpp/.h`: `pair_style curator`
- `compute_uncertainty.cpp/.h`: `compute uncertainty <key>`
- `ML-IAP/mliap_data.h/.cpp`: ML-IAP data plumbing needed by the CURATOR `mliap` bridge
- `ML-IAP/mliap_unified_couple.pyx`: Python bridge for `pair_style mliap unified`

The intended design is:

- keep CURATOR-specific LAMMPS code here
- prefer patching these files instead of editing official LAMMPS source files
- keep the `mliap` uncertainty path as a CURATOR-side interface change, not a `pair_mliap_kokkos.cpp` fork

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
