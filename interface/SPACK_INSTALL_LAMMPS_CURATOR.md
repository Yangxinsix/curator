# Install LAMMPS + PLUMED + CURATOR + MC with Spack

This document is for installing one production LAMMPS build from scratch on a new machine.

Target build:

- `pair_style curator`
- `pair_style mliap`
- `pair_style mliap/kk`
- `PLUMED`
- `MC`
- `KOKKOS`
- multi-GPU capable `mliap/kk`

The intended final result is one LAMMPS installation that includes all of the above.

## 1. Requirements

You need:

- a working Spack installation
- a CUDA toolkit already installed on the machine
- a Python environment you want LAMMPS to use
- this CURATOR repository
- a local Spack repo that contains the patched `packages/lammps/package.py`
- a LAMMPS source tarball compatible with the CURATOR patch

Recommended LAMMPS release for this interface:

- `22 Jul 2025 - Update 3`

## 2. Choose Machine-Local Paths

Set these variables on the target machine.

```bash
export CURATOR_ROOT=<path-to-curator-repo>
export SPACK_ROOT=<path-to-spack>
export SPACK_REPO=<path-to-your-local-spack-repo>
export PYTHON_EXE=<python-executable-to-use>
export CUDA_PREFIX=<cuda-install-prefix>
export CUDA_ARCH=<gpu-arch-such-as-80>
export LAMMPS_TARBALL=<path-to-lammps-tarball>
export ENV_DIR=<path-to-dedicated-spack-environment>
```

Examples of what these mean:

- `${CURATOR_ROOT}` contains `interface/`, `curator/`, and the rest of this repo
- `${SPACK_REPO}` contains `packages/lammps/package.py`
- `${PYTHON_EXE}` is the Python that already has the packages you want to run with

## 3. Prepare the Python Environment

LAMMPS should use one existing Python environment. Do not let Spack build a second Python unless you explicitly want that.

Make sure `${PYTHON_EXE}` can import at least:

- `torch`
- `numpy`
- `cython`

Check:

```bash
"${PYTHON_EXE}" - <<'PY'
import torch, numpy, Cython
print('python ok')
PY
```

Get the Python prefix and exact version:

```bash
export PYTHON_PREFIX=$(dirname "$(dirname "${PYTHON_EXE}")")
export PY_VER=$("${PYTHON_EXE}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
```

## 4. Load Spack and Add the Local Repo

```bash
source "${SPACK_ROOT}/share/spack/setup-env.sh"
spack repo add "${SPACK_REPO}" || true
```

Verify that your repo contains the patched LAMMPS package:

```bash
test -f "${SPACK_REPO}/packages/lammps/package.py"
```

## 5. Rebuild the CURATOR Patch Before Installing

Do this whenever you want the Spack package to reflect the current CURATOR interface files.

```bash
bash "${CURATOR_ROOT}/interface/spack/rebuild_curator_pytorch_patch.sh" \
  --lammps-tarball "${LAMMPS_TARBALL}" \
  --spack-package-dir "${SPACK_REPO}/packages/lammps" \
  --python-exe "${PYTHON_EXE}"
```

This script:

1. unpacks the LAMMPS tarball
2. copies the CURATOR-owned interface files into the unpacked source tree
3. regenerates `curator_pytorch.patch`
4. runs `patch --dry-run`
5. updates the patch `sha256` in `package.py`

The CURATOR-owned LAMMPS-facing files are:

- `interface/pair_curator.cpp`
- `interface/pair_curator.h`
- `interface/compute_uncertainty.cpp`
- `interface/compute_uncertainty.h`
- `interface/ML-IAP/mliap_data.cpp`
- `interface/ML-IAP/mliap_data.h`
- `interface/ML-IAP/mliap_unified_couple.pyx`

## 6. Use CUDA-Aware MPI

For single-GPU testing, non-CUDA-aware MPI may still work.

For multi-GPU `mliap/kk`, treat CUDA-aware MPI as required.

A practical OpenMPI spec is:

```bash
export MPI_SPEC='openmpi@5.0.10 +cuda +atomics +fortran fabrics=none schedulers=none romio-filesystem=none'
```

This is not tied to one machine. The exact version can be adjusted, but the key point is:

- MPI must be CUDA-aware

## 7. Create a Dedicated Spack Environment

```bash
mkdir -p "${ENV_DIR}"
cat > "${ENV_DIR}/spack.yaml" <<EOF2
spack:
  repos:
  - ${SPACK_REPO}

  packages:
    python:
      buildable: false
      externals:
      - spec: python@${PY_VER}
        prefix: ${PYTHON_PREFIX}

    cuda:
      buildable: false
      externals:
      - spec: cuda
        prefix: ${CUDA_PREFIX}

  specs:
  - lammps +kokkos +python +plumed +ml-iap +mc cuda_arch=${CUDA_ARCH} ^python@${PY_VER} ^${MPI_SPEC}
EOF2
```

Why this matters:

- `+kokkos`: needed for `mliap/kk`
- `+python`: needed for `mliap unified`
- `+plumed`: needed for PLUMED support
- `+ml-iap`: needed for `pair_style mliap`
- `+mc`: needed for `fix gcmc`, `fix atom/swap`, and other MC fixes
- external Python: avoids building an extra Python stack inside Spack
- external CUDA: makes the build use the machine CUDA installation

## 8. Concretize and Install

```bash
spack -e "${ENV_DIR}" concretize -f
spack -e "${ENV_DIR}" install
```

## 9. Load LAMMPS and Find the Installed Prefix

```bash
spack -e "${ENV_DIR}" find -p lammps
spack -e "${ENV_DIR}" load lammps
which lmp
```

## 10. Install the LAMMPS Python Module into the Same Python Environment

This step is required.

Having `PKG_PYTHON=ON` is not enough. The Python interpreter used at runtime must also be able to do:

```bash
import lammps
```

If your Spack install already put the module into the correct `site-packages`, verify it:

```bash
"${PYTHON_EXE}" -c "import lammps; from lammps import lammps as L; print(lammps.__file__)"
```

If that fails, install the LAMMPS Python package from the installed LAMMPS tree.

Typical options are:

```bash
cmake --build <build-dir> --target install-python
```

or, from the LAMMPS source tree:

```bash
cd <lammps-source>/python
"${PYTHON_EXE}" install.py -p lammps -l <installed-liblammps.so-or-.so.0> -v <lammps-source>/src/version.h -f
```

Acceptance condition for this step:

```bash
"${PYTHON_EXE}" -c "import lammps; from lammps import lammps as L"
```

must work without setting `PYTHONPATH` by hand.

## 11. Validate the Installed Binary

Run:

```bash
lmp -h
```

The installed packages should include at least:

- `KOKKOS`
- `MC`
- `ML-IAP`
- `PLUMED`
- `PYTHON`
- `curator`

A quick filter:

```bash
lmp -h | rg -i 'kokkos|mc|ml-iap|plumed|python|curator'
```

## 12. Validate CUDA-Aware MPI

For multi-GPU `mliap/kk`, verify MPI support explicitly:

```bash
ompi_info --all | rg -i 'mpi_built_with_cuda_support|opal_built_with_cuda_support'
```

Both should report `true`.

## 13. Minimal Runtime Checks

Before doing production runs, check each required feature.

### 13.1 `pair_style curator`

Use a minimal input that defines:

- `pair_style curator`
- `pair_coeff * * <compiled-curator-model.pt> ...`

and confirm `run 0` works.

### 13.2 `pair_style mliap`

Use a minimal input that defines:

- `pair_style mliap unified <mliap-model.pt> 0`
- `pair_coeff * * <elements...>`

and confirm `run 0` works.

### 13.3 `pair_style mliap/kk`

Use the same model with the Kokkos pair style and confirm GPU execution works.

Typical launch pattern:

```bash
lmp -k on g 1 -sf kk -pk kokkos neigh half newton on -in <input-file>
```

### 13.4 `fix gcmc`

Confirm the `MC` package is actually usable with a minimal test that includes:

```lammps
fix gc all gcmc ...
```

and check that both `pair_curator` and `mliap` can complete a short run.

### 13.5 `fix plumed`

Confirm PLUMED support with a minimal input that includes:

```lammps
fix pl all plumed plumedfile plumed.dat outfile plumed.out
```

## 14. Acceptance Criteria

The install is complete only if all of these are true:

1. `pair_style curator` works
2. `pair_style mliap` works
3. `pair_style mliap/kk` works on GPU
4. `fix gcmc` is available
5. `fix plumed` is available
6. `import lammps` works in the same Python environment used by LAMMPS
7. MPI is CUDA-aware if you need multi-GPU `mliap/kk`

## 15. What Not To Do

- Do not keep a production install on non-CUDA-aware MPI if your target is multi-GPU `mliap/kk`.
- Do not let Spack build a second Python unless you explicitly want that.
- Do not patch upstream `src/KOKKOS/pair_mliap_kokkos.cpp` in the normal workflow.
- Do not spread CURATOR-owned interface code across random files; keep it under `interface/`.
- Do not optimize this process around old install hashes or machine-specific absolute paths.
- Do not update only `mliap_unified_couple.pyx` while leaving `mliap_data.h/.cpp` stale. The current `mliap` fallback path needs all three ML-IAP files together.

## 16. Canonical Source of Truth

For this installation workflow, the source of truth is:

- CURATOR interface files under `interface/`
- the local Spack repo package under `${SPACK_REPO}/packages/lammps/package.py`
- the rebuilt `curator_pytorch.patch`

If interface files change, rebuild the patch first, then install or rebuild LAMMPS.
