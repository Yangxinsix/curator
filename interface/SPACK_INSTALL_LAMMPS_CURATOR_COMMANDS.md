# Command-Only: Install LAMMPS + PLUMED + CURATOR + MC with Spack

This is the short command version for another machine.

## 0. Set Variables

```bash
export CURATOR_ROOT=<path-to-curator-repo>
export SPACK_ROOT=<path-to-spack>
export SPACK_REPO=<path-to-local-spack-repo>
export PYTHON_EXE=<python-executable>
export CUDA_PREFIX=<cuda-prefix>
export CUDA_ARCH=<gpu-arch-such-as-80>
export LAMMPS_TARBALL=<path-to-lammps-tarball>
export ENV_DIR=<path-to-spack-env>
export MPI_SPEC='openmpi@5.0.10 +cuda +atomics +fortran fabrics=none schedulers=none romio-filesystem=none'

export PYTHON_PREFIX=$(dirname "$(dirname "${PYTHON_EXE}")")
export PY_VER=$("${PYTHON_EXE}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
```

## 1. Check Python Environment

```bash
"${PYTHON_EXE}" - <<'PY'
import torch, numpy, Cython
print('python ok')
PY
```

## 2. Load Spack and Add Repo

```bash
source "${SPACK_ROOT}/share/spack/setup-env.sh"
spack repo add "${SPACK_REPO}" || true
test -f "${SPACK_REPO}/packages/lammps/package.py"
```

## 3. Rebuild CURATOR Patch

```bash
bash "${CURATOR_ROOT}/interface/spack/rebuild_curator_pytorch_patch.sh" \
  --lammps-tarball "${LAMMPS_TARBALL}" \
  --spack-package-dir "${SPACK_REPO}/packages/lammps" \
  --python-exe "${PYTHON_EXE}"
```

This patch rebuild now expects all three ML-IAP interface files under `interface/ML-IAP/`:

- `mliap_data.cpp`
- `mliap_data.h`
- `mliap_unified_couple.pyx`

## 4. Create Spack Environment

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

## 5. Concretize and Install

```bash
spack -e "${ENV_DIR}" concretize -f
spack -e "${ENV_DIR}" install
```

## 6. Load LAMMPS

```bash
spack -e "${ENV_DIR}" find -p lammps
spack -e "${ENV_DIR}" load lammps
which lmp
lmp -h
```

## 7. Check Required Packages

```bash
lmp -h | rg -i 'kokkos|mc|ml-iap|plumed|python|curator'
```

## 8. Check CUDA-Aware MPI

```bash
ompi_info --all | rg -i 'mpi_built_with_cuda_support|opal_built_with_cuda_support'
```

Expected: both CUDA support checks should be `true`.

## 9. Install LAMMPS Python Module

If plain import already works, keep it.

```bash
"${PYTHON_EXE}" -c "import lammps; from lammps import lammps as L; print(lammps.__file__)"
```

If that fails, run one of these from the LAMMPS build or source tree:

```bash
cmake --build <build-dir> --target install-python
```

or

```bash
cd <lammps-source>/python
"${PYTHON_EXE}" install.py -p lammps -l <installed-liblammps.so-or-.so.0> -v <lammps-source>/src/version.h -f
```

Then verify again:

```bash
"${PYTHON_EXE}" -c "import lammps; from lammps import lammps as L; print(lammps.__file__)"
```

## 10. Minimal Runtime Checks

### 10.1 `pair_style curator`

```bash
lmp -in <input-that-uses-pair_style-curator>
```

### 10.2 `pair_style mliap`

```bash
lmp -in <input-that-uses-pair_style-mliap>
```

### 10.3 `pair_style mliap/kk` on GPU

```bash
lmp -k on g 1 -sf kk -pk kokkos neigh half newton on -in <input-that-uses-pair_style-mliap/kk>
```

### 10.4 `fix gcmc`

```bash
lmp -in <input-that-uses-fix-gcmc-with-curator-or-mliap>
```

### 10.5 `fix plumed`

```bash
lmp -in <input-that-uses-fix-plumed>
```

## 11. Acceptance Checklist

```bash
lmp -h | rg -i 'kokkos|mc|ml-iap|plumed|python|curator'
"${PYTHON_EXE}" -c "import lammps; from lammps import lammps as L"
ompi_info --all | rg -i 'mpi_built_with_cuda_support|opal_built_with_cuda_support'
```

Required final state:

- `pair_style curator` works
- `pair_style mliap` works
- `pair_style mliap/kk` works on GPU
- `fix gcmc` works
- `fix plumed` works
- `import lammps` works in the same Python environment
- CUDA-aware MPI is enabled for multi-GPU `mliap/kk`
