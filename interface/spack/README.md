# Spack Patch Helpers

This directory exists so users can manage the CURATOR LAMMPS patch without digging into editor-specific or agent-specific tooling.

## Files

- `rebuild_curator_pytorch_patch.sh`: rebuilds `curator_pytorch.patch` from the files under `interface/`

## What This Script Assumes

You already have:

- a local Spack repo that contains `packages/lammps/package.py`
- a LAMMPS source tarball
- a CURATOR checkout

## Typical Usage

```bash
bash "${CURATOR_ROOT}/interface/spack/rebuild_curator_pytorch_patch.sh" \
  --lammps-tarball "${LAMMPS_TARBALL}" \
  --spack-package-dir "${SPACK_REPO}/packages/lammps" \
  --python-exe "${PYTHON_EXE}"
```

The script will:

1. unpack the LAMMPS tarball
2. copy the CURATOR interface files into the unpacked source tree
3. regenerate `curator_pytorch.patch`
4. run `patch --dry-run`
5. update the patch `sha256` in `package.py` if it finds the expected patch entry

## Canonical CURATOR Sources

These are the files the script treats as the source of truth:

- `interface/pair_curator.cpp`
- `interface/pair_curator.h`
- `interface/compute_uncertainty.cpp`
- `interface/compute_uncertainty.h`
- `interface/ML-IAP/mliap_data.cpp`
- `interface/ML-IAP/mliap_data.h`
- `interface/ML-IAP/mliap_unified_couple.pyx`

Optional extra overrides can still live outside `interface/`, but the default user-facing workflow should start here.
