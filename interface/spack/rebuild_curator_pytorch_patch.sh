#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  rebuild_curator_pytorch_patch.sh \
    --lammps-tarball /path/to/lammps.tar.gz \
    --spack-package-dir /path/to/local-spack-repo/packages/lammps \
    [--curator-root /path/to/curator] \
    [--override-root /path/to/extra/overrides] \
    [--patch-name curator_pytorch.patch] \
    [--python-exe /path/to/python]

Required:
  --lammps-tarball     LAMMPS source tarball used as the patch base
  --spack-package-dir  Directory containing package.py for the local LAMMPS Spack package

Optional:
  --curator-root       CURATOR repo root; default is inferred from this script location
  --override-root      Extra override tree copied into the LAMMPS source tree
  --patch-name         Patch filename inside the Spack package dir (default: curator_pytorch.patch)
  --python-exe         Only used for informational echo; default: python3
EOF
}

CURATOR_ROOT=""
OVERRIDE_ROOT=""
LAMMPS_TARBALL=""
SPACK_PACKAGE_DIR=""
PATCH_NAME="curator_pytorch.patch"
PYTHON_EXE="python3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --curator-root)
      CURATOR_ROOT="$2"
      shift 2
      ;;
    --override-root)
      OVERRIDE_ROOT="$2"
      shift 2
      ;;
    --lammps-tarball)
      LAMMPS_TARBALL="$2"
      shift 2
      ;;
    --spack-package-dir)
      SPACK_PACKAGE_DIR="$2"
      shift 2
      ;;
    --patch-name)
      PATCH_NAME="$2"
      shift 2
      ;;
    --python-exe)
      PYTHON_EXE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${LAMMPS_TARBALL}" || -z "${SPACK_PACKAGE_DIR}" ]]; then
  usage >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
if [[ -z "${CURATOR_ROOT}" ]]; then
  CURATOR_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
fi

if [[ -z "${OVERRIDE_ROOT}" ]]; then
  OVERRIDE_ROOT="${CURATOR_ROOT}/lammps_patch_overrides"
fi

INTERFACE_DIR="${CURATOR_ROOT}/interface"
MLIAP_INTERFACE_DIR="${INTERFACE_DIR}/ML-IAP"
PATCH_FILE="${SPACK_PACKAGE_DIR}/${PATCH_NAME}"
PACKAGE_FILE="${SPACK_PACKAGE_DIR}/package.py"

required_files=(
  "${INTERFACE_DIR}/compute_uncertainty.cpp"
  "${INTERFACE_DIR}/compute_uncertainty.h"
  "${INTERFACE_DIR}/compute_uncertainty_atom.cpp"
  "${INTERFACE_DIR}/compute_uncertainty_atom.h"
  "${INTERFACE_DIR}/pair_curator.cpp"
  "${INTERFACE_DIR}/pair_curator.h"
  "${MLIAP_INTERFACE_DIR}/mliap_data.cpp"
  "${MLIAP_INTERFACE_DIR}/mliap_data.h"
  "${MLIAP_INTERFACE_DIR}/mliap_unified_couple.pyx"
  "${MLIAP_INTERFACE_DIR}/mliap_unified_couple_kokkos.pyx"
  "${MLIAP_INTERFACE_DIR}/pair_mliap.cpp"
  "${MLIAP_INTERFACE_DIR}/pair_mliap.h"
  "${LAMMPS_TARBALL}"
  "${PACKAGE_FILE}"
)

for f in "${required_files[@]}"; do
  if [[ ! -e "${f}" ]]; then
    echo "missing required file: ${f}" >&2
    exit 1
  fi
done

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmpdir}"
}
trap cleanup EXIT

tar -xf "${LAMMPS_TARBALL}" -C "${tmpdir}"
tar_root="$(
  "${PYTHON_EXE}" - "${LAMMPS_TARBALL}" <<'PY'
import sys
import tarfile

with tarfile.open(sys.argv[1], "r:*") as tf:
    first = tf.getmembers()[0].name
print(first.split("/", 1)[0])
PY
)"
base="${tmpdir}/${tar_root}"
mod="${tmpdir}/mod"

cp -r "${base}" "${mod}"

if [[ -f "${PATCH_FILE}" && -s "${PATCH_FILE}" ]]; then
  patch -p1 -d "${mod}" < "${PATCH_FILE}" >/dev/null
fi

cp "${INTERFACE_DIR}/compute_uncertainty.cpp" "${mod}/src/compute_uncertainty.cpp"
cp "${INTERFACE_DIR}/compute_uncertainty.h" "${mod}/src/compute_uncertainty.h"
cp "${INTERFACE_DIR}/compute_uncertainty_atom.cpp" "${mod}/src/compute_uncertainty_atom.cpp"
cp "${INTERFACE_DIR}/compute_uncertainty_atom.h" "${mod}/src/compute_uncertainty_atom.h"
cp "${INTERFACE_DIR}/pair_curator.cpp" "${mod}/src/pair_curator.cpp"
cp "${INTERFACE_DIR}/pair_curator.h" "${mod}/src/pair_curator.h"
mkdir -p "${mod}/src/ML-IAP"
cp "${MLIAP_INTERFACE_DIR}/mliap_data.cpp" "${mod}/src/ML-IAP/mliap_data.cpp"
cp "${MLIAP_INTERFACE_DIR}/mliap_data.h" "${mod}/src/ML-IAP/mliap_data.h"
cp "${MLIAP_INTERFACE_DIR}/mliap_unified_couple.pyx" "${mod}/src/ML-IAP/mliap_unified_couple.pyx"
cp "${MLIAP_INTERFACE_DIR}/pair_mliap.cpp" "${mod}/src/ML-IAP/pair_mliap.cpp"
cp "${MLIAP_INTERFACE_DIR}/pair_mliap.h" "${mod}/src/ML-IAP/pair_mliap.h"
mkdir -p "${mod}/src/KOKKOS"
cp "${MLIAP_INTERFACE_DIR}/mliap_unified_couple_kokkos.pyx" "${mod}/src/KOKKOS/mliap_unified_couple_kokkos.pyx"

override_targets=()
if [[ -d "${OVERRIDE_ROOT}" ]]; then
  while IFS= read -r -d '' src_file; do
    rel="${src_file#${OVERRIDE_ROOT}/}"
    if [[ "${rel}" == "src/ML-IAP/mliap_unified_couple.pyx" ]]; then
      continue
    fi
    mkdir -p "${mod}/$(dirname "${rel}")"
    cp "${src_file}" "${mod}/${rel}"
    override_targets+=("${rel}")
  done < <(find "${OVERRIDE_ROOT}" -type f -print0 | sort -z)
fi

targets=(
  "cmake/CMakeLists.txt"
  "src/compute_uncertainty.cpp"
  "src/compute_uncertainty.h"
  "src/compute_uncertainty_atom.cpp"
  "src/compute_uncertainty_atom.h"
  "src/pair_curator.cpp"
  "src/pair_curator.h"
  "src/ML-IAP/mliap_data.cpp"
  "src/ML-IAP/mliap_data.h"
  "src/ML-IAP/mliap_unified_couple.pyx"
  "src/ML-IAP/pair_mliap.cpp"
  "src/ML-IAP/pair_mliap.h"
  "src/KOKKOS/mliap_unified_couple_kokkos.pyx"
)
targets+=("${override_targets[@]}")

: > "${PATCH_FILE}"
for rel in "${targets[@]}"; do
  src="/dev/null"
  dst="/dev/null"
  [[ -f "${base}/${rel}" ]] && src="${base}/${rel}"
  [[ -f "${mod}/${rel}" ]] && dst="${mod}/${rel}"
  diff -u --label "a/${rel}" "${src}" --label "b/${rel}" "${dst}" >> "${PATCH_FILE}" || true
  printf '\n' >> "${PATCH_FILE}"
done

patch --dry-run -p1 -d "${base}" < "${PATCH_FILE}" >/dev/null

new_sha="$(sha256sum "${PATCH_FILE}" | awk '{print $1}')"

"${PYTHON_EXE}" - "${PACKAGE_FILE}" "${PATCH_NAME}" "${new_sha}" <<'PY'
import re
import sys
from pathlib import Path

package_file = Path(sys.argv[1])
patch_name = sys.argv[2]
new_sha = sys.argv[3]
text = package_file.read_text()
pattern = re.compile(rf'(patch\(\s*"{re.escape(patch_name)}",\s*sha256=")([0-9a-f]{{64}})(")', re.S)
text_new, n = pattern.subn(lambda m: f"{m.group(1)}{new_sha}{m.group(3)}", text, count=1)
if n == 1:
    package_file.write_text(text_new)
else:
    print(f"warning: did not update sha in {package_file}; update it manually", file=sys.stderr)
PY

echo "patch rebuilt successfully"
echo "patch: ${PATCH_FILE}"
echo "sha256: ${new_sha}"
echo "package: ${PACKAGE_FILE}"
