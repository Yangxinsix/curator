#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  patch_lammps.sh [-e] /path/to/lammps-source

Options:
  -e    use symlinks instead of copying files
  -h    show this help message

What this script does:
  1. updates LAMMPS CMakeLists.txt to add PKG_CURATOR and Torch linkage
  2. installs CURATOR interface files into the LAMMPS source tree
  3. installs the CURATOR ML-IAP bridge files into src/ML-IAP/

Notes:
  - Run this script from anywhere; it resolves paths relative to itself.
  - This script is for manual, non-Spack builds.
  - For reproducible Spack builds, prefer interface/spack/rebuild_curator_pytorch_patch.sh
EOF
}

use_symlink=false
while getopts ":he" option; do
  case "${option}" in
    e) use_symlink=true ;;
    h)
      usage
      exit 0
      ;;
    \?)
      echo "unknown option: -${OPTARG}" >&2
      usage >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
INTERFACE_DIR="${SCRIPT_DIR}"
MLIAP_INTERFACE_DIR="${INTERFACE_DIR}/ML-IAP"
LAMMPS_DIR="$1"

if [[ ! -d "${LAMMPS_DIR}" ]]; then
  echo "LAMMPS source directory does not exist: ${LAMMPS_DIR}" >&2
  exit 1
fi

if [[ ! -d "${LAMMPS_DIR}/cmake" || ! -d "${LAMMPS_DIR}/src" ]]; then
  echo "Target does not look like a LAMMPS source tree: ${LAMMPS_DIR}" >&2
  exit 1
fi

required_files=(
  "${INTERFACE_DIR}/pair_curator.cpp"
  "${INTERFACE_DIR}/pair_curator.h"
  "${INTERFACE_DIR}/compute_uncertainty.cpp"
  "${INTERFACE_DIR}/compute_uncertainty.h"
  "${MLIAP_INTERFACE_DIR}/mliap_data.cpp"
  "${MLIAP_INTERFACE_DIR}/mliap_data.h"
  "${MLIAP_INTERFACE_DIR}/mliap_unified_couple.pyx"
  "${LAMMPS_DIR}/cmake/CMakeLists.txt"
)

for f in "${required_files[@]}"; do
  if [[ ! -e "${f}" ]]; then
    echo "missing required file: ${f}" >&2
    exit 1
  fi
done

echo "Patching ${LAMMPS_DIR}/cmake/CMakeLists.txt ..."
if grep -q "PKG_CURATOR" "${LAMMPS_DIR}/cmake/CMakeLists.txt"; then
  echo "CMakeLists.txt already contains PKG_CURATOR; leaving it unchanged."
else
  python - "${LAMMPS_DIR}/cmake/CMakeLists.txt" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

text = text.replace("set(CMAKE_CXX_STANDARD 11)", "set(CMAKE_CXX_STANDARD 14)")
needle = "add_library(lammps ${ALL_SOURCES})"
if needle not in text:
    raise SystemExit("add_library(lammps ${ALL_SOURCES}) not found in CMakeLists.txt")

block = '''option(PKG_CURATOR "Enable CURATOR package" OFF)
set(CURATOR_SOURCES
  ${LAMMPS_SOURCE_DIR}/pair_curator.cpp
  ${LAMMPS_SOURCE_DIR}/compute_uncertainty.cpp
)
if(NOT PKG_CURATOR)
  list(REMOVE_ITEM ALL_SOURCES ${CURATOR_SOURCES})
endif()

add_library(lammps ${ALL_SOURCES})

if(PKG_CURATOR)
  find_package(Torch REQUIRED)
  if (TORCH_CXX_FLAGS)
    target_compile_options(lammps PUBLIC ${TORCH_CXX_FLAGS})
  endif()
  if (TARGET Torch::Torch)
    target_link_libraries(lammps PUBLIC Torch::Torch)
  else()
    target_include_directories(lammps PUBLIC "${TORCH_INCLUDE_DIRS}")
    target_link_libraries(lammps PUBLIC "${TORCH_LIBRARIES}")
  endif()
endif()
'''

path.write_text(text.replace(needle, block, 1))
PY
fi

install_one() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "${dst}")"
  if [[ "${use_symlink}" == true ]]; then
    ln -sfn "$(realpath -s "${src}")" "${dst}"
  else
    cp -f "${src}" "${dst}"
  fi
}

echo "Installing CURATOR interface files into ${LAMMPS_DIR}/src ..."
install_one "${INTERFACE_DIR}/pair_curator.cpp" "${LAMMPS_DIR}/src/pair_curator.cpp"
install_one "${INTERFACE_DIR}/pair_curator.h" "${LAMMPS_DIR}/src/pair_curator.h"
install_one "${INTERFACE_DIR}/compute_uncertainty.cpp" "${LAMMPS_DIR}/src/compute_uncertainty.cpp"
install_one "${INTERFACE_DIR}/compute_uncertainty.h" "${LAMMPS_DIR}/src/compute_uncertainty.h"
install_one "${MLIAP_INTERFACE_DIR}/mliap_data.cpp" "${LAMMPS_DIR}/src/ML-IAP/mliap_data.cpp"
install_one "${MLIAP_INTERFACE_DIR}/mliap_data.h" "${LAMMPS_DIR}/src/ML-IAP/mliap_data.h"
install_one "${MLIAP_INTERFACE_DIR}/mliap_unified_couple.pyx" "${LAMMPS_DIR}/src/ML-IAP/mliap_unified_couple.pyx"

if [[ "${use_symlink}" == true ]]; then
  echo "Installed via symlinks."
else
  echo "Installed via copies."
fi

cat <<EOF
Done.

Next steps for a manual build:
  1. configure LAMMPS with -D PKG_CURATOR=on
  2. enable any other packages you need, for example:
     -D PKG_PYTHON=on
     -D PKG_ML-IAP=on
     -D PKG_KOKKOS=on
     -D PKG_PLUMED=on
  3. point CMake to your Torch installation, for example with:
     -D CMAKE_PREFIX_PATH="\$(python - <<'PY'
import torch
print(torch.utils.cmake_prefix_path)
PY
)"
EOF
