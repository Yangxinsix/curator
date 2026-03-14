#!/bin/bash
# usage: patch_lammps.sh [-e] /path/to/lammps/
#
#
# References:
#
#    .. [#pair_nequip] https://github.com/mir-group/pair_nequip


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo $SCRIPT_DIR


do_e_mode=false

while getopts "he" option; do
   case $option in
      e)
         do_e_mode=true;;
      h) # display Help
         echo "patch_lammps.sh [-e] /path/to/lammps/"
         exit;;
   esac
done

# https://stackoverflow.com/a/9472919
shift $(($OPTIND - 1))
lammps_dir=$1

if [ "$lammps_dir" = "" ];
then
    echo "lammps_dir must be provided"
    exit 1
fi

if [ ! -d "$lammps_dir" ]
then
    echo "$lammps_dir doesn't exist"
    exit 1
fi

if [ ! -d "$lammps_dir/cmake" ]
then
    echo "$lammps_dir doesn't look like a LAMMPS source directory"
    exit 1
fi

# Check if root directory is correct
if [ ! -f pair_curator.cpp ]; then
    echo "Please run `patch_lammps.sh` from the `pair_curator.cpp` root directory."
    exit 1
fi

echo "Updating CMakeLists.txt..."
# Check for double-patch
if grep -q "PKG_CURATOR" $lammps_dir/cmake/CMakeLists.txt
then
    echo "This LAMMPS installation _seems_ to already have been patched. CMakeLists.txt file not modified."
else
    # Update CMakeLists.txt
    sed -i "s/set(CMAKE_CXX_STANDARD 11)/set(CMAKE_CXX_STANDARD 14)/" $lammps_dir/cmake/CMakeLists.txt

    # Add PKG_CURATOR option + Torch linkage with proper source gating
    python - << "PY"
from pathlib import Path

path = Path(r"$lammps_dir/cmake/CMakeLists.txt")
text = path.read_text()
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

# check if files need to be copied to lammps directory
if [ ! -f $lammps_dir/src/pair_curator.cpp ]; then
    if [ "$do_e_mode" = true ]
    then
        echo "Making source symlinks (-e)..."
        for file in *.{cpp,h}; do
            ln -s `realpath -s $file` $lammps_dir/src/$file
        done
    else
        echo "Copying files..."
        for file in *.{cpp,h}; do
            cp $file $lammps_dir/src/$file
        done
    fi
else
    echo "pair_curator.cpp file already exists. No files copied."
fi


echo "Done!"
