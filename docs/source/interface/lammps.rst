.. _LAMMPS: https://github.com/lammps/lammps
.. _CUDA: https://developer.nvidia.com/cuda-11-7-0-download-archive
.. _cuDNN: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
.. _KOKKOS: https://github.com/kokkos/kokkos

LAMMPS interface
=================
ASE MD engine is aimed for convenience but not for performance, so we need a new engine to make molecular dynamics faster.
Now Curator has an interface to LAMMPS_, which is a popular MD code and supports GPU computing.

Requirements
-------------
CUDA_ (11.8)

cuDNN_

Standalone CUDA and cuDNN are required for building LAMMPS. If you want to use different version of CUDA, make sure that ``PyTorch`` version is compatible with CUDA.
The installation guide can be referred to:

`CUDA installation <https://developer.nvidia.com/cuda-11-7-0-download-archive>`_

`cuDNN installation <https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html>`_


Build LAMMPS on Gbar
---------------------
DTU Gbar has quite useful interactive GPUs and pre-installed packages for building LAMMPS. 
Before building LAMMPS, the following packages are needed to be loaded.

.. code-block:: bash

    module purge
    module load cmake/3.24.0
    module load intel/2020u1
    module load cuda/11.8 gcc/10.2.0 openmpi/4.0.5-intel


Download LAMMPS
---------------------
The installation is only tested with the following version of LAMMPS:

.. code-block:: bash

    git clone -b stable_23Jun2022_update3 --depth 1 git@github.com:lammps/lammps


Patch curator into LAMMPS
------------------------

.. code-block:: bash

    cd <path_to_Curator/interface>
    ./patch_lammps.sh <path_to_lammps>


Configure LAMMPS
-----------------
We need to provide the path of Torch library to CMake for installation. This can be achieved by adding `-DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`` tag to CMake.
CMake will then use the Torch library from PyTorch python pacakge to configure LAMMPS.
.. code-block:: bash

    cd <path_to_lammps>
    mkdir build
    cd build
    cmake ../cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DMKL_INCLUDE_DIR="$CONDA_PREFIX/include"
    cmake -C ../cmake/presets/basic.cmake -C ../cmake/presets/kokkos-cuda.cmake

Note that the default Torch library may use pre-C++11 ABI, which is not compatible with the C++17 ABI required by KOKKOS_. In this case, the users should download LibTorch with C++11 ABI and set the path to the CMake configuration.
.. code-block::bash

    # change cuda and pytorch to your version
    wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu118.zip
    unzip libtorch-cxx11-abi-shared-with-deps-2.3.1+cu118.zip
    cd <path_to_lammps>
    mkdir build
    cd build
    cmake ../cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch -DMKL_INCLUDE_DIR="$CONDA_PREFIX/include" -C ../cmake/presets/basic.cmake -C ../cmake/presets/kokkos-cuda.cmake

Another potential problem is that LibTorch usually requires quite new version of GLIBC library. This often requires the users to upgrade their GLIBC, but this can be quite cumbersome and the users are referred to elsewhere for this issue.

Build LAMMPS
---------------
.. code-block:: bash

    make -j$(nproc)

Deploy trained model
---------------------
No documentation yet.

Deploy trained model ensemble
------------------------------
No documentation yet.