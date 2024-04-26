.. _LAMMPS: https://github.com/lammps/lammps
.. _CUDA: https://developer.nvidia.com/cuda-11-7-0-download-archive
.. _cuDNN: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

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
    module load intel/2022.2.0.mpi
    module load cmake
    module load gcc/11.2.0-binutils-2.37


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

.. code-block:: bash

    cd <path_to_lammps>
    mkdir build
    cd build
    cmake ../cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DMKL_INCLUDE_DIR="$CONDA_PREFIX/include" -D PKG_GPU=on

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