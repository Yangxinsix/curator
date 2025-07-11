.. _LAMMPS: https://github.com/lammps/lammps
.. _CUDA: https://developer.nvidia.com/cuda-11-7-0-download-archive
.. _cuDNN: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
.. _KOKKOS: https://github.com/kokkos/kokkos

Using CURATOR with LAMMPS ML-IAP package
=================
ML-IAP provided an unified interface for message-passing models and can optimize the performance of these models on GPUs.
It is possible to run large-scale simulations in LAMMPS with this pacakge.

Now models trained with CURATOR can be used in LAMMPS through the original GPU package and ML-IAP interface.

In the following we will focus on the installation and usage of this new ML-IAP interface, which calls the model directly from python and supports cuEquivariance acceleration, multi-GPU simulation, and atomic virials.

Install LAMMPS with ML-IAP package
----------------------------------

Requirements
~~~~~~~~~~~~~
In addition to standard CURATOR dependencies, you will also need:


Preparing the model for ML-IAP
------------------------------
ML-IAP uses atomic energies and pair-wise forces to calculate the energy, forces, and virials of a simulation system. 
Therefore we need to convert the original model to output atomic energies and pair-wise forces. This can be done with the following command.

.. code-block:: bash

    curator-deploy <path_to_model.ckpt> --lammps --elements <element1> <element2>

The element list must contain all elements that you will use in your lammps simulation.
By default, the converted model will be deployed to ``lmp_model.pt`` at current directory. You can also specify the path by using ``--target_path <path_to_lmp_model.pt>``.

Using CURATOR with ML-IAP in LAMMPS
-----------------------------------

LAMMPS input file
~~~~~~~~~~~~~~~~~
Your LAMMPS input should begin with standard settings:

.. code-block:: bash

    units         metal
    atom_style    atomic
    atom_modify   map yes
    newton        on

Then define the ML-IAP pair style with your converted model:

.. code-block:: bash

    pair_style      mliap unified lmp_model.pt 0
    pair_coeff      * * C H O N

.. note::

    The 0 after the model filename is a required parameter for the unified ML-IAP interface.

The element list after ``pair_coeff * *`` should be ordered as you want them to appear in LAMMPS, and must be a subset of the elements your model was trained on.

Command line inputs
~~~~~~~~~~~~~~~~~~~~
When running LAMMPS with MACE/ML-IAP, use these command line options for GPU acceleration:

.. code-block:: bash

    lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in your_input.in

This enables 1 GPU with Kokkos. You can change ``g 1`` to use multiple GPUs if your system supports it.

For multi-GPU simulations with MPI, use:

.. code-block:: bash

    mpirun -np 2 lmp -k on g 2 -sf kk -pk kokkos newton on neigh half -in input.in

This example uses 2 MPI processes with 2 GPUs. Adjust the number of processes (``-np``) and GPUs (``g``) based on your hardware.

Example LAMMPS script
~~~~~~~~~~~~~~~~~~~~~
Here's a complete example LAMMPS script for using CURATOR models with ML-IAP:

.. code-block:: bash

    # MACE ML-IAP example
    units         metal
    atom_style    atomic
    atom_modify   map yes
    newton        on

    # Read structure
    read_data     structure.data

    # Set up MACE potential
    pair_style    mliap unified lmp_model.pt 0
    pair_coeff    * * C H O N

    # Run settings
    timestep      0.0001
    thermo        100

    # MD run
    fix           1 all nvt temp 300 300 100
    run           1000

Run this script with:

.. code-block:: bash

    lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in input.in