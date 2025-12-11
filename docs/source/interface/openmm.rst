OpenMM interface
=================
OpenMM is a highly-optimized GPU MD code. By using OpenMM-Torch plugin, it enables full GPU MD simulation with machine learning potentials.

Intallation
------------
.. code-block:: bash

    conda install pytorch-gpu=1.13.1 openmm-torch -c conda-forge
    pip install openmm-ml


Export a CURATOR model for OpenMM
----------------------------------
We expose a small adapter around `openmm-ml` so a trained CURATOR potential can run through the OpenMM-Torch plugin.

.. code-block:: python

    from ase.io import read
    from curator.simulate import export_curator_to_openmm_torchscript

    atoms = read("structure.pdb")
    export_curator_to_openmm_torchscript(
        model="runs/best.ckpt",                   # path or torch.nn.Module
        output_path="curator_openmm.pt",         # TorchScript to load in OpenMM
        atomic_numbers=atoms.get_atomic_numbers(),   # also accepts element symbols ["C", "H", ...]
        # cutoff=None will be inferred from the model if available
        length_scale=10.0,                       # nm -> Angstrom conversion
        energy_scale=96.48533212331002,          # eV -> kJ/mol
    )

The exported TorchScript module accepts OpenMM tensors (positions and box vectors in nm) and returns a scalar energy in kJ/mol.
For debugging outside OpenMM you can call ``forward_with_forces`` on the scripted module to get both energy and forces.


Using with openmm-ml
---------------------
``openmm-ml`` wraps OpenMM-Torch and will automatically compute forces from the returned energy. A minimal driver looks like:

.. code-block:: python

    from openmm import app, unit, openmm
    from openmmml import MLPotential

    pdb = app.PDBFile("structure.pdb")
    ff = app.ForceField("amber14-all.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff)

    # load TorchScript exported above
    ml = MLPotential(model_path="curator_openmm.pt", model_format="torchscript")
    system.addForce(ml.createForce(pdb.topology))

    integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    sim = app.Simulation(pdb.topology, system, integrator)
    sim.context.setPositions(pdb.positions)
    sim.minimizeEnergy()
    sim.step(1000)

Notes:
- The wrapper assumes the CURATOR model was trained on energies in eV and positions in Å; adjust ``energy_scale`` or ``length_scale`` during export if your units differ.
- ``atomic_numbers`` are baked into the scripted module because OpenMM does not pass them to TorchForce by default; export once per chemical system/topology.
