Using models for simulations
============================

When model training is finished, you can use the models for simulations.

Similar to the training part, you can run a minimal example after simply providing the trained model path and the initial structure for the simulation:

.. colde-block:: bash
    :linenos:

    curator-simulate model_path=best_model.ckpt simulator.init_traj=initial_structure.traj

CURATOR currently supports ASE and LAMMPS simulators for different simulation tasks. We pre-defined some simulation parameters which mostly aim to support some common simulation tasks that can be conducted in ASE or LAMMPS. 

Considering the massive number of simulations that machine learning potentials can be used for, it is also possible for the users to define a class that is capable of running simulations they would like to perform.

ASE calculator
-------------

It is possible to directly use the model by passing it into a ASE calculator. 

You can create a calculator by simply providing the path of your model. If a set of models are provided, an ensemble model will be created which uses the mean of calculated properties for simulation.

The following example shows how to run a MD simulation for a cubix box with water molecules.

.. code-block:: python
    :linenos:

    from ase.io import read
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase import units
    from ase.md.langevin import Langevin
    from curator.simulate import MLCalculator
    mlcalc = MLCalculator('/workspace/work/water_test/model_path')    # should be the path of your model or the directy it at

    atoms = read('water_test/dataset_1593.traj', -1)
    atoms.calc = mlcalc

    MaxwellBoltzmannDistribution(atoms, temperature_K=300)        # initialize the velocities of atoms using Maxwell-Boltzmann distribution

    dyn = Langevin(
        atoms,
        timestep=0.5 * units.fs,
        temperature_K=300.0,  # temperature in K
        friction=0.01 / units.fs,
    )

    # define a function to record energy and temperature 
    steps = 0
    def print_energy(atoms=atoms):
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB) / atoms.get_global_number_of_atoms()
        global steps
        steps += 1
        status = f"Steps={steps:12.3f} Epot={epot:12.3f} Ekin={ekin:12.3f} temperature={temp:8.2f}"
        print(status)
        with open('MD.log', 'a') as f:
            f.write(status + '\n')

    traj = Trajectory('MD.traj', mode='a', atoms=atoms)
    dyn.attach(print_energy)
    dyn.attach(traj.write, interval=10)
    dyn.run(1000)

ASE simulator
-------------
ASE is the default simulator for 


Lammps simulator
----------------


