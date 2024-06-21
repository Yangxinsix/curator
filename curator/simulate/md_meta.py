from ase.md.langevin import Langevin
from ase.calculators.plumed import Plumed
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read, write, Trajectory

import numpy as np
import torch
import sys
import glob
import toml
import argparse
from pathlib import Path
import logging
from ase.constraints import FixAtoms
from curator.simulate import MDControl
from omegaconf import DictConfig
from ase.calculators.calculator import Calculator
from curator.simulate import PrintEnergy

def MD_meta(config: DictConfig, MLcalc: Calculator,PE: PrintEnergy) -> None:
    """ MD simulation with meta dynamics. Requires plumed installed.

    Args:
        config (DictConfig): configuration file
        MLcalc (Calculator): ML calculator
        PE (PrintEnergy): PrintEnergy object
    """
    # Setup log file 
    log = PE.logger
    runHandler = logging.FileHandler('MD.log', mode='w')
    runHandler.setLevel(logging.DEBUG)
    runHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    errorHandler = logging.FileHandler('warning.log', mode='w')
    errorHandler.setLevel(logging.WARNING)
    errorHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    log.addHandler(runHandler)
    log.addHandler(errorHandler)
    log.addHandler(logging.StreamHandler())
    
    # set up md start configuration
    if not os.path.isfile(config.init_traj):
        raise RuntimeError("Please provide valid initial data path!")
    images = read(config.init_traj, ':')
    start_index = np.random.choice(len(images)) if config.start_index == None else config.start_index
    log.debug(f'MD starts from No.{start_index} configuration in {config.init_traj}')
    atoms = images[start_index] 
    atoms.wrap() #Wrap positions to unit cell.
    
    # Settings for atom object
    if config.rattle:
        log.debug(f'Rattle atoms with {config.rattle} eV/A^2')
        atoms.rattle(config.rattle)
    if config.fix_under:
        log.debug(f'Fix atoms under {config.fix_under} A')
        cons = FixAtoms(mask=atoms.positions[:, 2] < config.fix_under) if config.fix_under else []
        atoms.set_constraint(cons)
    
    # Set up MD calculator with plummed
    setup = open("plumed.dat", "r").read().splitlines()
    atoms.calc = Plumed(
        calc=MLcalc,
        input=setup,
        timestep=config.time_step / units.fs,  # unit change, the first line of plumed must use TIME=fs
        atoms=atoms,
        kT=config.temperature * units.kB,
        restart=config.plumed_restart,
        )
    #if config.plumed_restart:
    #    atoms.calc.istep = start_index if start_index >= 0 else len(images) + start_index 

    # Test MD calculator
    atoms.get_potential_energy()

    # Set up MD dynamics
    MaxwellBoltzmannDistribution(atoms, temperature_K=config.temperature)
    dyn = Langevin(atoms, config.time_step * units.fs, temperature_K=config.temperature, friction=config.friction)

    # Set up MD control
    md_control = MDControl(config,PE,atoms)
    dyn.attach(md_control.run, interval=config.print_step)

    # Set up MD trajectory and run MD simualtion
    traj = Trajectory('MD.traj', 'w', atoms)
    dyn.attach(traj.write, interval=config.dump_step)
    dyn.run(config.max_steps)