from ase.md.langevin import Langevin
#from ase.calculators.plumed import Plumed
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read, write, Trajectory

import numpy as np
import torch
import os
import glob
import toml
import argparse
from pathlib import Path
import logging
from ase.constraints import FixAtoms
from curator.simulator import MDControl
from omegaconf import DictConfig
from ase.calculators.calculator import Calculator
from curator.simulator import PrintEnergy

logger = logging.getLogger(__name__)

def MD(config: DictConfig, MLcalc: Calculator,PE: PrintEnergy) -> None:
    """ MD simulation with ASE

    Args:
        config (DictConfig): configuration file
        MLcalc (Calculator): ML calculator
        PE (PrintEnergy): PrintEnergy object
    
    """
    # Setup logger file 
    logger = PE.logger
    runHandler = logging.FileHandler('MD.log', mode='w')
    runHandler.setLevel(logging.DEBUG)
    runHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    errorHandler = logging.FileHandler('warning.log', mode='w')
    errorHandler.setLevel(logging.WARNING)
    errorHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    logger.addHandler(runHandler)
    logger.addHandler(errorHandler)
    logger.addHandler(logging.StreamHandler())
    
    # set up md start configuration
    if not os.path.isfile(config.init_traj):
        raise RuntimeError("Please provide valid initial data path!")
    images = read(config.init_traj, ':')
    start_index = np.random.choice(len(images)) if config.start_index == None else config.start_index
    logger.debug(f'MD starts from No.{start_index} configuration in {config.init_traj}')
    atoms = images[start_index] 
    atoms.wrap() #Wrap positions to unit cell.
    
    # Settings for atom object
    if config.rattle:
        logger.debug(f'Rattle atoms with {config.rattle} eV/A^2')
        atoms.rattle(config.rattle)
    if config.fix_under:
        logger.debug(f'Fix atoms under {config.fix_under} A')
        cons = FixAtoms(mask=atoms.positions[:, 2] < config.fix_under) if config.fix_under else []
        atoms.set_constraint(cons)
    
    # Set up MD calculator and test it 
    atoms.calc = MLcalc
    
    # Set up MD
    atoms.get_potential_energy()
    
    # Set up MD dynamics
    MaxwellBoltzmannDistribution(atoms, temperature_K=config.temperature)
    dyn = Langevin(atoms, config.time_step * units.fs, temperature_K=config.temperature, friction=config.friction)

    # Set up MD control
    md_control = MDControl(config,PE,atoms)
    dyn.attach(md_control.run, interval=config.print_step)

    # Set up MD trajectory and run MD simulation
    traj = Trajectory('MD.traj', 'w', atoms)
    dyn.attach(traj.write, interval=config.dump_step)
    dyn.run(config.max_steps)