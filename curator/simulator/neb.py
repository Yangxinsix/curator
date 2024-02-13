from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read, write, Trajectory
from ase import units
from ase.db import connect
from ase.build import make_supercell
from ase.neb import NEB, NEBTools
from ase.constraints import FixAtoms

import pandas as pd
import numpy as np
import torch
import os
import sys
from glob import glob
from pathlib import Path
import toml
import argparse
import logging

from curator.simulator.optimizer import get_optimizer
from curator.simulator import MDControl, NEB_control
import ase
from omegaconf import DictConfig
from ase.calculators.calculator import Calculator
from curator.simulator import PrintEnergy
from typing import Tuple 

def load_images(img_IS: Path, img_FS: Path):
    """Load initial and final images from dft calculation.

    Args:
        img_IS (Path): path to initial image
        img_FS (Path): path to final image
    
    Returns:
        initial (ase.Atom): initial image
        final (ase.Atom): final image
    """
    if not os.path.isfile(img_IS) and not os.path.isfile(img_FS):
        raise RuntimeError("Please provide valid initial and final data path!")
    elif not os.path.isfile(img_IS):
        raise RuntimeError("Please provide valid initial data path!")
    elif not os.path.isfile(img_FS):
        raise RuntimeError("Please provide valid final data path!")
    
    # Read inital and final image
    initial = read(img_IS)
    final = read(img_FS)
    
    return initial, final

def NEB_sim(config: DictConfig,MLcalc: Calculator,PE: PrintEnergy) -> None:
    """NEB simulation with ASE

    Args:
        config (DictConfig): configuration file
        MLcalc (Calculator): ML calculator
        PE (PrintEnergy): PrintEnergy object
    
    """

    # Setup log file
    log = PE.logger
    runHandler = logging.FileHandler('neb.log', mode='w')
    runHandler.setLevel(logging.DEBUG)
    runHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    errorHandler = logging.FileHandler('warning.log', mode='w')
    errorHandler.setLevel(logging.WARNING)
    errorHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    log.addHandler(runHandler)
    log.addHandler(errorHandler)
    log.addHandler(logging.StreamHandler())

    # Read initial and final images from dft
    initial_image = config.initial_image
    final_image = config.final_image
    initial_dft, final_dft = load_images(initial_image, final_image)

    # Optimize intial and final images from dft using ML-FF
    log.info('Optimizing images with ML')
    initial_dft.calc, final_dft.calc = MLcalc, MLcalc
    opt_initial = get_optimizer(config.optimizer,initial_dft, traj_path='NEB_init_img.traj',log_path='initial_bfgs_img.log')
    opt_final = get_optimizer(config.optimizer,final_dft, traj_path='NEB_final_img.traj',log_path='final_bfgs_img.log')
    opt_steps = config.optimizer_step
    fmax = config.fmax
    opt_initial.run(fmax=fmax, steps=opt_steps)
    opt_final.run(fmax=fmax, steps=opt_steps)

    # Loading the optimized structures for initial and final image
    traj_inital = Trajectory('NEB_init_img.traj')
    traj_final = Trajectory('NEB_final_img.traj')
    initial, final = traj_inital[-1], traj_final[-1]
    
    # Check if the systems converged:
    fmax_init = pd.read_csv('initial_bfgs_img.log',delimiter="\s+")['fmax'].values.astype(float)[-1]
    fmax_final = pd.read_csv('final_bfgs_img.log',delimiter="\s+")['fmax'].values.astype(float)[-1]
    if fmax_init >fmax or fmax_final > fmax:
        RuntimeError(f'Maximum optimization step reached. System did not converged. Check your system or increase the number of optimization steps from {opt_steps} steps')

    # Set calculator for each image
    initial.calc, final.calc = MLcalc, MLcalc
    images = [initial]
    for i in range(config.num_img):
        image = initial.copy()
        image.calc = MLcalc
        images.append(image)
    images.append(final)
    test = [i.get_potential_energy() for i in images]
    # Load NEB
    neb = NEB(images, climb=True, allow_shared_calculator=True)
    
    # Interpolate linearly the positions of all middle images
    neb.interpolate(mic=True)
    write('neb_initial.traj', images)
    
    # Optimize:
    log.info('Optimize')
    optimizer = get_optimizer(config.optimizer,neb, traj_path='NEB.traj',log_path='neb_bfgs.log')
    test = [i.get_potential_energy() for i in images]
    # Set up NEB control
    #printenergy = NEB_control(PE,images)
    #optimizer.attach(printenergy.run, interval=config.print_step)
    log.info('run NEB')
    optimizer.run(fmax=fmax)

    # Finding the last structures of NEB.
    traj_old = Trajectory('NEB.traj')
    ind = config.num_img+2 #last x images, including initial and final image
    traj_NEB = traj_old[-ind:]
    write('NEB_MD.traj',traj_NEB)

    ### Do small MD sim on a few NEB images ###
    
    # Run small MD on the some images
    if config.MD.small_MD:

        #Finding the largest energies
        if config.MD.num_MD:
            nebtools = NEBTools(traj_NEB)
            E = nebtools.get_fit().energies
            sort_max = np.argsort(np.abs(E))[::-1]
            image_ind = sort_max[:config.MD.num_MD] 
        else:
            image_ind = np.arange(0,ind)
    
        # MD on the largest energies or last images
        log.info('random indices:')
        log.info(image_ind)
        for i in image_ind:
            # Set up MD calculator and test it
            atoms = traj_NEB[i]
            atoms.calc = MLcalc
            atoms.get_potential_energy()

            # Set up MD dynamics
            MaxwellBoltzmannDistribution(atoms, temperature_K=config.MD.temperature)
            dyn = Langevin(atoms, config.MD.time_step * units.fs, temperature_K=config.MD.temperature, friction=config.MD.friction)
            
            # Set up MD control
            PE.calls = 0
            md_control = MDControl(config.MD,PE)

            # Set up MD trajectory and run MD simulation
            dyn.attach(md_control.run, interval=config.MD.print_step)
            
            log.debug(f'MD starts for Image: {i} for {config.MD.time_step} fs')
            traj = Trajectory('NEB_MD.traj', 'a',atoms)
            dyn.attach(traj.write, interval=config.MD.time_step)
            dyn.run(config.MD.total_time)
    