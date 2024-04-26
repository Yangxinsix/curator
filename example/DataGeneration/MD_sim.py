from ase.md import Langevin
from ase.io.trajectory import Trajectory
from ase.io import read
from ase import units 
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import numpy as np 

import os
import toml
# Optimize the waterbox
from gpaw import GPAW
from ase.optimize import BFGS
import argparse

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="General Active Learning", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--traj_path",
        type=str,
        help="What LiFePO4 structure to relax",
    )
    return parser.parse_args(arg_list)

def main():
    # Read the pristine structure:
    args = get_arguments()
    traj_path = args.traj_path
    atom = read(traj_path)
    print(atom)

    # Load parameters for the calculator
    params_gpaw= toml.load('params_GPAW.toml')
    # Load parameters for the MD simulation
    params_md = toml.load('params_MD.toml')

    # Get index for the sytem
    index = int(traj_path.split('.traj')[0].split('relax_')[-1])


    atom.calc = GPAW(txt=f'LiFePO4_MD_{index}.txt',symmetry='off', **params_gpaw)

    # setting directory for the saved files
    relaxsim_directory =os.getcwd()
    mdsim_name_log= f'LiFePO4_MD_{index}.log'
    mdsim_name_traj= f'LiFePO4_MD_{index}.traj'

    # Set the momenta corresponding to T=1000K
    T = params_md['temperature'] # The temperature of the MD simulation
    T0 = str(T)
    f = params_md['friction_term'] # Frictional term in the Langevin equation

    # Set up MaxwellBoltzmann distribution to distort the psoition of the molecules
    MaxwellBoltzmannDistribution(atom, temperature_K=T)


    # Set up MD with the langevin algorithm and thermostat
    md = Langevin(atom, params_md['time_step'] * units.fs,
              temperature_K=T,
              friction=f,
              logfile=relaxsim_directory + "/" + mdsim_name_log)

    # Set up the trajectory file to save the MD trajectory
    traj = Trajectory(relaxsim_directory + "/" + mdsim_name_traj,
	    "w",atom)
   
    # Set and attach logger to save MD log file
    md.attach(traj.write, interval=params_md['dump_step']) 
    
    # Start MD simulation
    md.run(params_md['max_step']) # Number of steps we want the simulation to run for 

if __name__ == "__main__":
    main()