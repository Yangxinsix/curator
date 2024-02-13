import logging
import sys
from typing import Any, Dict
from omegaconf import DictConfig
import numpy as np
import ase
from ase import units
from ase.io import Trajectory
from .uncertainty import GetUncertainty

class PrintEnergy:  # store a reference to atoms in the definition.
    """ Main class to print the physical quantities in the different simulations."""
    def __init__(self,config: DictConfig,logger: logging.Logger) -> None:
        """ Class to setup uncertainty method and print physical quantities in a simualtion.
        
        Args:
            config (DictConfig): configuration file
            logger (logging.Logger): logger object

        """
        self.calls = 0
        self.uncertain_calls = 0
        self.collect_traj = config.uncertain_traj
        if self.collect_traj:
            self.collect_traj = Trajectory(config.uncertain_traj, 'a')
        self.get_uncertainty = GetUncertainty(method=config.method,threshold=config.threshold)
        self.threshold_value = config.threshold_value
        self.threshold_maxvalue = config.threshold_maxvalue
        self.logger = logger
        
    def __call__(self, atoms: ase.Atom,print_step: int=1) -> Dict:
        """Class to print the potential, kinetic and total energy.
        
        Args:
            atoms (ase.Atom): Atoms object
            print_step (int, optional): print step. Defaults to 1.
            
        Returns:
            uncertainty (dict): uncertainty dictionary
        """        
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB) / atoms.get_global_number_of_atoms()
    
        uncertainty = self.get_uncertainty(atoms)

        string = "Steps={:10d} Epot={:12.3f} Ekin={:12.3f} temperature={:8.2f} ".format(
            self.calls * print_step,
            epot,
            ekin,
            temp,)
        for key, value in uncertainty.items():
            string += f"{key}={value:10.6f} "
    
        if uncertainty["threshold"] > self.threshold_value:
            self.uncertain_calls += 1
            self.logger.warning(string)
        else:
            self.logger.info(string)
        
        self.calls += 1
        
        return uncertainty

class MDControl:
    """Class to control MD simulation based on uncertainty."""
    def __init__(self,config: DictConfig,printenergy: PrintEnergy,atoms: ase.Atom) -> None:
        """ Class to control MD simulation based on uncertainty.
        
        Args:
            config (DictConfig): configuration file
            printenergy (PrintEnergy): printenergy object
            atoms (ase.Atom): Atoms object
        """
        # Get uncertainty results based on method
        self.collect_traj = printenergy.collect_traj
        self.threshold_value = printenergy.threshold_value
        self.threshold_maxvalue = printenergy.threshold_maxvalue
        self.print_step = config.print_step
        self.min_steps = config.min_steps
        self.printenergy = printenergy
        self.num_uncertain = config.num_uncertain
        self.atoms = atoms
        self.logger = printenergy.logger
        
    def run(self) -> None:
        """ Control MD simulation based on uncertainty."""
        atoms = self.atoms
        uncertainty =self.printenergy(atoms,self.print_step)
        if uncertainty['threshold'] > self.threshold_maxvalue:
            self.logger.error("Too large uncertainty!")
            if self.printenergy.calls + self.printenergy.uncertain_calls > self.min_steps:
                raise RuntimeError("Done")
            else:
                raise RuntimeError("Too large uncertainty!")
        elif uncertainty['threshold'] > self.threshold_value:
            if self.collect_traj:
                self.collect_traj.write(atoms)

            if self.printenergy.uncertain_calls > self.num_uncertain:
                self.logger.error(f"More than {self.num_uncertain} uncertain structures are collected!")
                if self.printenergy.calls + self.printenergy.uncertain_calls > self.min_steps:
                    raise RuntimeError("Done")
                else:
                    raise RuntimeError(f"More than {self.num_uncertain} uncertain structures are collected!")

class NEB_control:
    """Class to control NEB simulation based on uncertainty."""
    def __init__(self, printenergy: PrintEnergy,atoms: list) -> None:
        """ Control NEB simulation based on uncertainty.
    
        Args:
            printenergy (PrintEnergy): printenergy object
            atoms (ase.Atom): Atoms object
        """
        self.printenergy = printenergy
        self.atoms = atoms
        self.logger = printenergy.logger
        

    def run(self) -> None:
        """ Control NEB simulation based on uncertainty."""
        self.printenergy.calls = 0
        if isinstance(self.atoms, list):
            for a in self.atoms:
                uncertainty = self.printenergy(a)
        else:
            uncertainty = self.printenergy(self.atoms)
