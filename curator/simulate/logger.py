from abc import ABC, abstractmethod
from ase import Atoms, units
from ase.io import Trajectory
from typing import Optional, Union
from .uncertainty import BaseUncertainty
import logging
import sys

class BaseLogger(ABC):
    # Abstract class for logging simulation results
    def __init__(self):
        pass

    def attach_uncertainty(self, uncertainty: Optional[BaseUncertainty]=None):
        self.uncertainty_calc = uncertainty
        if self.uncertainty_calc is not None:
            self.add_error_handler()

    def add_error_handler(self, filename: str='warning.log'):
        error_handler_exists = any(isinstance(handler, logging.FileHandler) and handler.baseFilename == filename 
                                   for handler in self.logger.handlers)
        if not error_handler_exists:
            errorHandler = logging.FileHandler(filename, mode='w')
            errorHandler.setLevel(logging.WARNING)
            errorHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
            self.logger.addHandler(errorHandler)

class MDLogger(BaseLogger):
    # Logger for molecular dynamics simulations
    def __init__(
            self, 
            logger: Optional[logging.Logger]=None, 
            uncertainty: Optional[BaseUncertainty]=None,
            min_calls: int = 0,          # if uncertaint_calls > min_calls and meets stop criteria, just stop the simulation, else raise errors.
        ):
        """ Class to setup uncertainty method and print physical quantities in a simualtion.
        
        Args:
            logger (logging.Logger): logger object
            uncertainty (BaseUncertainty): uncertainty calculator

        """
        self.calls = 0
        self.min_calls = min_calls

        self.logger = logger if logger is not None else logging.getLogger(__name__)
        if uncertainty is not None:
            self.attach_uncertainty(uncertainty)

    def __call__(self, atoms: Atoms, print_step: int=1):
        """ Method to log physical quantities of a simulation.
        
        Args:
            atoms (ase.Atoms): Atoms object
            print_step (int): print step

        """
        self.log(atoms, print_step)    

    def log(self, atoms: Atoms, print_step: int=1):
        """ Method to log physical quantities of a simulation.
        
        Args:
            atoms (ase.Atoms): Atoms object
            print_step (int): print step

        """
        self.calls += 1
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB) / atoms.get_global_number_of_atoms()

        string = "Steps={:10d} Epot={:12.3f} Ekin={:12.3f} temperature={:8.2f} ".format(
            self.calls * print_step,
            epot,
            ekin,
            temp,
        )

        if self.uncertainty_calc is not None:
            uncertainty = self.uncertainty_calc(atoms)
            for key, value in uncertainty.items():
                string += f"{key}={value:12.3f} "
            # check uncertainty
            check_uncertainty = self.uncertainty_calc.check()
            if check_uncertainty > 0:
                self.logger.warning(string)

                # error control
                if check_uncertainty == 2:
                    if self.calls > self.min_calls:
                        self.logger.warning(f"{self.uncertainty_calc.threshold_key}: {self.uncertainty_calc.uncertainty[self.uncertainty_calc.threshold_key]} > {self.uncertainty_calc.high_threshold}! Uncertainty is too high!")
                        sys.exit(0)
                    else:
                        raise RuntimeError(f"{self.uncertainty_calc.threshold_key}: {self.uncertainty_calc.uncertainty[self.uncertainty_calc.threshold_key]} > {self.uncertainty_calc.high_threshold}! Uncertainty is too high!")
                elif check_uncertainty == 1 and self.uncertainty_calc.uncertain_calls > self.uncertainty_calc.max_uncertain_calls:
                    if self.calls > self.min_calls:
                        self.logger.warning("Max number {self.uncertainty_calc.max_uncertain_calls} of uncertain structures collected. Exiting.")
                        sys.exit(0)
                    else:
                        raise RuntimeError("Max number {self.uncertainty_calc.max_uncertain_calls} of uncertain structures collected. Exiting.")
            else:
                self.logger.info(string)
        else:
            self.logger.info(string)