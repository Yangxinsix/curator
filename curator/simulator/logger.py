from abc import ABC, abstractmethod
from ase import Atoms
from typing import Optional
from .uncertainty import BaseUncertainty
import logging

class BaseLogger(ABC):
    # Abstract class for logging simulation results
    def __init__(self):
        pass

    def attach(self, uncertainty: Optional[BaseUncertainty]=None):
        self.uncertainty_calc = uncertainty

class MDLogger(BaseLogger):
    # Logger for molecular dynamics simulations
    def __init__(self, logger: Optional[logging.Logger]=None, uncertainty: Optional[BaseUncertainty]=None):
        """ Class to setup uncertainty method and print physical quantities in a simualtion.
        
        Args:
            logger (logging.Logger): logger object
            uncertainty (BaseUncertainty): uncertainty calculator

        """
        self.calls = 0
        self.uncertain_calls = 0

        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.uncertainty_calc = uncertainty
        if self.uncertainty_calc is not None:
            errorHandler = logging.FileHandler('warning.log', mode='w')
            errorHandler.setLevel(logging.WARNING)
            errorHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
            self.logger.addHandler(errorHandler)
    
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

        string = "Steps={:10d} Epot={:12.3g} Ekin={:12.3g} temperature={:8.2g} ".format(
            self.calls * print_step,
            epot,
            ekin,
            temp,
        )

        if self.uncertainty_calc is not None:
            uncertainty = self.uncertainty_calc(atoms)
            for key, value in uncertainty.items():
                string += f"{key}={value:12.3g} "
            # check uncertainty
            check_uncertainty = self.uncertainty_calc.check()
            if check_uncertainty > 0:
                self.logger.warning(string)
                self.uncertain_calls += check_uncertainty
                if check_uncertainty > 1:
                    raise RuntimeError("Uncertainty is too high! Stopping simulation!")
            else:
                self.logger.info(string)