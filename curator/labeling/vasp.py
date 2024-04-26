from ase.calculators.calculator import CalculationFailed
from ase.calculators.vasp import Vasp
import subprocess
from shutil import copy
import os
from typing import Dict
import ase

class VASP:
    """Function class to run VASP calculation."""
    def __init__(self, parameters: Dict, check_convergence: bool=True):
        """ Function class to run VASP calculation.
        
        Args:
            parameters (dict): VASP parameters
            check_convergence (bool): Check convergence or not. Defaults to True.
        """
        self.parameters = parameters
        self.calc = Vasp(**parameters)
        self.check_convergence = check_convergence
        self.count = 0
        #os.putenv('ASE_VASP_VDW', '/home/energy/modules/software/VASP/vasp-potpaw-5.4')
        #os.putenv('VASP_PP_PATH', '/home/energy/modules/software/VASP/vasp-potpaw-5.4')
        #os.putenv('ASE_VASP_COMMAND', 'mpirun vasp_std')
    def label(self,atoms:ase.Atom) -> bool:
        """Function to run VASP calculation.
        
        Args:
            atoms (ase.Atom): Atoms object
        
        Returns:
            bool: True if converged, False if not converged
        """
        # Set calculator
        atoms.set_calculator(self.calc)
        # Run calculation
        try:
            atoms.get_potential_energy()
            copy('OSZICAR', f'OSZICAR_{self.count}')
            os.remove('WAVECAR')
            os.remove('CHGCAR')
            self.count += 1
        except CalculationFailed:
            copy('OSZICAR', f'OSZICAR_{self.count}')
            os.remove('WAVECAR')
            os.remove('CHGCAR')
            self.count += 1
            return False
        else:
            # Check convergence
            if self.check_convergence:

                return self.convergence_criteria()
            return True
        
    def convergence_criteria(self) -> bool:
        """Function to check convergence using max number of steps in self consitent loop.

        Returns:
            bool: True if converged, False if not converged
        """
        steps = int(subprocess.getoutput('grep LOOP OUTCAR | wc -l'))
        if steps <= self.parameters['nelm']:
            return True 
        else:
            return False  