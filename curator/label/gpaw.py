from gpaw import GPAW, KohnShamConvergenceError
from typing import Dict
import ase

class GPAW:
    """Function class to run GPAW calculation."""
    def __init__(self, parameters: Dict, check_convergence: bool=False):
        """Function class to run GPAW calculation.

        Args:
            parameters (dict): GPAW parameters
            check_convergence (bool): Check convergence or not. Defaults to True.
        """
        self.parameters = parameters
        self.calc = GPAW(**parameters)
        self.calc.set(txt='GPAW.txt')
        self.check_convergence = check_convergence
        self.count = 0
        #os.putenv('ASE_VASP_VDW', '/home/energy/modules/software/VASP/vasp-potpaw-5.4')
        #os.putenv('VASP_PP_PATH', '/home/energy/modules/software/VASP/vasp-potpaw-5.4')
        #os.putenv('ASE_VASP_COMMAND', 'mpirun vasp_std')
    def label(self,atoms: ase.Atom) -> bool:
        """Function to run GPAW calculation.
        
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
            copy('GPAW.txt', f'GPAW_{self.count}.txt')
            self.count += 1
        except KohnShamConvergenceError:
            copy('GPAW.txt', f'GPAW_{self.count}.txt')
            self.count += 1
            return False
        else:
            if self.check_convergence:

                return self.convergence_criteria()
            return True
        
    def convergence_criteria(self) -> bool:
        """Function to check convergence.

        Returns:
            bool: True if converged, False if not converged
        """
        return True