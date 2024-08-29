import abc
import subprocess, os
from shutil import copy
import logging
from typing import Optional
from ase.calculators.calculator import Calculator

class Inspector(abc.ABC):
    @abc.abstractmethod
    def post_process(self, *args, **kwargs):
        pass

class VASPInspector(Inspector):
    def __init__(self, nelm: int=500, logger: Optional[logging.Logger] = None,) -> None:
        super().__init__()
        self.nelm = nelm
        self.count = 0          # number of post-processing steps
        self.logger = logging.getLogger(__name__) if logger is None else logger
    
    def is_converged(self) -> bool:
        steps = int(subprocess.getoutput('grep LOOP OUTCAR | wc -l')) 
        return steps <= self.nelm

    def post_process(self) -> bool:
        copy('OSZICAR', f'OSZICAR_{self.count}')
        converged = self.is_converged()
        if not converged:
            self.logger.warning(f"Structure {self.count} is not converged over {self.nelm} electronic steps. Try to increase max steps.")
        self.count += 1

        return converged
    
    def sweep(self) -> None:
        self.logger.info("Sweeping files: WAVECAR, CHGCAR...")
        os.remove('WAVECAR')
        os.remove('CHGCAR')

    def initialize_from_calculator(self, calculator: Calculator):
        self.nelm = calculator.parameters.get('nelm', 500)
        self.logger.info(f"NELM is set to {self.nelm}.")
