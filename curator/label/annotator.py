import ase
from ase.io import Trajectory
from ase.calculators.calculator import Calculator, CalculationFailed
from typing import Optional
from .inspector import Inspector
import logging


class AtomsAnnotator:
    def __init__(
            self, 
            calculator: Calculator, 
            inspector: Optional[Inspector] = None,
            logger: Optional[logging.Logger] = None,
        ) -> None:
        self.calc = calculator
        self.inspector = inspector
        self.logger = logging.getLogger(__name__) if logger is None else logger
        self.count = 0

        if self.inspector is None:
            self.logger.warning("No labelling inspector is provided. The convergence of calculation will not be checked.")
        else:
            self.inspector.initialize_from_calculator(calculator)

    def annotate(self, atoms: ase.Atoms):
        atoms.set_calculator(self.calc)

        try:
            atoms.get_potential_energy()
            # post process the calculation. Boolean value must be returned
            if self.inspector is not None:
                status = self.inspector.post_process()
            else:
                status = True
            self.logger.info(f"Finish labelling {self.count}th structure.")
        except CalculationFailed:
            self.logger.warning(f"Caculation failed for {self.count}th structure.")
            status = False
        finally:
            self.count += 1
            return status
        
    def sweep(self):
        if self.inspector is not None:
            self.inspector.sweep()