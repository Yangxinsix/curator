import numpy as np
import ase
import sys
from ase.calculators.calculator import Calculator
from typing import Dict, Optional, Union
from abc import ABC, abstractmethod
from curator.data import properties

class BaseUncertainty(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def calculate(self, atoms: ase.Atom):
        pass

    @abstractmethod
    def check(self):
        pass

class EnsembleUncertainty(BaseUncertainty):
    def __init__(
            self, 
            threshold_key: str = properties.f_sd, 
            high_threshold: float = 0.5, 
            low_threshold: float = 0.05,
            calculator: Optional[Calculator] = None,
            save_uncertain_atoms: Optional[str] = 'warning_struct.traj',
            max_uncertain_calls: int = sys.maxsize,
        ):
        self.threshold_key = threshold_key
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.calc = calculator

        # error control
        self.uncertain_calls = 0
        self.max_uncertain_calls = max_uncertain_calls
        self.save_uncertain_atoms = save_uncertain_atoms
        if self.save_uncertain_atoms is not None:
            self.uncertain_traj = ase.io.Trajectory(self.save_uncertain_atoms, 'w')

    def __call__(self, atoms: ase.Atoms) -> Dict:
        return self.calculate(atoms)

    def calculate(self, atoms: ase.Atoms) -> Dict:
        if atoms.calc and properties.uncertainty in atoms.calc.results:
            self.uncertainty = atoms.calc.results[properties.uncertainty]
        elif self.calc is not None:
            self.calc.calculate(atoms)
            self.uncertainty = self.calc.results[properties.uncertainty]
        else:
            # deal with cases with no uncertainty outputs
            self.uncertainty = {}
            return self.uncertainty
        
        check = self.check()
        if check > 0:
            self.uncertain_calls += 1
            if self.save_uncertain_atoms is not None:
                self.uncertain_traj.write(atoms)
            
            if check == 2:
                raise RuntimeError(f"{self.threshold_key}: {self.uncertainty[self.threshold_key]} > {self.high_threshold}! Uncertainty is too high!")
            if self.uncertain_calls > self.max_uncertain_calls:
                raise ValueError('Max number {self.max_uncertain_calls} of uncertain structures collected. Exiting.')
        return self.uncertainty

    def check(self, low_threshold: Optional[float]=None, high_threshold: Optional[float]=None):
        if self.threshold_key in self.uncertainty:
            if low_threshold is None:
                low_threshold = self.low_threshold
            if high_threshold is None:
                high_threshold = self.high_threshold

            if self.uncertainty[self.threshold_key] > high_threshold:
                return 2
            elif self.uncertainty[self.threshold_key] < low_threshold:
                return 0
            else:
                return 1
        else: return 0

class MCDropoutUncertainty(BaseUncertainty):
    def __init__(self):
        pass

    def calculate(self, atoms: ase.Atom) -> Dict:
        pass

    def check(self):
        pass

class MahalanobisUncertainty(BaseUncertainty):
    def __init__(self):
        pass

    def calculate(self, atoms: ase.Atom) -> Dict:
        pass

    def check(self):
        pass