import numpy as np
import ase
from ase.calculators.calculator import Calculator
from typing import Dict
from typing import Optional
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
        ):
        self.threshold_key = threshold_key
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.calc = calculator

    def __call__(self, atoms: ase.Atoms) -> Dict:
        return self.calculate(atoms)

    def calculate(self, atoms: ase.Atoms) -> Dict:
        if atoms.calc and properties.uncertainty in atoms.calc.results:
            self.uncertainty = atoms.calc.results[properties.uncertainty]
        else:
            self.calc.calculate(atoms)
            self.uncertainty = self.calc.results[properties.uncertainty]
        return self.uncertainty

    def check(self, low_threshold: Optional[float]=None, high_threshold: Optional[float]=None):
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