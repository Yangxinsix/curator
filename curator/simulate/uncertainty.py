import numpy as np
import ase
import sys
from ase.calculators.calculator import Calculator
from typing import Dict, Optional, Union, List
from abc import ABC, abstractmethod
from curator.data import properties
import torch
import logging

logger = logging.getLogger(__name__)

class BaseUncertainty(ABC):
    def __init__(self):
        pass
    
    # wrapper for calculate
    @abstractmethod
    def __call__(self, atoms: ase.Atoms):
        pass

    # get uncertainty and check
    @abstractmethod
    def calculate(self, atoms: ase.Atoms):
        pass
    
    # get uncertainty from calculator in self or atoms
    @abstractmethod
    def get_uncertainty(self, atoms: ase.Atoms):
        pass
    
    # check if uncertainty is above threshold
    @abstractmethod
    def check(self):
        pass

class EnsembleUncertainty(BaseUncertainty):
    def __init__(
            self, 
            key: List[str] = properties.f_sd,
            uncertainty_keys: List[str] = [properties.f_sd, properties.f_var],
            high_threshold: float = 0.5, 
            low_threshold: float = 0.05,
            calculator: Optional[Calculator] = None,
            save_uncertain_atoms: Optional[str] = 'warning_struct.traj',
            max_uncertain_calls: int = sys.maxsize,
        ):
        self.key = key
        self.uncertainty_keys = uncertainty_keys
        assert self.key in self.uncertainty_keys, f"Uncertainty threshold key {self.key} not in uncertainty keys {self.uncertainty_keys}!"
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.calc = calculator

        # error control
        self.uncertain_calls = 0
        self.max_uncertain_calls = max_uncertain_calls
        self.save_uncertain_atoms = save_uncertain_atoms
        if self.save_uncertain_atoms is not None:
            self.uncertain_traj = ase.io.Trajectory(self.save_uncertain_atoms, 'w')

    def __call__(self, atoms: ase.Atoms, return_check=False) -> Dict:
        return self.calculate(atoms, return_check=return_check)

    def calculate(self, atoms: ase.Atoms, return_check=False) -> Dict:
        uncertainty = self.get_uncertainty(atoms)
        
        # check uncertainty
        check = self.check()
        if check > 0:
            # save uncertain atoms
            self.uncertain_calls += 1
            if self.save_uncertain_atoms is not None:
                self.uncertain_traj.write(atoms)
            
            if check == 1:
                logger.warning(f"Uncertainty {self.key} is above low threshold ({uncertainty[self.key]} > {self.low_threshold}).")
                if self.uncertain_calls > self.max_uncertain_calls:
                    logger.warning("Max number {self.uncertainty_calc.max_uncertain_calls} of uncertain structures collected.")
                    check = 2
            elif check == 2:
                logger.warning(f"Uncertainty {self.key} is above threshold ({uncertainty[self.key]} > {self.high_threshold}). Uncertainty is too high!")
        
        if return_check:
            return check, uncertainty
        else:
            return self.uncertainty

    def get_uncertainty(self, atoms: ase.Atoms):
        if atoms.calc and self.key in atoms.calc.results:
            self.uncertainty = {k: atoms.calc.results[k] for k in self.uncertainty_keys}
        elif self.calc is not None:
            self.calc.calculate(atoms)
            self.uncertainty = {k: self.calc.results[k] for k in self.uncertainty_keys}
        else:
            # deal with cases with no uncertainty outputs
            self.uncertainty = {}
        return self.uncertainty

    def check(self, low_threshold: Optional[float]=None, high_threshold: Optional[float]=None):
        if self.key in self.uncertainty:
            if low_threshold is None:
                low_threshold = self.low_threshold
            if high_threshold is None:
                high_threshold = self.high_threshold

            if self.uncertainty[self.key] > high_threshold:
                return 2
            elif self.uncertainty[self.key] < low_threshold:
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
    def __init__(
        self,
        high_threshold: float = 1.1, 
        low_threshold: float = 0.95,
        calculator: Optional[Calculator] = None,
        dataset: Union[str, None] = None,
    ):
        from curator.layer import FeatureCalculator
        if calculator is not None:
            initialized = False
            for module in calculator.model.output_modules:
                if isinstance(module, FeatureCalculator):
                    initialized = True
                    break
            if not initialized:
                feat_calc = FeatureCalculator(dataset=dataset, compute_maha_dist=True)
                calculator.model.output_modules.append(feat_calc)

            self.high_threshold = torch.quantile(feat_calc.maha_dist, high_threshold).item() if high_threshold < 1.0 else torch.max(feat_calc.maha_dist).item() * high_threshold
            self.low_threshold = torch.quantile(feat_calc.maha_dist, low_threshold).item() if low_threshold < 1.0 else torch.max(feat_calc.maha_dist).item() * low_threshold
        else:
            self.high_threshold = high_threshold
            self.low_threshold = low_threshold

    def calculate(self, atoms: ase.Atom) -> Dict:
        # if atoms.calc and properties.maha_dist in atoms.calc.results:
        pass
 
    def check(self):
        pass