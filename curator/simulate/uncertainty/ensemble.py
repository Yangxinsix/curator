from __future__ import annotations

"""Ensemble-based uncertainty estimator."""

from typing import Dict, Optional, Sequence, Union

from ase import Atoms
from ase.calculators.calculator import Calculator

from curator.data import properties
from .base import BaseUncertainty


class EnsembleUncertainty(BaseUncertainty):
    def __init__(
        self,
        uncertainty_keys: Sequence[str] = (properties.f_sd, properties.f_var),
        calculator: Optional[Calculator] = None,
        low_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
    ) -> None:
        super().__init__(
            uncertainty_keys=uncertainty_keys,
            calculator=calculator,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )

    def __call__(self, atoms: Atoms) -> Dict[str, Union[float, bool, None]]:
        values: Dict[str, float] = {}
        if atoms.calc and all(key in atoms.calc.results for key in self.uncertainty_keys):
            values = {key: float(atoms.calc.results[key]) for key in self.uncertainty_keys}
        elif self.calc is not None:
            self.calc.calculate(atoms)
            values = {
                key: float(self.calc.results[key])
                for key in self.uncertainty_keys
                if key in self.calc.results
            }

        return self._format_output(values)
