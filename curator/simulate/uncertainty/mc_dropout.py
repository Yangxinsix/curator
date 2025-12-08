from __future__ import annotations

"""MC Dropout uncertainty estimator."""

from typing import Dict, Optional, Union

from ase import Atoms
from ase.calculators.calculator import Calculator

from curator.data import properties
from .base import BaseUncertainty


class MCDropoutUncertainty(BaseUncertainty):
    def __init__(
        self,
        predictor,
        n_samples: int = 20,
        key: str = properties.f_sd,
        calculator: Optional[Calculator] = None,
        low_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
    ) -> None:
        self.predictor = predictor
        self.n = int(n_samples)
        uncertainty_keys = (key, properties.f_var)
        super().__init__(
            uncertainty_keys=uncertainty_keys,
            calculator=calculator,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            threshold_key=key,
        )

    def __call__(self, atoms: Atoms) -> Dict[str, Union[float, bool, None]]:
        import numpy as np

        samples = [self.predictor(atoms) for _ in range(self.n)]
        arr = np.asarray(samples, dtype=float)
        values = {
            self.uncertainty_keys[0]: float(arr.std()),
            self.uncertainty_keys[1]: float(arr.var()),
        }
        return self._format_output(values)
