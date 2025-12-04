from __future__ import annotations

"""Shared base class for uncertainty estimators."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Union

from ase import Atoms
from ase.calculators.calculator import Calculator


class BaseUncertainty(ABC):
    """Base class for uncertainty estimators."""

    def __init__(
        self,
        uncertainty_keys: Sequence[str],
        calculator: Optional[Calculator] = None,
        low_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
        threshold_key: Optional[str] = None,
    ) -> None:
        self.uncertainty_keys = tuple(uncertainty_keys)
        self.calc = calculator
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.threshold_key = threshold_key or (self.uncertainty_keys[0] if self.uncertainty_keys else None)

    @abstractmethod
    def __call__(self, atoms: Atoms) -> Dict[str, Union[float, bool]]:
        """Compute uncertainty values for ``atoms``."""

    def _format_output(self, values: Dict[str, float]) -> Dict[str, Union[float, bool, None]]:
        """Return a result dictionary including warning and outlier flags."""

        result: Dict[str, Union[float, bool, None]] = {key: values.get(key) for key in self.uncertainty_keys}

        warn = False
        outlier = False
        if self.threshold_key and result.get(self.threshold_key) is not None:
            value = float(result[self.threshold_key])
            warn = self.low_threshold is not None and value > self.low_threshold
            outlier = self.high_threshold is not None and value > self.high_threshold
            if outlier and not warn:
                warn = True

        result["is_warning"] = warn
        result["is_outlier"] = outlier
        return result
