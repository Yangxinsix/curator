from __future__ import annotations

"""Shared base class for uncertainty estimators."""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Optional, Sequence, Union

from ase import Atoms
from ase.calculators.calculator import Calculator


class BaseUncertainty(ABC):
    """Base class for uncertainty estimators."""

    def __init__(
        self,
        uncertainty_keys: Sequence[str],
        calculator: Optional[Any] = None,
        low_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
        threshold_key: Optional[str] = None,
        device: Optional[Any] = None,
    ) -> None:
        self.uncertainty_keys = tuple(uncertainty_keys)
        self.calc: Optional[Calculator] = None
        self._calc_like = calculator
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.threshold_key = threshold_key or (self.uncertainty_keys[0] if self.uncertainty_keys else None)
        self.device = device
        self.log = logging.getLogger(__name__)
        self._logged_atoms_calc = False

    @abstractmethod
    def __call__(self, atoms: Atoms) -> Dict[str, Union[float, bool]]:
        """Compute uncertainty values for ``atoms``."""

    def _ensure_calculator(self, atoms: Atoms) -> Optional[Calculator]:
        if atoms.calc is not None and self.calc is not atoms.calc:
            self.calc = atoms.calc
            if not self._logged_atoms_calc:
                self.log.info("Uncertainty using atoms.calc for calculator.")
                self._logged_atoms_calc = True
            return self.calc
        if self.calc is not None:
            return self.calc
        if self._calc_like is None:
            return None
        self.calc = self._resolve_calculator(self._calc_like)
        return self.calc

    def _resolve_calculator(self, calc_like: Any) -> Calculator:
        from curator.simulate.core.calculator import MLCalculator

        if isinstance(calc_like, Calculator):
            return calc_like
        if callable(calc_like) and not isinstance(calc_like, (str, bytes)):
            made = calc_like()
            if not isinstance(made, Calculator):
                raise TypeError(f"Calculator factory must return Calculator, got {type(made)}")
            return made
        return MLCalculator(model=calc_like, device=self.device)

    def attach_to_model(self, model: Any) -> None:
        """Optional hook for model-based engines (e.g., torchsim)."""
        return

    def extract_from_outputs(self, outputs: Optional[Dict[str, Any]]) -> Dict[str, Union[float, bool, None]]:
        if not outputs:
            return {}
        values: Dict[str, float] = {}
        for key in self.uncertainty_keys:
            if key in outputs:
                values[key] = float(outputs[key])
        if not values:
            return {}
        return self._format_output(values)

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
