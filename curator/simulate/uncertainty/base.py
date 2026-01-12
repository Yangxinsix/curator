from __future__ import annotations

"""Shared base class for uncertainty estimators."""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

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
        values: Dict[str, Any] = {}
        batch_len: Optional[int] = None
        for key in self.uncertainty_keys:
            if key in outputs:
                raw = outputs[key]
                arr = None
                if hasattr(raw, "detach"):
                    try:
                        raw_cpu = raw.detach().cpu()
                        if getattr(raw_cpu, "numel", lambda: 1)() > 1:
                            arr = raw_cpu.view(-1).numpy()
                        else:
                            values[key] = float(raw_cpu.item())
                            continue
                    except Exception:
                        pass
                if arr is None and isinstance(raw, (list, tuple, np.ndarray)):
                    arr = np.asarray(raw).reshape(-1)
                if arr is None:
                    values[key] = float(raw)
                    continue
                if batch_len is None:
                    batch_len = int(len(arr))
                values[key] = arr
        if not values:
            return {}
        if batch_len is None:
            return self._format_output(values)

        results = []
        for i in range(batch_len):
            row: Dict[str, float] = {}
            for key in self.uncertainty_keys:
                if key not in values:
                    continue
                val = values[key]
                if isinstance(val, np.ndarray):
                    row[key] = float(val[i])
                else:
                    row[key] = float(val)
            results.append(self._format_output(row))
        return results

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
