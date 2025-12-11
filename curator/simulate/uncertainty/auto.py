from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

from ase import Atoms
from ase.calculators.calculator import Calculator

from curator.model import EnsembleModel
from curator.simulate.core.calculator import MLCalculator
from .mahalanobis import MahalanobisUncertainty
from .ensemble import EnsembleUncertainty


class AutoUncertainty:
    """
    Pick Mahalanobis for single-model runs (when dataset is provided), otherwise Ensemble uncertainty.
    - maha_kwargs: forwarded to MahalanobisUncertainty
    - ensemble_kwargs: forwarded to EnsembleUncertainty
    """

    def __init__(
        self,
        calculator: Any,
        dataset: Optional[Union[str, None]] = None,
        maha_kwargs: Optional[Dict[str, Any]] = None,
        ensemble_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.calculator = self._resolve_calculator(calculator)
        self.dataset = dataset
        self.maha_kwargs = dict(maha_kwargs or {})
        if self.dataset is not None and "dataset" not in self.maha_kwargs:
            self.maha_kwargs["dataset"] = self.dataset
        if "kernel" not in self.maha_kwargs:
            self.maha_kwargs["kernel"] = "local-full-g"

        self.ensemble_kwargs = dict(ensemble_kwargs or {})

        self._backend = self._select_backend()
        self.threshold_key = getattr(self._backend, "threshold_key", None) if self._backend else None
        self.uncertainty_keys = getattr(self._backend, "uncertainty_keys", ()) if self._backend else ()

    def _select_backend(self):
        model = getattr(self.calculator, "model", None)
        is_ensemble = isinstance(model, EnsembleModel)

        if not is_ensemble and self.dataset is not None:
            return MahalanobisUncertainty(calculator=self.calculator, **self.maha_kwargs)

        if is_ensemble:
            return EnsembleUncertainty(calculator=self.calculator, **self.ensemble_kwargs)

        return None

    def __call__(self, atoms: Atoms):
        if self._backend is None:
            return {}
        return self._backend(atoms)

    def _resolve_calculator(self, calc_like: Any) -> Calculator:
        # Already a calculator
        if isinstance(calc_like, Calculator):
            return calc_like
        # Factory returning calculator
        if callable(calc_like) and not isinstance(calc_like, (str, bytes)):
            made = calc_like()
            if not isinstance(made, Calculator):
                raise TypeError(f"Calculator factory must return Calculator, got {type(made)}")
            return made
        # Model-like (path/paths/module/list)
        return MLCalculator(model=calc_like)
