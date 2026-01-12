from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

from ase import Atoms
from ase.calculators.calculator import Calculator

from curator.model import EnsembleModel
from .mahalanobis import MahalanobisUncertainty
from .ensemble import EnsembleUncertainty
from .base import BaseUncertainty


class AutoUncertainty(BaseUncertainty):
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
        device: Optional[Any] = None,
    ) -> None:
        super().__init__(
            uncertainty_keys=(),
            calculator=calculator,
            low_threshold=None,
            high_threshold=None,
            threshold_key=None,
            device=device,
        )
        self.dataset = dataset
        self.maha_kwargs = dict(maha_kwargs or {})
        if self.dataset is not None and "dataset" not in self.maha_kwargs:
            self.maha_kwargs["dataset"] = self.dataset
        if "kernel" not in self.maha_kwargs:
            self.maha_kwargs["kernel"] = "local-full-g"

        self.ensemble_kwargs = dict(ensemble_kwargs or {})

        self._backend = None

    def _select_backend(self):
        if self.calc is None:
            return None
        model = getattr(self.calc, "model", None)
        is_ensemble = isinstance(model, EnsembleModel)

        if not is_ensemble and self.dataset is not None:
            return MahalanobisUncertainty(calculator=self.calc, **self.maha_kwargs)

        if is_ensemble:
            return EnsembleUncertainty(calculator=self.calc, **self.ensemble_kwargs)

        return None

    def attach_to_model(self, model):
        is_ensemble = isinstance(model, EnsembleModel)
        if not is_ensemble and self.dataset is not None:
            self._backend = MahalanobisUncertainty(calculator=None, **self.maha_kwargs)
        elif is_ensemble:
            self._backend = EnsembleUncertainty(calculator=None, **self.ensemble_kwargs)
        else:
            self._backend = None
            return
        if hasattr(self._backend, "attach_to_model"):
            self._backend.attach_to_model(model)
        self.threshold_key = getattr(self._backend, "threshold_key", None)
        self.uncertainty_keys = getattr(self._backend, "uncertainty_keys", ())
        self.low_threshold = getattr(self._backend, "low_threshold", None)
        self.high_threshold = getattr(self._backend, "high_threshold", None)

    def __call__(self, atoms: Atoms):
        if self._backend is None:
            calc = self._ensure_calculator(atoms)
            if calc is None:
                return {}
            self._backend = self._select_backend()
            if self._backend is None:
                return {}
            self.threshold_key = getattr(self._backend, "threshold_key", None)
            self.uncertainty_keys = getattr(self._backend, "uncertainty_keys", ())
            self.low_threshold = getattr(self._backend, "low_threshold", None)
            self.high_threshold = getattr(self._backend, "high_threshold", None)
        return self._backend(atoms)

    @property
    def calculator(self) -> Optional[Calculator]:
        return self.calc
