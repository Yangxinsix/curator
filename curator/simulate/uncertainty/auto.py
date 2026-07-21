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
        device: Optional[Any] = None,
    ) -> None:
        self.device = device
        calc_like = self._maybe_parse_list(calculator)
        self.dataset = self._normalize_dataset(dataset)
        self.maha_kwargs = dict(maha_kwargs or {})
        self.ensemble_kwargs = dict(ensemble_kwargs or {})

        self.calculator = self._resolve_calculator(calc_like)
        self._backend = self._select_backend()
        self.threshold_key = getattr(self._backend, "threshold_key", None) if self._backend else None
        self.uncertainty_keys = getattr(self._backend, "uncertainty_keys", ()) if self._backend else ()
        self.device = device

    def _select_backend(self):
        model = getattr(self.calculator, "model", None)
        is_ensemble = isinstance(model, EnsembleModel)
        has_dataset = self.dataset is not None

        if is_ensemble:
            return EnsembleUncertainty(calculator=self.calculator, **self.ensemble_kwargs)

        if has_dataset:
            self._prepare_runtime_mahalanobis(model)
            if "dataset" not in self.maha_kwargs:
                self.maha_kwargs["dataset"] = self.dataset
            if "kernel" not in self.maha_kwargs:
                self.maha_kwargs["kernel"] = "local-full-g"
            return MahalanobisUncertainty(calculator=self.calculator, **self.maha_kwargs)

        return None

    def _prepare_runtime_mahalanobis(self, model: Any) -> None:
        """Attach Mahalanobis feature scoring for raw single-model calculators.

        Deploy already prepares FeatureCalculator for exported models. Runtime MD
        often receives a plain checkpoint, so prepare the same adapter in memory
        before MahalanobisUncertainty inspects the calculator.
        """

        if model is None:
            return
        if self._model_has_mahalanobis_output(model):
            return
        try:
            from curator.layer import FeatureCalculator
            from curator.simulate.uncertainty._deploy import prepare_deploy_uncertainty
        except Exception:
            return

        for module in getattr(model, "output_modules", []):
            if isinstance(module, FeatureCalculator) and getattr(module, "compute_maha_dist", False):
                return

        maha_cfg = dict(self.maha_kwargs or {})
        spec = {
            "method": "mahalanobis",
            "dataset": self.dataset,
            "maha": {
                "kernel": maha_cfg.get("kernel", "local-full-g"),
                "max_structures": maha_cfg.get("max_structures"),
                "regularization": maha_cfg.get("regularization", 1e-6),
                "streaming": maha_cfg.get("streaming", False),
            },
        }
        prepare_deploy_uncertainty(model, spec, lammps_mliap=True)

    @staticmethod
    def _model_has_mahalanobis_output(model: Any) -> bool:
        try:
            from curator.data import properties
        except Exception:
            return False
        outputs = getattr(model, "model_outputs", ())
        if properties.maha_dist in outputs:
            return True
        for module in getattr(model, "output_modules", []):
            if properties.maha_dist in getattr(module, "model_outputs", ()):
                return True
        return False

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
        return MLCalculator(model=calc_like, device=self.device)

    def _maybe_parse_list(self, calc_like: Any):
        """Handle bracketed list strings from CLI (e.g., "[a,b]")."""
        if isinstance(calc_like, str):
            text = calc_like.strip()
            if text.startswith("[") and text.endswith("]"):
                inner = text[1:-1]
                parts = [p.strip().strip("\"'") for p in inner.split(",") if p.strip()]
                return parts
        return calc_like

    def _normalize_dataset(self, dataset: Optional[Union[str, None]]):
        """Treat empty/None/null datasets as missing to avoid accidental Maha."""
        if dataset is None:
            return None
        if isinstance(dataset, str):
            if dataset.strip().lower() in {"", "none", "null"}:
                return None
        return dataset
