from __future__ import annotations

"""Mahalanobis-distance-based uncertainty estimator."""

from typing import Dict, Optional, Union

from ase import Atoms
from ase.calculators.calculator import Calculator

from curator.data import properties
from .base import BaseUncertainty
import logging
import torch
from curator.layer import FeatureCalculator


class MahalanobisUncertainty(BaseUncertainty):
    def __init__(
        self,
        high_threshold: float = 1.1,
        low_threshold: float = 0.95,
        calculator: Optional[Calculator] = None,
        dataset: Union[str, None] = None,
        max_structures: Optional[int] = 128,
        kernel: str = "local-full-g",
    ) -> None:
        self.dataset = dataset
        self.max_structures = max_structures
        self.kernel = kernel
        self.feature_calculator: Optional[FeatureCalculator] = None
        self._initialized = False
        self._high_spec = high_threshold
        self._low_spec = low_threshold
        uncertainty_keys = (properties.maha_dist, "is_outlier", "is_warning")

        super().__init__(
            uncertainty_keys=uncertainty_keys,
            calculator=calculator,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            threshold_key=properties.maha_dist,
        )

        self._logger = logging.getLogger(__name__)

    def __call__(self, atoms: Atoms) -> Dict[str, Union[float, bool, None]]:
        if not self._ensure_initialized(atoms):
            raise RuntimeError("MahalanobisUncertainty requires a calculator with FeatureCalculator output.")
        if atoms.calc and properties.maha_dist in atoms.calc.results:
            dist = atoms.calc.results[properties.maha_dist]
        else:
            calc = self._ensure_calculator(atoms)
            if calc is None:
                raise RuntimeError("MahalanobisUncertainty requires a calculator with FeatureCalculator output.")
            calc.calculate(atoms)
            if properties.maha_dist not in calc.results:
                raise RuntimeError(
                    "Mahalanobis distance was not produced by the calculator; "
                    "ensure the FeatureCalculator is initialized with compute_maha_dist=True."
                )
            dist = calc.results[properties.maha_dist]

        # Convert to scalar, handling numpy/tensor/list containers without silently
        # swallowing multi-valued outputs.
        if hasattr(dist, "numel") and callable(getattr(dist, "detach", None)):
            dist = dist.detach().cpu()
            if dist.numel() != 1:
                raise ValueError(
                    f"Mahalanobis distance must be a scalar per configuration; got shape {tuple(dist.shape)}"
                )
            dist = dist.item()
        elif hasattr(dist, "__len__") and not isinstance(dist, (str, bytes)):
            from numpy import ndarray

            if isinstance(dist, ndarray):
                if dist.size != 1:
                    raise ValueError(
                        f"Mahalanobis distance must be a scalar per configuration; got shape {dist.shape}"
                    )
                dist = float(dist.item())
            else:
                raise ValueError(
                    f"Mahalanobis distance must be a scalar per configuration; got type {type(dist)}"
                )
        else:
            dist = float(dist)

        values = {properties.maha_dist: dist}
        return self._format_output(values)

    def _ensure_initialized(self, atoms: Atoms) -> bool:
        if self._initialized:
            return True
        calc = self._ensure_calculator(atoms)
        if calc is None:
            return False
        if not hasattr(calc, "model"):
            raise RuntimeError("MahalanobisUncertainty requires a calculator with a model attribute.")
        return self._init_from_model(calc.model)

    def attach_to_model(self, model) -> None:
        if self._initialized:
            return
        if model is None:
            return
        self._init_from_model(model)

    def _init_from_model(self, model) -> bool:
        if not hasattr(model, "output_modules"):
            raise RuntimeError("MahalanobisUncertainty requires a model with output_modules.")

        for module in model.output_modules:
            if isinstance(module, FeatureCalculator):
                self.feature_calculator = module
                break

        if self.feature_calculator is None:
            self.feature_calculator = FeatureCalculator(
                dataset=self.dataset,
                compute_maha_dist=True,
                max_dataset_size=self.max_structures,
                kernel=self.kernel,
            )
            model.output_modules.append(self.feature_calculator)

        if not hasattr(self.feature_calculator, "maha_dist"):
            raise RuntimeError("FeatureCalculator must be initialized with mahalanobis statistics.")

        high_source = self.feature_calculator.maha_dist
        self.high_threshold = (
            torch.quantile(high_source, self._high_spec).item()
            if self._high_spec < 1.0
            else torch.max(high_source).item() * self._high_spec
        )
        self.low_threshold = (
            torch.quantile(high_source, self._low_spec).item()
            if self._low_spec < 1.0
            else torch.max(high_source).item() * self._low_spec
        )
        self._logger.info(
            f"Mahalanobis thresholds derived from dataset: low={self.low_threshold:.6f}, high={self.high_threshold:.6f}"
        )
        self._initialized = True
        return True
