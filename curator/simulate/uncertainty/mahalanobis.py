from __future__ import annotations

"""Mahalanobis-distance-based uncertainty estimator."""

from typing import Dict, Optional, Union

from ase import Atoms
from ase.calculators.calculator import Calculator

from curator.data import properties
from .base import BaseUncertainty


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
        from curator.layer import FeatureCalculator
        import torch

        self.calc = calculator
        self.dataset = dataset
        self.feature_calculator: Optional[FeatureCalculator] = None
        uncertainty_keys = (properties.maha_dist, "is_outlier", "is_warning")

        super().__init__(
            uncertainty_keys=uncertainty_keys,
            calculator=calculator,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            threshold_key=properties.maha_dist,
        )

        if self.calc is not None:
            for module in self.calc.model.output_modules:
                if isinstance(module, FeatureCalculator):
                    self.feature_calculator = module
                    break

            if self.feature_calculator is None:
                # pre-define a feature calculator, then add this module into model's output modules.
                self.feature_calculator = FeatureCalculator(
                    dataset=dataset,
                    compute_maha_dist=True,
                    max_dataset_size=max_structures,
                    kernel=kernel,
                )
                self.calc.model.output_modules.append(self.feature_calculator)

            if not hasattr(self.feature_calculator, "maha_dist"):
                raise RuntimeError("FeatureCalculator must be initialized with mahalanobis statistics.")

            high_source = self.feature_calculator.maha_dist
            self.high_threshold = (
                torch.quantile(high_source, high_threshold).item()
                if high_threshold < 1.0
                else torch.max(high_source).item() * high_threshold
            )
            self.low_threshold = (
                torch.quantile(high_source, low_threshold).item()
                if low_threshold < 1.0
                else torch.max(high_source).item() * low_threshold
            )
        else:
            self.high_threshold = high_threshold
            self.low_threshold = low_threshold

    def __call__(self, atoms: Atoms) -> Dict[str, Union[float, bool, None]]:
        if atoms.calc and properties.maha_dist in atoms.calc.results:
            dist = atoms.calc.results[properties.maha_dist]
        elif self.calc is not None:
            self.calc.calculate(atoms)
            if properties.maha_dist not in self.calc.results:
                raise RuntimeError(
                    "Mahalanobis distance was not produced by the calculator; "
                    "ensure the FeatureCalculator is initialized with compute_maha_dist=True."
                )
            dist = self.calc.results[properties.maha_dist]
        else:
            raise RuntimeError("MahalanobisUncertainty requires a calculator with FeatureCalculator output.")

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
