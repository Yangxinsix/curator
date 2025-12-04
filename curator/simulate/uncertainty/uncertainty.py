from __future__ import annotations

"""Uncertainty calculation helpers."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Union

from ase import Atoms
from ase.calculators.calculator import Calculator

from curator.data import properties


class BaseUncertainty(ABC):
    """Base class for uncertainty estimators.

    Subclasses should populate :attr:`uncertainty_keys` with the metrics they
    compute, optionally attach a calculator, and define thresholds for warning
    and outlier detection. The :meth:`__call__` output always includes the
    uncertainty values along with ``is_warning`` and ``is_outlier`` flags.
    """

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


class MahalanobisUncertainty(BaseUncertainty):
    def __init__(
        self,
        high_threshold: float = 1.1,
        low_threshold: float = 0.95,
        calculator: Optional[Calculator] = None,
        dataset: Union[str, None] = None,
        max_structures: Optional[int] = 128,
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
                self.feature_calculator = FeatureCalculator(
                    repr_callback=self.calc.model,
                    dataset=dataset,
                    compute_maha_dist=True,
                    max_dataset_size=max_structures,
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
