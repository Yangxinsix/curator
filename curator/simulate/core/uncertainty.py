from __future__ import annotations
from typing import Protocol, Dict, Optional, Sequence, Union
from ase import Atoms
from ase.calculators.calculator import Calculator
from curator.data import properties

# uncertainty_keys must be presented in these classes
class UncertaintyBackend(Protocol):
    def __call__(self, atoms: Atoms) -> Dict[str, float]: ...

class EnsembleUncertainty:
    def __init__(self,
                 uncertainty_keys: Sequence[str] = (properties.f_sd, properties.f_var),
                 calculator: Optional[Calculator] = None,
                 **_: object):
        self.uncertainty_keys = tuple(uncertainty_keys)
        self.calc = calculator

    def __call__(self, atoms: Atoms) -> Dict[str, float]:
        if atoms.calc and all(k in atoms.calc.results for k in self.uncertainty_keys):
            return {k: float(atoms.calc.results[k]) for k in self.uncertainty_keys}
        if self.calc is not None:
            self.calc.calculate(atoms)
            return {k: float(self.calc.results[k]) for k in self.uncertainty_keys if k in self.calc.results}
        return {}
    
class MCDropoutUncertainty:
    def __init__(self, predictor, n_samples: int = 20, key: str = properties.f_sd):
        self.predictor = predictor
        self.n = int(n_samples)
        self.key = key
    def __call__(self, atoms: Atoms) -> Dict[str, float]:
        import numpy as np
        samples = [self.predictor(atoms) for _ in range(self.n)]
        arr = np.asarray(samples, dtype=float)
        return {self.key: float(arr.std()), "var": float(arr.var())}
    
class MahalanobisUncertainty:
    def __init__(
        self,
        high_threshold: float = 1.1,
        low_threshold: float = 0.95,
        calculator: Optional[Calculator] = None,
        dataset: Union[str, None] = None,
        max_structures: Optional[int] = 128,
    ):
        from curator.layer import FeatureCalculator
        import torch
        self.calc = calculator
        self.dataset = dataset
        self.feature_calculator: Optional[FeatureCalculator] = None
        self.uncertainty_keys = (properties.maha_dist, "is_outlier", "is_warning")

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

    def __call__(self, atoms: Atoms) -> Dict[str, float]:
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

        return {
            properties.maha_dist: dist,
            "is_outlier": dist > self.high_threshold,
            "is_warning": dist > self.low_threshold,
        }
