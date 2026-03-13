from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.utils.data import Subset

from curator.data import AseDataset, properties
from ..utils import find_layer_by_name_recursive
from .common import (
    _DEFAULT_KERNEL,
    _DEFAULT_N_RANDOM_FEATURES,
    ExtractedFeatures,
    KernelName,
    normalize_kernel,
)
from .extractor import FeatureExtractor
from .kernel import FeatureKernel
from .projector import FeatureProjector, RandomProjections

logger = logging.getLogger(__name__)


class FeatureCalculator(nn.Module):
    """Orchestrate feature extraction, aggregation, and optional distance scoring."""

    def __init__(
        self,
        extractor: Optional[FeatureExtractor] = None,
        kernels: Optional[Sequence[Union[FeatureKernel, Tuple, List]]] = None,
        kernel_calculators: Optional[List[FeatureKernel]] = None,
        output_features: bool = True,
        compute_maha_dist: bool = False,
        dataset: Optional[Union[torch.utils.data.Dataset, str, Path]] = None,
        distance_kernel: Optional[KernelName] = None,
        max_dataset_size: Optional[int] = None,
        streaming: bool = False,
        regularization: float = 1e-6,
        target_domain: Optional[Union[str, int]] = None,
    ) -> None:
        super().__init__()
        if extractor is None:
            self.extractor = FeatureExtractor(target_domain=target_domain)
        else:
            self.extractor = extractor
            if target_domain is not None and self.extractor.target_domain != target_domain:
                logger.warning(
                    "FeatureCalculator target_domain overrides extractor target_domain (%s -> %s).",
                    self.extractor.target_domain,
                    target_domain,
                )
                self.extractor.target_domain = target_domain
        if kernel_calculators is not None:
            self.kernels = list(kernel_calculators)
        else:
            self.kernels = self._build_kernels(
                kernels,
                repr_callback=self.extractor.repr_callback,
                target_layer=self.extractor.target_layer,
                target_domain=self.extractor.target_domain,
            )
        self.repr_callback: Optional[nn.Module] = None
        self.output_features = output_features
        self.model_outputs = [properties.feature] if self.output_features else []
        self.compute_maha_dist = compute_maha_dist
        self.dataset = dataset
        self._resolved_distance_kernel: Optional[KernelName] = None
        self._distance_kernel: Optional[KernelName] = None
        self.distance_kernel = distance_kernel
        self.max_dataset_size = max_dataset_size
        self.streaming = streaming
        self.regularization = regularization
        self._skip_forward = False
        if self.compute_maha_dist and properties.maha_dist not in self.model_outputs:
            self.model_outputs.append(properties.maha_dist)

    def extract(self, data: properties.Type, predict: bool = False) -> ExtractedFeatures:
        feature_data = self.extractor(data, predict=predict)
        return ExtractedFeatures(
            image_idx=data[properties.image_idx],
            feats=feature_data[properties.feature],
            grads=feature_data[properties.gradient],
            atomic_numbers=data.get(properties.atomic_numbers),
            num_atoms=data.get(properties.n_atoms),
        )

    def register_repr_callback(self, repr_callback: nn.Module) -> None:
        self.repr_callback = repr_callback
        self.extractor.attach(repr_callback)
        self.kernels = self._build_kernels(
            self.kernels,
            repr_callback=repr_callback,
            target_layer=self.extractor.target_layer,
            target_domain=self.extractor.target_domain,
        )
        self._resolved_distance_kernel = None
        if self.compute_maha_dist and self.dataset is not None:
            self.fit_distance(self.dataset, kernel=self.distance_kernel)

    def compute(self, data: properties.Type, predict: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if predict:
            self._skip_forward = True
        try:
            extracted = self.extract(data, predict=predict)
            if not self.kernels:
                raise RuntimeError("FeatureCalculator kernels are not initialized.")
            if any(not isinstance(kc, FeatureKernel) for kc in self.kernels):
                raise RuntimeError(
                    "FeatureCalculator kernels require initialized projectors; "
                    "call register_repr_callback or pass FeatureProjector."
                )
            if len(self.kernels) == 1:
                return self.kernels[0].compute(extracted)
            return {kc.kernel: kc.compute(extracted) for kc in self.kernels}
        finally:
            if predict:
                self._skip_forward = False

    def forward(self, data: properties.Type, predict: bool = False) -> properties.Type:
        if self._skip_forward:
            return data
        computed = self.compute(data, predict=predict)
        if self.output_features:
            data[properties.feature] = computed
        if self.compute_maha_dist:
            if not hasattr(self, "precision") or not hasattr(self, "feature_mean"):
                raise RuntimeError("Mahalanobis statistics are not initialized.")
            kernel = self._resolve_distance_kernel()
            feats = computed[kernel] if isinstance(computed, dict) else computed
            feats = (feats - self.feature_mean) / self.feature_std
            dist_sq = torch.einsum("bi,ij,bj->b", feats, self.precision, feats)
            distances = torch.sqrt(torch.clamp(dist_sq, min=0.0))
            if kernel.startswith("local_"):
                from curator.utils import scatter_mean

                distances = scatter_mean(distances, data[properties.image_idx], dim=0)
            data[properties.maha_dist] = distances
        return data

    def fit_distance(
        self,
        dataset: Optional[Union[torch.utils.data.Dataset, str, Path]] = None,
        kernel: Optional[KernelName] = None,
    ) -> None:
        from .distance import DistanceMetrics
        from .statistics import FeatureStatistics

        if dataset is None:
            dataset = self.dataset
        else:
            self.dataset = dataset
        if dataset is None:
            raise ValueError("Dataset is required to compute Mahalanobis statistics.")
        if self.repr_callback is None:
            raise ValueError("repr_callback must be set before computing Mahalanobis statistics.")
        if any(not isinstance(kc, FeatureKernel) for kc in self.kernels):
            raise RuntimeError(
                "FeatureCalculator kernels require initialized projectors; "
                "call register_repr_callback first."
            )
        dataset = self._resolve_dataset(dataset)
        if self.max_dataset_size is not None and hasattr(dataset, "__len__"):
            max_n = min(int(self.max_dataset_size), len(dataset))
            dataset = Subset(dataset, range(max_n))
        kernel = self._resolve_distance_kernel(kernel)
        image_idx = None
        reduction = None
        if kernel.startswith("local_"):
            image_idx = self._build_image_idx(dataset)
            reduction = "mean"
        stats = FeatureStatistics(
            models=[self.repr_callback],
            dataset=dataset,
            calculators=[self],
            batch_size=self.max_dataset_size or 1,
            device=str(next(self.repr_callback.parameters()).device),
        )
        metrics = DistanceMetrics(regularization=self.regularization)
        metrics.fit_from_stats(stats, kernel, image_idx=image_idx, reduction=reduction, streaming=self.streaming)
        device = next(self.repr_callback.parameters()).device
        self.register_buffer("feature_mean", metrics.mean.to(device))
        self.register_buffer("feature_std", metrics.std.to(device))
        self.register_buffer("precision", metrics.precision.to(device))
        if metrics.reference_distances is None:
            raise RuntimeError("DistanceMetrics did not compute reference distances.")
        self.register_buffer("maha_dist", metrics.reference_distances.to(device))
        self.distance_kernel = kernel

    @staticmethod
    def _build_image_idx(dataset: torch.utils.data.Dataset) -> torch.Tensor:
        counts: List[int] = []
        for i in range(len(dataset)):
            n_atoms = dataset[i][properties.n_atoms]
            counts.append(int(n_atoms.item()) if torch.is_tensor(n_atoms) else int(n_atoms))
        if counts:
            return torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(counts)])
        return torch.empty((0,), dtype=torch.long)

    def _resolve_distance_kernel(self, kernel: Optional[KernelName] = None) -> KernelName:
        if kernel is None and self._resolved_distance_kernel is not None:
            return self._resolved_distance_kernel
        if kernel is not None:
            kernel = normalize_kernel(str(kernel))
            logger.info("Distance kernel override: %s", kernel)
        if kernel is None:
            if self.distance_kernel is not None:
                kernel = normalize_kernel(str(self.distance_kernel))
                logger.info("Distance kernel from config: %s", kernel)
            elif self.kernels:
                first = self.kernels[0]
                if isinstance(first, FeatureKernel):
                    kernel = first.kernel
                else:
                    kernel = normalize_kernel(str(first[0]))
                logger.info("Distance kernel from first kernel: %s", kernel)
            else:
                kernel = _DEFAULT_KERNEL
                logger.info("Distance kernel fallback: %s", kernel)
        self._resolved_distance_kernel = normalize_kernel(str(kernel))
        return self._resolved_distance_kernel

    @property
    def distance_kernel(self) -> Optional[KernelName]:
        return self._distance_kernel

    @distance_kernel.setter
    def distance_kernel(self, kernel: Optional[KernelName]) -> None:
        self._distance_kernel = kernel
        self._resolved_distance_kernel = None

    def _resolve_dataset(self, dataset: Union[torch.utils.data.Dataset, str, Path]) -> torch.utils.data.Dataset:
        if isinstance(dataset, (str, Path)):
            cutoff = find_layer_by_name_recursive(self.repr_callback, "cutoff") if self.repr_callback else None
            return AseDataset(dataset, cutoff=cutoff or 5.0)
        return dataset

    @staticmethod
    def _build_kernels(
        kernels: Optional[Sequence[Union[FeatureKernel, Tuple, List]]],
        repr_callback: Optional[nn.Module],
        target_layer: str = "readout_mlp",
        target_domain: Optional[Union[str, int]] = None,
    ) -> List[Union[FeatureKernel, Tuple]]:
        if kernels is None:
            kernels = [(_DEFAULT_KERNEL, _DEFAULT_N_RANDOM_FEATURES)]
        built: List[Union[FeatureKernel, Tuple]] = []
        for item in kernels:
            if isinstance(item, FeatureKernel):
                built.append(item)
                continue
            if not isinstance(item, (tuple, list)) or len(item) not in {2, 3}:
                raise ValueError(
                    "kernels must be FeatureKernel, (kernel, projector/n_random_features), "
                    "or (kernel, projector/n_random_features, aggregator)."
                )
            kernel, projector = item[:2]
            aggregator = item[2] if len(item) == 3 else None
            if isinstance(projector, FeatureProjector):
                built.append(FeatureKernel(kernel, projector, aggregator=aggregator))
            elif isinstance(projector, int):
                if repr_callback is None:
                    built.append((kernel, projector, aggregator) if aggregator is not None else (kernel, projector))
                else:
                    module = repr_callback
                    if target_domain is not None:
                        readout = find_layer_by_name_recursive(repr_callback, "readout")
                        domain_modules = getattr(readout, "domain_modules", None)
                        if domain_modules is None:
                            raise ValueError("target_domain is set but model has no domain_modules.")
                        dom = str(target_domain)
                        if dom not in domain_modules:
                            raise ValueError(f"target_domain '{dom}' not found in model domain_modules.")
                        module = domain_modules[dom]
                    built.append(
                        FeatureKernel(
                            kernel,
                            RandomProjections(module, projector, target_layer=target_layer),
                            aggregator=aggregator,
                        )
                    )
            else:
                raise ValueError("kernels projector must be FeatureProjector or int.")
        return built
