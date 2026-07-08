from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from curator.data import properties

try:
    from tqdm.contrib.logging import logging_redirect_tqdm
except ImportError:
    logging_redirect_tqdm = None

from .calculator import FeatureCalculator
from .common import _DEFAULT_KERNEL, FeatureSpec, KernelName, feature_spec_from_object, normalize_kernel
from .extractor import FeatureExtractor
from .store import H5Feature

logger = logging.getLogger(__name__)


class FeatureStatistics:
    """Compute and cache features over datasets and ensembles."""

    def __init__(
        self,
        models: List[nn.Module],
        dataset: torch.utils.data.Dataset,
        kernels: Optional[Sequence[Union[KernelName, FeatureSpec, dict]]] = None,
        calculators: Optional[List[FeatureCalculator]] = None,
        target_layer: str = "readout",
        num_layers: Optional[Union[int, str]] = None,
        invariants_only: bool = True,
        batch_size: int = 8,
        device: Optional[str] = None,
        store: Optional[H5Feature] = None,
        checkpoint_interval: int = 0,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.models = models
        self.dataset = dataset
        self.kernels = kernels
        self.calculators = calculators
        self.target_layer = target_layer
        self.num_layers = num_layers
        self.invariants_only = invariants_only
        self.batch_size = batch_size
        self.device = device or next(models[0].parameters()).device
        self.store = store
        self.checkpoint_interval = max(int(checkpoint_interval), 0)
        self.save_path = Path(save_path) if save_path else None
        self._ens_stats: Optional[Dict[str, torch.Tensor]] = None
        self._ensemble = None

    def get_features(
        self,
        dataset: Optional[torch.utils.data.Dataset] = None,
        normalize: bool = True,
        save: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if dataset is not None:
            self.dataset = dataset
        calculators, kernel_names = self._resolve_calculators()
        cache = self._compute(calculators, kernel_names)
        features = self._load_features(cache, kernel_names)
        if normalize:
            features = {k: self._normalize_features(v) for k, v in features.items()}
        if save and self.save_path is not None:
            torch.save(features, self.save_path)
        return features

    def get_ens_stats(self, dataset: Optional[torch.utils.data.Dataset] = None) -> Dict[str, torch.Tensor]:
        if dataset is not None:
            self.dataset = dataset
            self._ens_stats = None
        if self._ens_stats is not None:
            return self._ens_stats
        from curator.model import EnsembleModel

        if self._ensemble is None:
            self._ensemble = EnsembleModel(self.models)
        outputs: List[Dict[str, torch.Tensor]] = []
        model_dtype = next(self.models[0].parameters()).dtype
        size = len(self.dataset) if hasattr(self.dataset, "__len__") else None
        desc = f"ensemble size={size if size is not None else '?'} bs={self.batch_size}"
        log_ctx = logging_redirect_tqdm() if logging_redirect_tqdm is not None else contextlib.nullcontext()
        with log_ctx:
            logger.info("Computing ensemble stats")
            for batch in self._iter_batches(self.dataset, dtype=model_dtype, desc=desc):
                out = self._ensemble(batch)
                outputs.append({k: v.detach().cpu() for k, v in out.items()})
        self._ens_stats = {k: torch.cat([o[k] for o in outputs]) for k in outputs[0].keys()}
        return self._ens_stats

    def _resolve_calculators(self) -> Tuple[List[FeatureCalculator], List[str]]:
        if self.calculators is not None:
            if len(self.calculators) != len(self.models):
                raise ValueError("Number of calculators must match number of models.")
            kernel_names = self._kernel_names_from_calculators(self.calculators)
            return self.calculators, kernel_names

        kernel_specs = self._resolve_kernel_specs()
        calculators = []
        for model in self.models:
            extractor = FeatureExtractor(
                repr_callback=model,
                target_layer=self.target_layer,
                num_layers=self.num_layers,
                invariants_only=self.invariants_only,
            )
            calculators.append(FeatureCalculator(extractor=extractor, kernels=kernel_specs))
        kernel_names = [spec.kernel_name for spec in kernel_specs]
        return calculators, kernel_names

    def iter_kernel_features(self, kernel: KernelName):
        calculators, kernel_names = self._resolve_calculators()
        norm_kernel = normalize_kernel(str(kernel))
        if norm_kernel not in kernel_names:
            raise ValueError(f"Kernel '{norm_kernel}' is not available in FeatureStatistics.")
        size = len(self.dataset) if hasattr(self.dataset, "__len__") else None
        model_dtype = next(self.models[0].parameters()).dtype
        desc = f"stream-kernel={norm_kernel} size={size if size is not None else '?'} bs={self.batch_size}"
        log_ctx = logging_redirect_tqdm() if logging_redirect_tqdm is not None else contextlib.nullcontext()
        with log_ctx:
            logger.info("Streaming features for kernel %s", norm_kernel)
            for batch in self._iter_batches(self.dataset, dtype=model_dtype, desc=desc):
                per_model = []
                for calculator in calculators:
                    computed = calculator.compute(batch, predict=True)
                    results = self._as_dict(computed, kernel_names)
                    per_model.append(results[norm_kernel])
                yield per_model[0] if len(per_model) == 1 else torch.stack(per_model).mean(dim=0)

    def _resolve_kernel_specs(self) -> List[FeatureSpec]:
        kernels = list(self.kernels) if self.kernels is not None else [
            {
                "name": _DEFAULT_KERNEL,
                "raw_feature": normalize_kernel(_DEFAULT_KERNEL),
                "mapping": "gaussian_sketch",
                "num_features": 500,
                "layer_combine": "concat",
                "layer_norm": "none",
                "pooling": "sum",
                "sigma": 1.0,
                "seed": 0,
            }
        ]
        specs: List[FeatureSpec] = []
        seen: set[str] = set()
        for item in kernels:
            if isinstance(item, str):
                raise ValueError(
                    "String kernel names are no longer supported directly. "
                    "Pass a feature spec dict or FeatureSpec instead."
                )
            spec = feature_spec_from_object(item)
            norm = spec.kernel_name
            if norm in seen:
                raise ValueError(f"Duplicate kernel '{norm}'.")
            seen.add(norm)
            specs.append(spec)
        return specs

    @staticmethod
    def _kernel_names_from_calculators(calculators: List[FeatureCalculator]) -> List[str]:
        kernels = calculators[0].kernels
        names = [kc.kernel for kc in kernels]
        for calc in calculators[1:]:
            other = [kc.kernel for kc in calc.kernels]
            if other != names:
                raise ValueError("All calculators must share the same kernels.")
        return names

    def _iter_batches(
        self,
        dataset: Union[torch.utils.data.Dataset, DataLoader],
        dtype: Optional[torch.dtype] = None,
        desc: Optional[str] = None,
    ):
        from curator.data.utils import iter_batches

        yield from iter_batches(dataset=dataset, batch_size=self.batch_size, device=self.device, dtype=dtype, desc=desc)

    def _compute(
        self,
        calculators: List[FeatureCalculator],
        kernel_names: List[str],
    ) -> Optional[Dict[str, List[List[torch.Tensor]]]]:
        if self.store is None:
            return self._compute_cache(calculators, kernel_names)
        self._compute_store(calculators, kernel_names)
        return None

    def _compute_cache(
        self,
        calculators: List[FeatureCalculator],
        kernel_names: List[str],
    ) -> Dict[str, List[List[torch.Tensor]]]:
        cache: Dict[str, List[List[torch.Tensor]]] = {k: [list() for _ in self.models] for k in kernel_names}
        size = len(self.dataset) if hasattr(self.dataset, "__len__") else None
        for model_idx, (model, calculator) in enumerate(zip(self.models, calculators)):
            model_dtype = next(model.parameters()).dtype
            desc = f"model={model.__class__.__name__} kernels={len(kernel_names)} size={size if size is not None else '?'} bs={self.batch_size}"
            log_ctx = logging_redirect_tqdm() if logging_redirect_tqdm is not None else contextlib.nullcontext()
            with log_ctx:
                logger.info("Computing features for model %s", model.__class__.__name__)
                for b, batch in enumerate(self._iter_batches(self.dataset, dtype=model_dtype, desc=desc)):
                    computed = calculator.compute(batch, predict=True)
                    results = self._as_dict(computed, kernel_names)
                    for kernel in kernel_names:
                        cache[kernel][model_idx].append(results[kernel].cpu())
                    if self.checkpoint_interval > 0 and (b + 1) % self.checkpoint_interval == 0 and self.save_path is not None:
                        torch.save(self._stack_features(cache, kernel_names), self.save_path)
        return cache

    def _compute_store(self, calculators: List[FeatureCalculator], kernel_names: List[str]) -> None:
        size = len(self.dataset) if hasattr(self.dataset, "__len__") else None
        self.store.ensure(kernel_names, dataset_size=size)
        offsets: List[int] = [0] * len(self.models)
        image_idx = self.store.load_image_idx(kernel_names[0])
        if image_idx is not None:
            for i in range(len(self.models)):
                count = self.store.count(kernel_names[0], i)
                if count > 0:
                    offsets[i] = int(image_idx[i, count - 1].item()) + 1

        for model_idx, (model, calculator) in enumerate(zip(self.models, calculators)):
            model_dtype = next(model.parameters()).dtype
            offset = offsets[model_idx]
            global_index = 0
            desc = f"model={model.__class__.__name__} kernels={len(kernel_names)} size={size if size is not None else '?'} bs={self.batch_size}"
            log_ctx = logging_redirect_tqdm() if logging_redirect_tqdm is not None else contextlib.nullcontext()
            with log_ctx:
                if offset > 0:
                    logger.info("Resuming feature store for model %s at index %d/%s", model.__class__.__name__, offset, size)
                logger.info("Computing features for model %s", model.__class__.__name__)
                for batch in self._iter_batches(self.dataset, dtype=model_dtype, desc=desc):
                    n_structures = len(batch[properties.n_atoms])
                    batch_start = global_index
                    batch_end = global_index + n_structures
                    if batch_end <= offset:
                        global_index = batch_end
                        continue
                    computed = calculator.compute(batch, predict=True)
                    results = self._as_dict(computed, kernel_names)
                    local_idx = batch[properties.image_idx] + batch_start
                    global_idx = torch.arange(n_structures, device=local_idx.device) + batch_start
                    if batch_start < offset:
                        local_mask = local_idx >= offset
                        global_cut = offset - batch_start
                    for kernel in kernel_names:
                        feats = results[kernel]
                        if kernel.startswith("local_"):
                            idx = local_idx
                            if batch_start < offset:
                                feats = feats[local_mask]
                                idx = idx[local_mask]
                        else:
                            idx = global_idx
                            if batch_start < offset:
                                feats = feats[global_cut:]
                                idx = idx[global_cut:]
                        self.store.append(kernel, model_idx, feats, idx)
                    global_index = batch_end

    @staticmethod
    def _as_dict(computed: Union[torch.Tensor, Dict[str, torch.Tensor]], kernel_names: List[str]) -> Dict[str, torch.Tensor]:
        return computed if isinstance(computed, dict) else {kernel_names[0]: computed}

    def _load_features(
        self,
        cache: Optional[Dict[str, List[List[torch.Tensor]]]],
        kernel_names: List[str],
    ) -> Dict[str, torch.Tensor]:
        if self.store is None:
            return self._stack_features(cache, kernel_names)
        return {k: self.store.load(k) for k in kernel_names}

    @staticmethod
    def _stack_features(
        cache: Optional[Dict[str, List[List[torch.Tensor]]]],
        kernel_names: List[str],
    ) -> Dict[str, torch.Tensor]:
        if cache is None:
            return {k: torch.empty((0, 0, 0)) for k in kernel_names}
        output: Dict[str, torch.Tensor] = {}
        for kernel in kernel_names:
            per_model = [torch.cat(batches) if batches else torch.empty((0, 0)) for batches in cache[kernel]]
            output[kernel] = torch.stack(per_model)
        return output

    @staticmethod
    def _normalize_features(features: torch.Tensor) -> torch.Tensor:
        if features.numel() == 0:
            return features
        if features.dim() == 2:
            mean = torch.mean(features, dim=0)
            var = torch.var(features, dim=0)
        elif features.dim() == 3:
            mean = torch.mean(features, dim=1, keepdim=True)
            var = torch.var(features, dim=1, keepdim=True)
        else:
            raise ValueError("Features must be 2D or 3D for normalization.")
        var = torch.where(var == 0, torch.ones_like(var), var)
        return (features - mean) / var
