from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.utils.data import Subset

from curator.data import AseDataset, properties
from curator.layer._feature import (
    FeatureCalculator,
    FeatureExtractor,
    FeatureStatistics,
    H5Feature,
    normalize_kernel,
)
from curator.select.filter import Filter
from curator.select.kernel import (
    DiagonalKernelMatrix,
    FeatureKernelMatrix,
    KernelMatrix,
)
from curator.select.select import (
    deterministic_CUR,
    lcmd_greedy,
    max_det_greedy,
    max_det_greedy_local,
    max_diag,
    max_dist_greedy,
)

_DEFAULT_KERNEL = 'full-g'
_DEFAULT_N_RANDOM_FEATURES = 500


class GeneralActiveLearning:
    """Compute features, build kernel matrices, and select structures."""

    def __init__(
        self,
        models: List[nn.Module],
        kernel: str = _DEFAULT_KERNEL,
        n_random_features: int = _DEFAULT_N_RANDOM_FEATURES,
        selection: str = "max_diag",
        kernels: Optional[Sequence[Union[str, Tuple[str, int]]]] = None,
        target_layer: str = "readout_mlp",
        batch_size: int = 8,
        device: Optional[str] = None,
        store_dir: Optional[Union[str, Path]] = None,
        checkpoint_interval: int = 0,
        structure_filter: Optional[Union[Filter, Sequence[Filter]]] = None,
    ) -> None:
        self.models = models
        self.kernel = kernel
        self.selection = selection
        self.kernels = kernels
        self.n_random_features = n_random_features
        self.target_layer = target_layer
        self.batch_size = batch_size
        self.device = device or next(models[0].parameters()).device
        self.store_dir = Path(store_dir) if store_dir else None
        self.checkpoint_interval = max(int(checkpoint_interval), 0)
        self.structure_filter = structure_filter
        self._resolved_kernels = self._resolve_kernels()
        self._calculators = (
            self._build_calculators(self.models, self._resolved_kernels)
            if self._resolved_kernels
            else None
        )

    def select(
        self,
        pool_set: Union[str, Path, torch.utils.data.Dataset],
        train_set: Optional[Union[str, Path, torch.utils.data.Dataset]] = None,
        select_batch_size: int = 100,
        save_dir: Optional[Union[str, Path]] = None,
        save_json: Optional[Union[str, Path]] = None,
        normalize_features: bool = True,
    ) -> List[int]:
        kernels = self._resolved_kernels
        save_root = Path(save_dir) if save_dir else None
        if save_root is not None:
            save_root.mkdir(parents=True, exist_ok=True)

        pool_dataset = self._load_dataset(pool_set)
        train_dataset = self._load_dataset(train_set) if train_set is not None else None
        filtered_pool, pool_map = self._filter_set(pool_dataset, label="pool")

        pool_save = save_root / "pool.pt" if save_root is not None else None
        pool_stats = self._stats(
            self.models,
            filtered_pool,
            kernels,
            "pool",
            pool_save,
            self._calculators,
        )
        pool_features: Dict[str, torch.Tensor] = {}
        if kernels:
            pool_features = pool_stats.get_features(
                normalize=normalize_features,
                save=pool_save is not None,
            )

        train_stats = None
        train_features: Optional[Dict[str, torch.Tensor]] = None
        if train_dataset is not None and kernels:
            train_save = save_root / "train.pt" if save_root is not None else None
            train_stats = self._stats(
                self.models,
                train_dataset,
                kernels,
                "train",
                train_save,
                self._calculators,
            )
            train_features = train_stats.get_features(
                normalize=normalize_features,
                save=train_save is not None,
            )

        matrix, num_atoms, n_train = self._kernel_matrix(
            pool_stats=pool_stats,
            pool_features=pool_features,
            pool_set=filtered_pool,
            train_stats=train_stats,
            train_features=train_features,
            train_set=train_dataset,
        )

        if isinstance(matrix, DiagonalKernelMatrix) and self.selection != "max_diag":
            raise ValueError("Diagonal kernels only support max_diag selection.")

        if self.selection == "max_diag":
            idxs = max_diag(matrix=matrix, batch_size=select_batch_size)
        elif self.selection == "max_dist_greedy":
            idxs = max_dist_greedy(
                matrix=matrix,
                batch_size=select_batch_size,
                n_train=n_train,
            )
        elif self.selection == "max_det_greedy":
            idxs = max_det_greedy(matrix=matrix, batch_size=select_batch_size)
        elif self.selection == "max_det_greedy_local":
            if num_atoms is None:
                raise ValueError("max_det_greedy_local requires local features.")
            idxs = max_det_greedy_local(
                matrix=matrix,
                batch_size=select_batch_size,
                num_atoms=num_atoms,
            )
        elif self.selection == "lcmd_greedy":
            idxs = lcmd_greedy(
                matrix=matrix,
                batch_size=select_batch_size,
                n_train=n_train,
            )
        elif self.selection == "deterministic_CUR":
            idxs = deterministic_CUR(matrix=matrix, batch_size=select_batch_size)
        else:
            raise ValueError(f"Unknown selection method '{self.selection}'.")
        selected = idxs.cpu().tolist()
        if pool_map is not None:
            selected = [pool_map[i] for i in selected]
        if save_json is not None:
            save_path = Path(save_json)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "kernel": self.kernel,
                "selection": self.selection,
                "dataset": {
                    "pool": str(pool_set),
                    "train": str(train_set) if train_set is not None else None,
                },
                "selected": selected,
                "summary": {
                    "count": len(selected),
                    "filter_enabled": self.structure_filter is not None,
                    "pool_size_before": len(pool_dataset),
                    "pool_size_after": len(filtered_pool),
                },
            }
            with open(save_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        return selected

    def _stats(
        self,
        models: List[nn.Module],
        dataset: torch.utils.data.Dataset,
        kernels: List[Union[str, Tuple[str, int]]],
        key: str,
        save_path: Optional[Path],
        calculators: Optional[List[FeatureCalculator]] = None,
    ) -> FeatureStatistics:
        store = None
        if self.store_dir is not None:
            self.store_dir.mkdir(parents=True, exist_ok=True)
            store = H5Feature(self.store_dir / f"{key}.h5", num_models=len(models))
        return FeatureStatistics(
            models=models,
            dataset=dataset,
            kernels=kernels if kernels else None,
            calculators=calculators,
            n_random_features=self.n_random_features,
            target_layer=self.target_layer,
            batch_size=self.batch_size,
            device=self.device,
            store=store,
            checkpoint_interval=self.checkpoint_interval,
            save_path=save_path,
        )

    def _kernel_matrix(
        self,
        pool_stats: FeatureStatistics,
        pool_features: Dict[str, torch.Tensor],
        pool_set: torch.utils.data.Dataset,
        train_stats: Optional[FeatureStatistics] = None,
        train_features: Optional[Dict[str, torch.Tensor]] = None,
        train_set: Optional[torch.utils.data.Dataset] = None,
    ) -> Tuple[KernelMatrix, Optional[torch.Tensor], int]:
        kernel = normalize_kernel(self.kernel)

        if kernel in {"full-gradient", "ll-gradient", "gnn", "local_full-gradient", "local_ll-gradient", "local_gnn"}:
            if kernel not in pool_features:
                raise ValueError(f"Features for kernel '{kernel}' are not available.")
            pool_feats = pool_features[kernel]
            n_train = 0
            num_atoms = None
            feats = [pool_feats]
            if train_stats is not None and train_features is not None and train_set is not None:
                if kernel not in train_features:
                    raise ValueError(f"Features for kernel '{kernel}' are not available.")
                train_feats = train_features[kernel]
                feats.append(train_feats)
                if kernel.startswith("local_"):
                    n_train = int(self._num_atoms(train_set).sum().item())
                else:
                    n_train = len(train_set)
            if kernel.startswith("local_"):
                if train_stats is not None and train_set is not None:
                    num_atoms = torch.cat(
                        [self._num_atoms(pool_set), self._num_atoms(train_set)]
                    )
                else:
                    num_atoms = self._num_atoms(pool_set)
            return FeatureKernelMatrix(torch.cat(feats, dim=1)), num_atoms, n_train

        if kernel == "qbc-energy":
            diag = pool_stats.get_ens_stats()[properties.e_var]
            return DiagonalKernelMatrix(diag), None, 0
        if kernel == "qbc-force":
            diag = pool_stats.get_ens_stats()[properties.f_var]
            return DiagonalKernelMatrix(diag), None, 0
        if kernel == "ae-energy":
            diag = pool_stats.get_ens_stats()[properties.e_ae]
            return DiagonalKernelMatrix(diag), None, 0
        if kernel == "ae-force":
            diag = pool_stats.get_ens_stats()[properties.f_ae]
            return DiagonalKernelMatrix(diag), None, 0
        if kernel == "random":
            n_pool = len(pool_set)
            return DiagonalKernelMatrix(torch.rand(n_pool)), None, 0

        raise ValueError(f"Unknown kernel '{kernel}'.")

    def _build_calculators(
        self,
        models: List[nn.Module],
        kernels: List[Union[str, Tuple[str, int]]],
    ) -> List[FeatureCalculator]:
        specs: List[Tuple[str, int]] = []
        for item in kernels:
            if isinstance(item, str):
                specs.append((item, self.n_random_features))
            else:
                specs.append((str(item[0]), int(item[1])))
        calculators: List[FeatureCalculator] = []
        for model in models:
            extractor = FeatureExtractor(repr_callback=model, target_layer=self.target_layer)
            calculators.append(FeatureCalculator(extractor=extractor, kernels=specs))
        return calculators

    def _filter_set(
        self,
        dataset: torch.utils.data.Dataset,
        label: str,
    ) -> Tuple[torch.utils.data.Dataset, Optional[List[int]]]:
        if self.structure_filter is None:
            return dataset, None
        if isinstance(self.structure_filter, (list, tuple)):
            filters = list(self.structure_filter)
        else:
            filters = [self.structure_filter]
        filtered = dataset
        for filt in filters:
            filtered = filt.filter_dataset(filtered, label=label)
        return filtered, self._index_map(dataset, filtered)

    @staticmethod
    def _load_dataset(
        source: Union[str, Path, torch.utils.data.Dataset],
    ) -> torch.utils.data.Dataset:
        if isinstance(source, (str, Path)):
            return AseDataset(source)
        return source

    @staticmethod
    def _index_map(
        original: torch.utils.data.Dataset,
        filtered: torch.utils.data.Dataset,
    ) -> Optional[List[int]]:
        if not isinstance(filtered, Subset):
            return None
        if isinstance(original, Subset):
            if filtered.dataset is original:
                return list(filtered.indices)
            if filtered.dataset is original.dataset:
                base_indices = list(original.indices)
                pos = {idx: i for i, idx in enumerate(base_indices)}
                mapped: List[int] = []
                for idx in filtered.indices:
                    if idx not in pos:
                        raise ValueError("Filtered indices are outside the original subset.")
                    mapped.append(pos[idx])
                return mapped
        return list(filtered.indices)

    def _resolve_kernels(self) -> List[Union[str, Tuple[str, int]]]:
        if self.kernels is not None:
            resolved: List[Union[str, Tuple[str, int]]] = []
            for item in self.kernels:
                if isinstance(item, str):
                    resolved.append(normalize_kernel(item))
                else:
                    kernel, n_features = item
                    resolved.append((normalize_kernel(str(kernel)), int(n_features)))
            return resolved
        kernel = normalize_kernel(self.kernel)
        if kernel in {
            "full-gradient",
            "ll-gradient",
            "gnn",
            "local_full-gradient",
            "local_ll-gradient",
            "local_gnn",
        }:
            return [kernel]
        return []

    @staticmethod
    def _num_atoms(dataset: torch.utils.data.Dataset) -> torch.Tensor:
        counts: List[int] = []
        for i in range(len(dataset)):
            sample = dataset[i]
            n_atoms = sample[properties.n_atoms]
            if torch.is_tensor(n_atoms):
                n_atoms = int(n_atoms.item())
            else:
                n_atoms = int(n_atoms)
            counts.append(n_atoms)
        return torch.tensor(counts, dtype=torch.long)
