from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, Literal
import logging

import torch
from torch import nn
from torch.utils.data import Subset

from curator.data import AseDataset, properties, read_trajectory
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

logger = logging.getLogger(__name__)

_DEFAULT_KERNEL = 'full-g'
_DEFAULT_N_RANDOM_FEATURES = 500
KernelName = Literal[
    "full-g",
    "ll-g",
    "local-full-g",
    "local_full-g",
    "local-ll-g",
    "local_ll-g",
    "local-gnn",
    "full-gradient",
    "ll-gradient",
    "gnn",
    "local_full-gradient",
    "local_ll-gradient",
    "local_gnn",
]
SelectionName = Literal[
    "max_diag",
    "max_dist_greedy",
    "max_det_greedy",
    "max_det_greedy_local",
    "lcmd_greedy",
    "deterministic_CUR",
]


class GeneralActiveLearning:
    """Compute features, build kernel matrices, and select structures."""

    def __init__(
        self,
        models: List[nn.Module],
        kernel: KernelName = _DEFAULT_KERNEL,
        n_random_features: int = _DEFAULT_N_RANDOM_FEATURES,
        selection: SelectionName = "max_diag",
        kernels: Optional[Sequence[Union[KernelName, Tuple[KernelName, int]]]] = None,
        target_layer: str = "readout_mlp",
        batch_size: int = 8,
        device: Optional[str] = None,
        dataset_cutoff: Optional[float] = None,
        transforms: Optional[Sequence] = None,
        save_features: Optional[Union[str, Path]] = None,
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
        if dataset_cutoff is None:
            representation = getattr(models[0], "representation", None)
            dataset_cutoff = getattr(representation, "cutoff", None)
        self.dataset_cutoff = dataset_cutoff
        self.transforms = list(transforms) if transforms else None
        self.save_features = Path(save_features) if save_features else None
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
        pool_set: Union[str, Path],
        train_set: Optional[Union[str, Path]] = None,
        select_batch_size: int = 100,
        save_json: Optional[Union[str, Path]] = None,
        save_images: Optional[Union[bool, str, Path]] = None,
        normalize_features: bool = True,
    ) -> List[int]:
        kernels = self._merge_kernels(self._resolved_kernels, self.kernel)

        pool_atoms = self._read_trajectory(pool_set)
        pool_dataset = self._make_dataset(pool_atoms)
        train_dataset = None
        if train_set is not None:
            train_atoms = self._read_trajectory(train_set)
            train_dataset = self._make_dataset(train_atoms)
        filtered_pool, pool_map = self._filter_set(pool_dataset, label="pool")

        pool_store = self.save_features
        pool_stats = self._stats(
            self.models,
            filtered_pool,
            kernels,
            None,
            self._calculators,
            enable_store=True,
            store_path=pool_store,
        )
        pool_features: Dict[str, torch.Tensor] = {}
        if kernels:
            pool_features = pool_stats.get_features(
                normalize=normalize_features,
                save=False,
            )

        train_stats = None
        train_features: Optional[Dict[str, torch.Tensor]] = None
        if train_dataset is not None and kernels:
            train_store = None
            if pool_store is not None:
                path = Path(pool_store)
                train_store = path.with_name(f"{path.stem}_train{path.suffix}")
            train_stats = self._stats(
                self.models,
                train_dataset,
                kernels,
                None,
                self._calculators,
                enable_store=True,
                store_path=train_store,
            )
            train_features = train_stats.get_features(
                normalize=normalize_features,
                save=False,
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
        if save_images:
            if isinstance(save_images, (str, Path)):
                save_path = Path(save_images)
            else:
                save_path = Path("selected.traj")
            self._save_selected_images(pool_atoms, selected, save_path)
        return selected

    def _stats(
        self,
        models: List[nn.Module],
        dataset: torch.utils.data.Dataset,
        kernels: List[Union[str, Tuple[str, int]]],
        save_path: Optional[Path],
        calculators: Optional[List[FeatureCalculator]] = None,
        enable_store: bool = True,
        store_path: Optional[Union[str, Path]] = None,
    ) -> FeatureStatistics:
        store = None
        if enable_store and store_path is not None:
            store = H5Feature(store_path, num_models=len(models))
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
    def _read_trajectory(source: Union[str, Path]):
        if not isinstance(source, (str, Path)):
            raise TypeError("pool_set/train_set must be a path string.")
        return read_trajectory(source)

    def _make_dataset(self, atoms):
        cutoff = self.dataset_cutoff if self.dataset_cutoff is not None else 5.0
        return AseDataset(atoms, cutoff=cutoff, transforms=self.transforms)

    def _save_selected_images(
        self,
        atoms,
        indices: List[int],
        save_path: Path,
    ) -> None:
        from ase.io import Trajectory

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with Trajectory(str(save_path), "w") as traj:
            for idx in indices:
                traj.write(atoms[idx])
        logger.info("Saved %d selected images to %s", len(indices), save_path)

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
    def _merge_kernels(
        export_kernels: Optional[List[Union[str, Tuple[str, int]]]],
        selection_kernel: Optional[str],
    ) -> List[Union[str, Tuple[str, int]]]:
        feature_kernels = {
            "full-gradient",
            "ll-gradient",
            "gnn",
            "local_full-gradient",
            "local_ll-gradient",
            "local_gnn",
        }
        merged: List[Union[str, Tuple[str, int]]] = []
        if export_kernels:
            merged.extend(export_kernels)
        if selection_kernel:
            normalized = normalize_kernel(selection_kernel)
            if normalized in feature_kernels:
                merged.append(normalized)
        seen = set()
        deduped: List[Union[str, Tuple[str, int]]] = []
        for item in merged:
            raw = item[0] if isinstance(item, tuple) else item
            key = normalize_kernel(str(raw))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

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
