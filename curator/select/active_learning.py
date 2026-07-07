from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Literal

import h5py
import numpy as np

import torch
from torch import nn
from torch.utils.data import Subset

from curator.data import AseDataset, properties, read_trajectory
from curator.layer._feature import (
    FeatureCalculator,
    FeatureExtractor,
    FeatureSpec,
    FeatureStatistics,
    H5Feature,
    feature_spec_from_object,
    normalize_kernel,
)
from curator.select.filter import Filter
from curator.select.kernel import (
    DiagonalKernelMatrix,
    FeatureKernelMatrix,
    KernelMatrix,
)
from curator.select.select import (
    _call_selection,
    deterministic_CUR,
    direct_birch,
    lcmd_greedy,
    max_det_greedy,
    max_det_greedy_local,
    max_diag,
    max_dist_greedy,
)

logger = logging.getLogger(__name__)

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
    "direct_birch",
]


class GeneralActiveLearning:
    """Compute features, build kernel matrices, and select structures."""

    def __init__(
        self,
        models: List[nn.Module],
        selection: SelectionName = "max_diag",
        feature_specs: Optional[Sequence[Union[FeatureSpec, dict]]] = None,
        selection_feature: Optional[str] = None,
        target_layer: str = "readout_mlp",
        batch_size: int = 8,
        device: Optional[str] = None,
        dataset_cutoff: Optional[float] = None,
        transforms: Optional[Sequence] = None,
        save_features: Optional[Union[str, Path]] = None,
        checkpoint_interval: int = 0,
        structure_filter: Optional[Union[Filter, Sequence[Filter]]] = None,
        target_domain: Optional[Union[str, int]] = None,
        selection_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.models = models
        self.selection = selection
        self.selection_kwargs = dict(selection_kwargs or {})
        self.feature_specs = [feature_spec_from_object(spec) for spec in feature_specs] if feature_specs else []
        if not self.feature_specs:
            raise ValueError("feature_specs must contain at least one feature spec.")
        available = [spec.kernel_name for spec in self.feature_specs]
        if len(set(available)) != len(available):
            raise ValueError(f"feature_specs contain duplicate output names: {available}")
        self.feature_spec_map = {spec.kernel_name: spec for spec in self.feature_specs}
        self.selection_feature = normalize_kernel(selection_feature) if selection_feature else available[0]  # type: ignore[arg-type]
        if self.selection_feature not in available:
            raise ValueError(
                f"selection_feature '{self.selection_feature}' not found in feature_specs: {available}"
            )
        self.target_layer = target_layer
        self.batch_size = batch_size
        self.device = device or next(models[0].parameters()).device
        if dataset_cutoff is None:
            representation = getattr(models[0], "representation", None)
            dataset_cutoff = getattr(representation, "cutoff", None)
        self.dataset_cutoff = dataset_cutoff
        self.transforms = list(transforms) if transforms is not None else []
        self.save_features = Path(save_features) if save_features else None
        self.checkpoint_interval = max(int(checkpoint_interval), 0)
        self.structure_filter = structure_filter
        self.target_domain = target_domain
        self._calculators = self._build_calculators(self.models, self.feature_specs)

    def select(
        self,
        pool_set: Union[str, Path],
        train_set: Optional[Union[str, Path]] = None,
        select_batch_size: int = 100,
        save_json: Optional[Union[str, Path]] = None,
        save_images: Optional[Union[bool, str, Path]] = None,
        save_selected_features: Optional[Union[bool, str, Path]] = None,
        normalize_features: bool = True,
        compute_features_only: bool = False,
    ) -> List[int]:
        if compute_features_only:
            logger.info(
                "compute_features_only=True: features will be computed/exported and structure selection is skipped."
            )

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
            self.feature_specs,
            None,
            self._calculators,
            enable_store=True,
            store_path=pool_store,
        )
        pool_features = pool_stats.get_features(
            normalize=normalize_features,
            save=False,
        )

        train_stats = None
        train_features: Optional[Dict[str, torch.Tensor]] = None
        if train_dataset is not None:
            train_store = None
            if pool_store is not None:
                path = Path(pool_store)
                train_store = path.with_name(f"{path.stem}_train{path.suffix}")
            train_stats = self._stats(
                self.models,
                train_dataset,
                self.feature_specs,
                None,
                self._calculators,
                enable_store=True,
                store_path=train_store,
            )
            train_features = train_stats.get_features(
                normalize=normalize_features,
                save=False,
            )

        if compute_features_only:
            if save_selected_features:
                logger.warning(
                    "save_selected_features is ignored when compute_features_only=True."
                )
            if save_images:
                logger.warning(
                    "save_images is ignored when compute_features_only=True."
                )
            selected: List[int] = []
            if save_json is not None:
                save_path = Path(save_json)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "kernel": self.selection_feature,
                    "selection": None,
                    "compute_features_only": True,
                    "dataset": {
                        "pool": str(pool_set),
                        "train": str(train_set) if train_set is not None else None,
                    },
                    "selected": selected,
                    "summary": {
                        "count": 0,
                        "filter_enabled": self.structure_filter is not None,
                        "pool_size_before": len(pool_dataset),
                        "pool_size_after": len(filtered_pool),
                    },
                }
                with open(save_path, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2)
            logger.info(
                "Feature-only run completed; no structures were selected."
            )
            return selected

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

        selection_methods = {
            "max_diag": max_diag,
            "max_dist_greedy": max_dist_greedy,
            "max_det_greedy": max_det_greedy,
            "max_det_greedy_local": max_det_greedy_local,
            "lcmd_greedy": lcmd_greedy,
            "deterministic_CUR": deterministic_CUR,
            "direct_birch": direct_birch,
        }
        selection_fn = selection_methods.get(self.selection)
        if selection_fn is None:
            raise ValueError(f"Unknown selection method '{self.selection}'.")
        if self.selection == "max_det_greedy_local" and num_atoms is None:
            raise ValueError("max_det_greedy_local requires local features.")
        if self.selection == "direct_birch" and num_atoms is not None:
            raise ValueError("direct_birch currently requires global structure-level features.")
        idxs = _call_selection(
            selection_fn,
            selection_kwargs=self.selection_kwargs,
            matrix=matrix,
            batch_size=select_batch_size,
            n_train=n_train,
            num_atoms=num_atoms,
        )
        selected = idxs.cpu().tolist()
        if pool_map is not None:
            selected = [pool_map[i] for i in selected]
        if save_json is not None:
            save_path = Path(save_json)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "kernel": self.selection_feature,
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
        if save_selected_features:
            if pool_store is None:
                raise RuntimeError("save_selected_features requires save_features to be enabled.")
            if isinstance(save_selected_features, (str, Path)):
                save_path = Path(save_selected_features)
            else:
                save_path = Path("selected_features.h5")
            self._save_selected_feature_store(pool_store, save_path, selected)
        return selected

    def _stats(
        self,
        models: List[nn.Module],
        dataset: torch.utils.data.Dataset,
        feature_specs: List[FeatureSpec],
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
            kernels=feature_specs if feature_specs else None,
            calculators=calculators,
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
        kernel = self.selection_feature

        if kernel in self.feature_spec_map:
            spec = self.feature_spec_map[kernel]
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
                if spec.local:
                    n_train = int(self._num_atoms(train_set).sum().item())
                else:
                    n_train = len(train_set)
            if spec.local:
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
        feature_specs: List[FeatureSpec],
    ) -> List[FeatureCalculator]:
        calculators: List[FeatureCalculator] = []
        for model in models:
            extractor = FeatureExtractor(
                repr_callback=model,
                target_layer=self.target_layer,
                target_domain=self.target_domain,
            )
            calculators.append(
                FeatureCalculator(
                    extractor=extractor,
                    kernels=feature_specs,
                    target_domain=self.target_domain,
                )
            )
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

    def _save_selected_feature_store(
        self,
        store_path: Union[str, Path],
        save_path: Path,
        selected: List[int],
    ) -> None:
        selected_list = [int(i) for i in selected]
        if not selected_list:
            return
        path = Path(store_path)
        if save_path.exists():
            save_path.unlink()
        with h5py.File(path, "r") as src:
            kernels = src.attrs.get("kernels")
            if kernels is None:
                return
            kernels = [
                k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
                for k in kernels
            ]
            num_models = int(src.attrs.get("num_models", 0))
            if num_models <= 0:
                return
            selected_store = H5Feature(
                save_path,
                num_models=num_models,
                kernels=kernels,
                dataset_size=len(selected_list),
            )
            selected_set = set(selected_list)
            remap = {idx: i for i, idx in enumerate(selected_list)}
            for kernel in kernels:
                group = src.get(f"features/{kernel}")
                if group is None or "data" not in group:
                    continue
                data = group["data"]
                image_idx = group.get("image_idx")
                for model_idx in range(num_models):
                    if image_idx is not None:
                        idx = image_idx[model_idx][:]
                        mask = np.isin(idx, list(selected_set))
                        if not mask.any():
                            continue
                        feats = data[model_idx][mask]
                        mapped = np.array([remap[int(i)] for i in idx[mask]], dtype=np.int64)
                        order = np.argsort(mapped, kind="stable")
                        feats = feats[order]
                        mapped = mapped[order]
                        selected_store.append(
                            kernel,
                            model_idx,
                            torch.from_numpy(feats),
                            torch.from_numpy(mapped),
                        )
                    else:
                        feats = data[model_idx, selected_list, :]
                        selected_store.append(kernel, model_idx, torch.from_numpy(feats))
        logger.info("Saved selected features to %s", save_path)

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
