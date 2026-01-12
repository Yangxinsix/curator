import torch
from torch import nn
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union, Iterable
from curator.data import collate_atomsdata
from .select import *
from .kernel import *
from curator.data import properties
from curator.layer.utils import find_layer_by_name_recursive
try:
    from torch_scatter import scatter_add, scatter_mean, scatter_max
except ImportError:
    from curator.utils import scatter_add, scatter_mean, scatter_max
import logging
import shutil
import contextlib
import sys
import math
import hashlib
from pathlib import Path
from curator.layer._feature import FeatureExtractor, RandomProjections
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None
    logging_redirect_tqdm = None

logger = logging.getLogger(__name__)

class FeatureCache:
    def __init__(
        self,
        load_path: Optional[Path],
        save_path: Optional[Path],
        dataset_key: Optional[str] = None,
        num_sets: int = 1,
    ):
        self.load_path = load_path
        self.save_path = save_path
        self.dataset_key = dataset_key
        self.num_sets = num_sets
        self._backed_up: set[str] = set()

    @property
    def load_enabled(self) -> bool:
        return self.load_path is not None

    @property
    def save_enabled(self) -> bool:
        return self.save_path is not None

    @staticmethod
    def resolve_path(value: Optional[Union[bool, str, Path]], default_name: str) -> Optional[Path]:
        if value in (None, False):
            return None
        if value is True:
            return Path.cwd() / default_name
        return Path(value)

    @classmethod
    def from_config(
        cls,
        load_features: Optional[Union[bool, str, Path, Dict[str, Union[bool, str, Path]]]],
        save_features: Optional[Union[bool, str, Path, Dict[str, Union[bool, str, Path]]]],
        dataset_key: Optional[str] = None,
        num_sets: int = 1,
    ) -> "FeatureCache":
        load_value = load_features
        save_value = save_features
        if isinstance(load_features, dict) and dataset_key is not None:
            load_value = load_features.get(dataset_key)
        if isinstance(save_features, dict) and dataset_key is not None:
            save_value = save_features.get(dataset_key)
        load_path = cls.resolve_path(load_value, "features.pt")
        save_path = cls.resolve_path(save_value, "features.pt")
        return cls(load_path=load_path, save_path=save_path, dataset_key=dataset_key, num_sets=num_sets)

    @staticmethod
    def _with_suffix(base: Path, suffix: str) -> Path:
        if base.suffix:
            return base.with_name(f"{base.stem}_{suffix}{base.suffix}")
        return base / f"{suffix}.pt"

    def cache_path(self, kernel: str) -> Optional[Path]:
        if self.load_path is None:
            return None
        suffix = kernel
        if self.dataset_key and self.num_sets > 1:
            suffix = f"{self.dataset_key}_{kernel}"
        return self._with_suffix(self.load_path, suffix)

    def final_path(self) -> Optional[Path]:
        return self.save_path

    @staticmethod
    def _safe_torch_load(path: Path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _unique_backup_path(self, path: Path) -> Path:
        base = Path(f"{path}.bak")
        if not base.exists():
            return base
        for idx in range(1, 1000):
            candidate = Path(f"{path}.bak{idx}")
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"Too many backup files for {path}")

    def _backup_if_exists(self, path: Path, kind: str) -> None:
        key = str(path)
        if not path.exists() or key in self._backed_up:
            return
        backup = self._unique_backup_path(path)
        shutil.copy2(path, backup)
        logger.warning("%s file exists at %s; backing up to %s before overwrite.", kind, path, backup)
        self._backed_up.add(key)

    @staticmethod
    def _hash_indices(indices: Iterable[int]) -> str:
        hasher = hashlib.sha256()
        for idx in indices:
            hasher.update(int(idx).to_bytes(8, byteorder="little", signed=True))
        return hasher.hexdigest()

    def data_signature(self, dataset) -> Dict[str, object]:
        if isinstance(dataset, DataLoader):
            dataset = dataset.dataset
        if isinstance(dataset, torch.utils.data.Subset):
            return {
                "class": dataset.__class__.__name__,
                "length": len(dataset),
                "indices_hash": self._hash_indices(dataset.indices),
                "base": self.data_signature(dataset.dataset),
            }
        signature: Dict[str, object] = {"class": dataset.__class__.__name__}
        if hasattr(dataset, "__len__"):
            length = len(dataset)
            signature["length"] = length
            signature["indices_hash"] = self._hash_indices(range(length))
        return signature

    @staticmethod
    def model_signature(models: List[nn.Module]) -> List[Dict[str, object]]:
        signatures = []
        for model in models:
            rep = getattr(model, "representation", None)
            signatures.append(
                {
                    "class": model.__class__.__name__,
                    "representation": rep.__class__.__name__ if rep is not None else None,
                    "num_parameters": int(sum(p.numel() for p in model.parameters())),
                }
            )
        return signatures

    def build_metadata(self, models: List[nn.Module], dataset) -> Dict[str, object]:
        return {
            "version": 1,
            "data_signature": self.data_signature(dataset),
            "model_signature": self.model_signature(models),
        }

    @staticmethod
    def _metadata_matches(cached: Dict[str, object], expected: Dict[str, object]) -> bool:
        if not cached:
            return False
        keys = ("version", "data_signature", "model_signature")
        return all(cached.get(k) == expected.get(k) for k in keys)

    def load(
        self,
        kernel: str,
        models: List[nn.Module],
        dataset,
        random_projections: List[RandomProjections],
    ) -> Optional[List[List[torch.Tensor]]]:
        cache_path = self.cache_path(kernel)
        if cache_path is None or not cache_path.exists():
            return None
        logger.info("Loading cached features from %s", cache_path)
        cache = self._safe_torch_load(cache_path)
        if not isinstance(cache, dict):
            return None
        if not self._metadata_matches(cache.get("metadata"), self.build_metadata(models, dataset)):
            logger.warning("Cached features at %s do not match current dataset/model; recomputing.", cache_path)
            return None
        features = cache.get("features")
        if isinstance(features, torch.Tensor) and features.dim() == 3:
            features = [[features[i].cpu()] for i in range(features.shape[0])]
        if not isinstance(features, list) or len(features) != len(models):
            return None
        if any(not isinstance(batches, list) for batches in features):
            return None
        for model_batches in features:
            if any(not isinstance(t, torch.Tensor) for t in model_batches):
                return None
        features = [[t.cpu() for t in model_batches] for model_batches in features]
        state_dicts = cache.get("random_projections") or cache.get("random_projection_state_dicts")
        if isinstance(state_dicts, list) and len(state_dicts) == len(random_projections):
            try:
                for proj, state in zip(random_projections, state_dicts):
                    proj.load_state_dict(state)
            except Exception:
                return None
        logger.info("Using cached features from %s", cache_path)
        return features

    def save(
        self,
        kernel: str,
        models: List[nn.Module],
        dataset,
        random_projections: List[RandomProjections],
        features: List[List[torch.Tensor]],
    ) -> None:
        cache_path = self.cache_path(kernel)
        if cache_path is None:
            return
        self._backup_if_exists(cache_path, "Cache")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache = {
            "metadata": self.build_metadata(models, dataset),
            "features": features,
            "random_projections": [proj.state_dict() for proj in random_projections],
        }
        torch.save(cache, cache_path)

    def save_final(self, features: Dict[str, torch.Tensor]) -> None:
        save_path = self.final_path()
        if save_path is None:
            return
        self._backup_if_exists(save_path, "Final features")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, save_path)
        logger.info("Saved final features to %s", save_path)

class FeatureStatistics:
    """Generate features from trained models and datasets."""

    def __init__(
        self,
        models: List[nn.Module],
        dataset: torch.utils.data.Dataset,
        n_random_features: int=500,
        random_projections: Optional[List[RandomProjections]] = None,
        data_batch_size: int=8,
        device: Optional[str]=None,
        debug: bool=False,
        cache: Optional[FeatureCache] = None,
    ):
        self.models = models
        self.data_batch_size = data_batch_size
        self.dataset = dataset
        if random_projections is None:
            self.random_projections = [RandomProjections(model, n_random_features) for model in self.models]
        else:
            self.random_projections = random_projections
        self.device = device or next(models[0].parameters()).device
        self._features_cache: Dict[str, torch.Tensor] = {}
        self.ens_stats = None
        self.Fisher = None
        self.F_reg_inv = None
        self.debug = debug
        self.cache = cache or FeatureCache(None, None)

        self.ensemble = None
        self._kernel_handlers = {
            'full-gradient': self._full_gradient_features,
            'local_full-g': self._local_full_gradient_features,
            'll-gradient': self._ll_gradient_features,
            'local_ll-g': self._local_ll_gradient_features,
            'gnn': self._gnn_features,
            'local_gnn': self._local_gnn_features,
        }

    
    def _compute_ens_stats(self, model_inputs: Dict[str, torch.Tensor], method: str = "ensemble") -> Dict[str, torch.Tensor]:
        """Compute energy variance, forces variance, energy absolute error, and forces absolute error"""
        ens_stats = {}
        if method == "ensemble":
            result_dict = self.ensemble(model_inputs)
            if properties.uncertainty in result_dict:
                for k, v in result_dict[properties.uncertainty].items():
                    ens_stats[k] = v
            if properties.error in result_dict:
                for k, v in result_dict[properties.error].items():
                    ens_stats[k] = v
        
        return ens_stats
                
    def _compute_features(
        self,
        feature_extractor: FeatureExtractor,
        model_inputs: Dict[str, torch.Tensor],
        random_projection: RandomProjections,
        kernel: str='ll-gradient',
    ) -> torch.Tensor:
        """Dispatch feature computation to the registered kernel handlers."""

        if kernel not in self._kernel_handlers:
            raise RuntimeError(f"Unknown kernel '{kernel}'")
        return self._kernel_handlers[kernel](
            feature_extractor=feature_extractor,
            model_inputs=model_inputs,
            random_projection=random_projection,
        )

    def _project_all_layers(
        self,
        feats: List[torch.Tensor],
        grads: List[torch.Tensor],
        random_projection: RandomProjections,
        image_idx: torch.Tensor,
    ) -> torch.Tensor:
        assert random_projection.num_features != 0, "Error! Random projections must be provided!"
        atomic_g = torch.zeros((image_idx.shape[0], random_projection.num_features), device=image_idx.device)
        for feat, grad, in_proj, out_proj in zip(
            feats,
            grads,
            random_projection.in_feat_proj,
            random_projection.out_grad_proj,
        ):
            atomic_g += (feat @ in_proj) * (grad @ out_proj)
        return atomic_g

    def _aggregate_atomic_features(
        self,
        atomic_g: torch.Tensor,
        image_idx: torch.Tensor,
        reduce_to_structure: bool = True,
    ) -> torch.Tensor:
        if reduce_to_structure:
            g = scatter_add(atomic_g, image_idx, dim=0)
        else:
            g = atomic_g
        return g.cpu()

    def _layer_features(
        self,
        feat: torch.Tensor,
        grad: torch.Tensor,
        random_projection: RandomProjections,
        proj_idx: int,
    ) -> torch.Tensor:
        if random_projection.num_features != 0:
            return (feat @ random_projection.in_feat_proj[proj_idx]) * (
                grad @ random_projection.out_grad_proj[proj_idx]
            )
        return feat[:, :-1]

    def _full_gradient_features(
        self,
        feature_extractor: FeatureExtractor,
        model_inputs: Dict[str, torch.Tensor],
        random_projection: RandomProjections,
    ) -> torch.Tensor:
        image_idx = model_inputs[properties.image_idx]
        feature_data = feature_extractor(model_inputs, predict=True)
        feats, grads = feature_data[properties.feature], feature_data[properties.gradient]
        atomic_g = self._project_all_layers(feats, grads, random_projection, image_idx)
        return self._aggregate_atomic_features(atomic_g, image_idx, reduce_to_structure=True)

    def _local_full_gradient_features(
        self,
        feature_extractor: FeatureExtractor,
        model_inputs: Dict[str, torch.Tensor],
        random_projection: RandomProjections,
    ) -> torch.Tensor:
        image_idx = model_inputs[properties.image_idx]
        feature_data = feature_extractor(model_inputs, predict=True)
        feats, grads = feature_data[properties.feature], feature_data[properties.gradient]
        atomic_g = self._project_all_layers(feats, grads, random_projection, image_idx)
        return self._aggregate_atomic_features(atomic_g, image_idx, reduce_to_structure=False)

    def _ll_gradient_features(
        self,
        feature_extractor: FeatureExtractor,
        model_inputs: Dict[str, torch.Tensor],
        random_projection: RandomProjections,
    ) -> torch.Tensor:
        image_idx = model_inputs[properties.image_idx]
        feature_data = feature_extractor(model_inputs, predict=True)
        feats, grads = feature_data[properties.feature], feature_data[properties.gradient]
        atomic_g = self._layer_features(feats[-1], grads[-1], random_projection, -1)
        return self._aggregate_atomic_features(atomic_g, image_idx, reduce_to_structure=True)

    def _local_ll_gradient_features(
        self,
        feature_extractor: FeatureExtractor,
        model_inputs: Dict[str, torch.Tensor],
        random_projection: RandomProjections,
    ) -> torch.Tensor:
        image_idx = model_inputs[properties.image_idx]
        feature_data = feature_extractor(model_inputs, predict=True)
        feats, grads = feature_data[properties.feature], feature_data[properties.gradient]
        atomic_g = self._layer_features(feats[-1], grads[-1], random_projection, -1)
        return self._aggregate_atomic_features(atomic_g, image_idx, reduce_to_structure=False)

    def _gnn_features(
        self,
        feature_extractor: FeatureExtractor,
        model_inputs: Dict[str, torch.Tensor],
        random_projection: RandomProjections,
    ) -> torch.Tensor:
        image_idx = model_inputs[properties.image_idx]
        feature_data = feature_extractor(model_inputs, predict=True)
        feats, grads = feature_data[properties.feature], feature_data[properties.gradient]
        atomic_g = self._layer_features(feats[0], grads[0], random_projection, 0)
        return self._aggregate_atomic_features(atomic_g, image_idx, reduce_to_structure=True)

    def _local_gnn_features(
        self,
        feature_extractor: FeatureExtractor,
        model_inputs: Dict[str, torch.Tensor],
        random_projection: RandomProjections,
    ) -> torch.Tensor:
        image_idx = model_inputs[properties.image_idx]
        feature_data = feature_extractor(model_inputs, predict=True)
        feats, grads = feature_data[properties.feature], feature_data[properties.gradient]
        atomic_g = self._layer_features(feats[0], grads[0], random_projection, 0)
        return self._aggregate_atomic_features(atomic_g, image_idx, reduce_to_structure=False)

    def _iter_batches(self, dataset, desc: Optional[str] = None, dtype: Optional[torch.dtype] = None):
        if isinstance(dataset, DataLoader):
            loader = dataset
        else:
            device_str = str(self.device)
            pin_memory = device_str.startswith("cuda")
            loader = DataLoader(
                dataset=dataset,
                batch_size=self.data_batch_size,
                shuffle=False,
                collate_fn=collate_atomsdata,
                num_workers=0,
                pin_memory=pin_memory,
            )
        if tqdm is not None and desc is not None:
            iterator = tqdm(
                loader,
                desc=desc,
                total=len(loader),
                disable=not sys.stderr.isatty(),
            )
        else:
            iterator = loader
        for batch in iterator:
            moved = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    if dtype is not None and v.is_floating_point():
                        moved[k] = v.to(self.device, dtype=dtype)
                    else:
                        moved[k] = v.to(self.device)
            yield moved

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(features, dim=0)
        var = torch.var(features, dim=0)
        var = torch.where(var == 0, torch.ones_like(var), var)
        return (features - mean) / var
    
    def _compute_fisher(self, g: torch.Tensor) -> torch.Tensor:
        return torch.einsum('mci, mcj -> mij', g, g)

    @staticmethod
    def _get_dataset_size(dataset: Union[torch.utils.data.Dataset, DataLoader]) -> Optional[int]:
        if isinstance(dataset, DataLoader):
            base = getattr(dataset, "dataset", None)
            if base is not None and hasattr(base, "__len__"):
                return len(base)
        if hasattr(dataset, "__len__"):
            return len(dataset)
        return None

    @staticmethod
    def _get_num_batches(
        dataset: Union[torch.utils.data.Dataset, DataLoader],
        batch_size: int,
    ) -> Optional[int]:
        if isinstance(dataset, DataLoader):
            return len(dataset)
        dataset_size = FeatureStatistics._get_dataset_size(dataset)
        if dataset_size is None or batch_size <= 0:
            return None
        return int(math.ceil(dataset_size / batch_size))

    @staticmethod
    def _get_model_name(model: nn.Module) -> str:
        representation = getattr(model, "representation", None)
        if representation is not None:
            return representation.__class__.__name__
        return model.__class__.__name__
                                                                                               
    def get_features(
        self,
        dataset: Optional[torch.utils.data.Dataset]=None,
        kernel: str='full-gradient',
    ) -> torch.Tensor:
        """
        :return: Feature vector of ``shape=(n_models, n_structures, n_features)``.
        """
        if dataset == None:
            dataset = self.dataset
        else:
            self.dataset = dataset
            self._features_cache.clear()

        cache_key = kernel
        if cache_key not in self._features_cache:
            dataset_size = self._get_dataset_size(dataset)
            total_batches = self._get_num_batches(dataset, self.data_batch_size)
            cached_batches: Optional[List[List[torch.Tensor]]] = None

            if self.cache.load_enabled:
                cached_batches = self.cache.load(kernel, self.models, dataset, self.random_projections)

            if cached_batches is not None and total_batches is not None:
                counts = [len(b) for b in cached_batches]
                if any(c > total_batches for c in counts):
                    msg = "Cached features exceed expected batch count."
                    logger.info("%s Recomputing features.", msg)
                    cached_batches = None

            if cached_batches is not None and total_batches is not None:
                if all(len(b) == total_batches for b in cached_batches):
                    features = torch.stack(
                        [
                            torch.cat(batches) if batches else torch.empty((0, 0))
                            for batches in cached_batches
                        ]
                    )
                    self._features_cache[cache_key] = features
                    return features
            if self.cache.load_enabled and cached_batches is None:
                cached_batches = [list() for _ in self.models]

            global_g = []
            for model_idx, (model, random_proj) in enumerate(zip(self.models, self.random_projections)):
                feature_extractor = FeatureExtractor(model)
                model_batches: List[torch.Tensor] = []
                size_value = dataset_size if dataset_size is not None else "?"
                model_dtype = next(model.parameters()).dtype
                desc = (
                    f"model={self._get_model_name(model)} kernel={kernel} "
                    f"size={size_value} bs={self.data_batch_size} device={self.device}"
                )
                cached_model_batches = cached_batches[model_idx] if cached_batches is not None else None
                completed = len(cached_model_batches) if cached_model_batches else 0
                if total_batches is not None and completed == total_batches:
                    model_g = torch.cat(cached_model_batches) if cached_model_batches else torch.empty((0, 0))
                    global_g.append(self._normalize_features(model_g))
                    feature_extractor.unhook()
                    continue
                if total_batches is not None and completed:
                    logger.info(
                        "Found %d/%d cached batches for model %d",
                        completed,
                        total_batches,
                        model_idx,
                    )
                log_ctx = logging_redirect_tqdm() if logging_redirect_tqdm is not None else contextlib.nullcontext()
                with log_ctx:
                    for b, batch in enumerate(self._iter_batches(dataset, desc=desc, dtype=model_dtype)):
                        if b < completed:
                            continue
                        if self.debug:
                            logger.info(
                                f"Calculating features for batch {b}/{total_batches}.",
                                extra={"progress": True},
                            )
                        feats = self._compute_features(
                            feature_extractor=feature_extractor,
                            model_inputs=batch,
                            random_projection=random_proj,
                            kernel=kernel,
                        )
                        if cached_model_batches is not None:
                            cached_model_batches.append(feats.cpu())
                            if self.cache.load_enabled:
                                # Incremental cache write for resume.
                                self.cache.save(kernel, self.models, dataset, self.random_projections, cached_batches)
                        else:
                            model_batches.append(feats)
                feature_extractor.unhook()
                if cached_model_batches is not None:
                    model_g = torch.cat(cached_model_batches) if cached_model_batches else torch.empty((0, 0))
                elif model_batches:
                    model_g = torch.cat(model_batches)
                else:
                    model_g = torch.empty((0, 0))
                global_g.append(self._normalize_features(model_g))

            features = torch.stack(global_g)
            self._features_cache[cache_key] = features

        return self._features_cache[cache_key]

    def get_g(self, kernel: str='full-gradient') -> torch.Tensor:
        """Compatibility helper that returns cached features for a kernel."""

        return self.get_features(kernel=kernel)

    def get_num_atoms(
        self,
        dataset: Optional[torch.utils.data.Dataset]=None,
    ):
        if dataset == None:
            dataset = self.dataset
        else:
            self.dataset = dataset
            self._features_cache.clear()
        num_atoms = []
        # dataloader = torch.utils.data.DataLoader(
        #     dataset=dataset,
        #     batch_size=self.batch_size,
        #     collate_fn=collate_atomsdata,
        # )
        for batch in self._iter_batches(dataset):
            num_atoms.append(batch[properties.n_atoms])

        return torch.cat(num_atoms)

    def get_ens_stats(self, dataset: Optional[torch.utils.data.Dataset]=None, method="ensemble") -> Dict[str, torch.Tensor]:
        """
        :return: Dict of energy statistics
        """
        if dataset == None:
            dataset = self.dataset
        else:
            self.dataset = dataset
            self.ens_stats = None
            self._features_cache.clear()
            
        if self.ens_stats is None:
            if method == "ensemble":
                from curator.model import EnsembleModel
                if self.ensemble is None:
                    self.ensemble = EnsembleModel(self.models)
            else:
                raise NotImplementedError(f"Method {method} is not implemented.")

            # dataloader = torch.utils.data.DataLoader(
            #     dataset=dataset,
            #     batch_size=self.batch_size,
            #     collate_fn=collate_atomsdata,
            # )
            # Simply using dataset is faster?
            ens_stats = []
            dataset_size = self._get_dataset_size(dataset)
            size_value = dataset_size if dataset_size is not None else "?"
            model_dtype = next(self.models[0].parameters()).dtype
            log_ctx = logging_redirect_tqdm() if logging_redirect_tqdm is not None else contextlib.nullcontext()
            with log_ctx:
                desc = f"Ensemble size={size_value} bs={self.data_batch_size} device={self.device}"
                for i, batch in enumerate(self._iter_batches(dataset, desc=desc, dtype=model_dtype)):
                    if self.debug:
                        logger.info(
                            f"Predicting batch {i}.",
                            extra={"progress": True},
                        )
                    ens_stats.append(self._compute_ens_stats(batch, method))

            self.ens_stats = {k: torch.cat([ens[k] for ens in ens_stats]) for k in ens_stats[0].keys()}
            
        return self.ens_stats
    
    def get_fisher(self) -> torch.Tensor:
        if self.Fisher is None:
            self.Fisher = self._compute_fisher(self.get_features())
        return self.Fisher

    def get_F_inv(self) -> torch.Tensor:
        """
        :return: Regularized inverse of Fisher matrix of "shape=(n_models, n_features, n_features)".
        """
        if self.F_reg_inv is None:
            fisher = self.get_fisher()
            n_features = fisher.shape[-1]
            eye = torch.eye(n_features, device=fisher.device, dtype=fisher.dtype).unsqueeze(0)
            # empirical regularisation computed per-model to stabilise inversion
            lam = torch.linalg.trace(fisher, dim1=-2, dim2=-1) / max(n_features, 1)
            lam = lam[:, None, None]
            fisher_reg = fisher + lam * eye
            self.F_reg_inv = torch.linalg.inv(fisher_reg)
        return self.F_reg_inv


class DistanceMetrics:
    """Compute simple distance metrics from cached feature statistics."""

    def __init__(
        self,
        train_stats: FeatureStatistics,
        dataset_stats: Optional[FeatureStatistics] = None,
        regularization: float = 1e-6,
    ) -> None:
        self.train_stats = train_stats
        self.dataset_stats = dataset_stats
        self.regularization = regularization
        self._mean_cache: Dict[str, torch.Tensor] = {}
        self._precision_cache: Dict[str, torch.Tensor] = {}

    def get_mahalanobis_distance(
        self,
        stats: Optional[FeatureStatistics] = None,
        kernel: Optional[str] = None,
        local: bool = False,
        reduction: Optional[str] = None,
    ) -> torch.Tensor:
        kernel = kernel or self._default_kernel(local)
        stats = self._resolve_stats(stats)
        features = self._collapse_models(stats.get_features(kernel=kernel))
        mean = self.get_feature_mean(kernel)
        precision = self.get_feature_precision(kernel)
        diff = features - mean
        dist_sq = torch.einsum('bi,ij,bj->b', diff, precision, diff)
        distances = torch.sqrt(torch.clamp(dist_sq, min=0.0))
        return self._reduce(distances, stats, local, reduction)

    def get_euclidean_distance(
        self,
        stats: Optional[FeatureStatistics] = None,
        kernel: Optional[str] = None,
        local: bool = False,
        reduction: Optional[str] = None,
    ) -> torch.Tensor:
        kernel = kernel or self._default_kernel(local)
        stats = self._resolve_stats(stats)
        features = self._collapse_models(stats.get_features(kernel=kernel))
        mean = self.get_feature_mean(kernel)
        diff = features - mean
        distances = torch.sqrt(torch.clamp(torch.sum(diff * diff, dim=-1), min=0.0))
        return self._reduce(distances, stats, local, reduction)

    def get_cosine_distance(
        self,
        stats: Optional[FeatureStatistics] = None,
        kernel: Optional[str] = None,
        local: bool = False,
        reduction: Optional[str] = None,
    ) -> torch.Tensor:
        kernel = kernel or self._default_kernel(local)
        stats = self._resolve_stats(stats)
        features = self._collapse_models(stats.get_features(kernel=kernel))
        mean = self.get_feature_mean(kernel)
        norm_features = torch.linalg.norm(features, dim=-1)
        norm_mean = torch.linalg.norm(mean)
        similarity = torch.einsum('bi,i->b', features, mean) / (norm_features * norm_mean + 1e-12)
        distances = 1 - similarity
        return self._reduce(distances, stats, local, reduction)

    def set_dataset_stats(self, stats: FeatureStatistics) -> None:
        """Update dataset statistics without rebuilding the helper."""
        self.dataset_stats = stats

    def get_feature_mean(self, kernel: str = 'gnn') -> torch.Tensor:
        if kernel not in self._mean_cache:
            feats = self._collapse_models(self.train_stats.get_features(kernel=kernel))
            self._mean_cache[kernel] = torch.mean(feats, dim=0)
        return self._mean_cache[kernel]

    def get_feature_precision(self, kernel: str = 'gnn') -> torch.Tensor:
        if kernel not in self._precision_cache:
            feats = self._collapse_models(self.train_stats.get_features(kernel=kernel))
            mean = self.get_feature_mean(kernel)
            centered = feats - mean
            denom = max(centered.shape[0] - 1, 1)
            covariance = centered.T @ centered / denom
            eye = torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
            covariance = covariance + self.regularization * eye
            self._precision_cache[kernel] = torch.linalg.inv(covariance)
        return self._precision_cache[kernel]

    def _resolve_stats(self, stats: Optional[FeatureStatistics]) -> FeatureStatistics:
        if stats is not None:
            return stats
        if self.dataset_stats is None:
            raise ValueError("Dataset statistics are not provided.")
        return self.dataset_stats

    @staticmethod
    def _collapse_models(features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 3:
            raise ValueError("Expected features tensor with shape (n_models, n_items, n_features).")
        return features.mean(dim=0)

    @staticmethod
    def _default_kernel(local: bool) -> str:
        return 'local_gnn' if local else 'gnn'

    def _reduce(
        self,
        distances: torch.Tensor,
        stats: FeatureStatistics,
        local: bool,
        reduction: Optional[str],
    ) -> torch.Tensor:
        if not local or reduction is None:
            return distances
        if reduction not in {'mean', 'sum', 'max'}:
            raise ValueError(f"Unsupported reduction '{reduction}'.")
        image_idx = self._get_image_idx(stats)
        if reduction == 'mean':
            return scatter_mean(distances, image_idx, dim=0)
        if reduction == 'sum':
            return scatter_add(distances, image_idx, dim=0)
        max_result = scatter_max(distances, image_idx, dim=0)
        return max_result[0] if isinstance(max_result, tuple) else max_result

    @staticmethod
    def _get_image_idx(stats: FeatureStatistics) -> torch.Tensor:
        num_atoms = stats.get_num_atoms()
        device = num_atoms.device
        image_idx = torch.arange(num_atoms.shape[0], device=device)
        return torch.repeat_interleave(image_idx, num_atoms)

class GeneralActiveLearning:
    """Provides methods for selecting batches during active learning.

    :param kernel: Name of the kernel, e.g. "full-g", "ll-g", "full-F_inv", "ll-F_inv", "qbc-energy", "qbc-force".
                   "random" produces random selection and "ae-energy" and "ae-force" select by absolute errors
                   on the pool data, which is only possible if the pool data is already labeled.
    :param selection: Selection method, one of "max_dist_greedy", "deterministic_CUR", "lcmd_greedy", "max_det_greedy" or "max_diag".
    :param n_random_features: If "n_random_features = 0", do not use random projections.
                              Otherwise, use random projections of all linear-layer gradients.
    """
    def __init__(
        self,
        kernel = 'full-g',
        selection = 'max_diag',
        n_random_features = 0,
    ):
        self.kernel = kernel
        self.selection = selection
        self.n_random_features = n_random_features

    def select(
        self, 
        models: List[nn.Module], 
        datasets: Dict[str, torch.utils.data.Dataset], 
        data_batch_size: int = 8,
        select_batch_size: int = 100,
        debug: bool = False,
        load_features: Optional[Union[bool, str, Path, Dict[str, Union[str, Path, bool]]]] = None,
        save_features: Optional[Union[bool, str, Path, Dict[str, Union[str, Path, bool]]]] = None,
    ):
        """
        models: pytorch models,
        dataset: a dictionary containing pool, train, and validation dataset,
        data_batch_size: batch size for extracting features,
        select_batch_size: active learning selection batch size
        """        
        if (self.kernel == 'qbc-energy' or self.kernel == 'qbc-force' or self.kernel == 'ae-energy' or
            self.kernel == 'ae-force' or self.kernel == 'random') and self.selection != 'max_diag':
            raise RuntimeError(f'{self.kernel} kernel can only be used with max_diag selection method,'
                               f' not with {self.selection}!')

        num_sets = len(datasets)
        stats = {
            key: FeatureStatistics(
                models,
                ds,
                self.n_random_features,
                data_batch_size=data_batch_size,
                debug=debug,
                cache=FeatureCache.from_config(
                    load_features,
                    None,
                    dataset_key=key,
                    num_sets=num_sets,
                ),
            )
            for key, ds in datasets.items()
        }
        
        # pool-based selection or pool + train based selection
        if datasets.get('train'):
            matrix, num_atoms = self._get_kernel_matrix(stats['pool'], stats['train'])
            n_train = len(datasets['train'])
        else:
            matrix, num_atoms = self._get_kernel_matrix(stats['pool'])
            n_train = 0
        
        if self.selection == 'max_dist_greedy':
            idxs = max_dist_greedy(matrix=matrix, batch_size=select_batch_size, n_train=n_train)
        elif self.selection == 'max_diag':
            idxs = max_diag(matrix=matrix, batch_size=select_batch_size)
        elif self.selection == 'max_det_greedy':
            idxs = max_det_greedy(matrix=matrix, batch_size=select_batch_size)
        elif self.selection == 'lcmd_greedy':
            idxs = lcmd_greedy(matrix=matrix, batch_size=select_batch_size, n_train=n_train)
        elif self.selection == 'max_det_greedy_local':
            if num_atoms is None:
                raise RuntimeError("Local selection requires per-structure num_atoms metadata.")
            idxs = max_det_greedy_local(matrix=matrix, batch_size=select_batch_size, num_atoms=num_atoms)
        elif self.selection == False:
            idxs = torch.tensor([0])
        else:
            raise NotImplementedError(f"Unknown selection method '{self.selection}' for active learning!")
        
        final_cache = FeatureCache.from_config(None, save_features, dataset_key=None, num_sets=num_sets)
        if final_cache.save_enabled:
            features = {key: s.get_features() for key, s in stats.items()}
            # Final export of concatenated features for user inspection.
            final_cache.save_final(features)

        return idxs.cpu().tolist()

    def _get_kernel_matrix(
        self,
        pool_stats: FeatureStatistics,
        train_stats: Optional[FeatureStatistics]=None,
    ) -> Tuple[KernelMatrix, Optional[torch.Tensor]]:
        stats_list = [pool_stats] if train_stats == None else [pool_stats, train_stats]
        
        if self.kernel == 'full-g':
            return FeatureKernelMatrix(torch.cat([s.get_features(kernel='full-gradient') for s in stats_list], dim=1)), None
        elif self.kernel == 'll-g':
            return FeatureKernelMatrix(torch.cat([s.get_features(kernel='ll-gradient') for s in stats_list], dim=1)), None
        elif self.kernel == 'gnn':
            return FeatureKernelMatrix(torch.cat([s.get_features(kernel='gnn') for s in stats_list], dim=1)), None
        elif self.kernel == 'local_full-g':
            matrix = FeatureKernelMatrix(torch.cat([s.get_features(kernel='local_full-g') for s in stats_list], dim=1))
            num_atoms = torch.cat([s.get_num_atoms() for s in stats_list])
            return matrix, num_atoms
        elif self.kernel == 'local_ll-g':
            matrix = FeatureKernelMatrix(torch.cat([s.get_features(kernel='local_ll-g') for s in stats_list], dim=1))
            num_atoms = torch.cat([s.get_num_atoms() for s in stats_list])
            return matrix, num_atoms 
        elif self.kernel == 'local_gnn':
            matrix = FeatureKernelMatrix(torch.cat([s.get_features(kernel='local_gnn') for s in stats_list], dim=1))
            num_atoms = torch.cat([s.get_num_atoms() for s in stats_list])
            return matrix, num_atoms 
        elif self.kernel == 'full-F_inv':
            return FeatureCovKernelMatrix(
                torch.cat([s.get_features(kernel='full-gradient') for s in stats_list], dim=1),
                train_stats.get_F_reg_inv(),
            ), None
        elif self.kernel == 'll-F_inv':
            return FeatureCovKernelMatrix(
                torch.cat([s.get_features(kernel='ll-gradient') for s in stats_list], dim=1),
                train_stats.get_F_reg_inv(),
            ), None
        elif self.kernel == 'qbc-energy':
            return DiagonalKernelMatrix(pool_stats.get_ens_stats()['Energy-Var']), None
        elif self.kernel == 'qbc-force':
            return DiagonalKernelMatrix(pool_stats.get_ens_stats()['Forces-Var']), None
        elif self.kernel == 'ae-energy':
            return DiagonalKernelMatrix(pool_stats.get_ens_stats()['Energy-AE']), None
        elif self.kernel == 'ae-force':
            return DiagonalKernelMatrix(pool_stats.get_ens_stats()['Forces-AE']), None
        elif self.kernel == 'random':
            return DiagonalKernelMatrix(torch.rand([sum([len(s.dataset) for s in stats_list])])), None
        else:
            raise RuntimeError(f"Unknown active learning kernel {self.kernel}!")
