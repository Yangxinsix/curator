from __future__ import annotations

from importlib import util
import contextlib
import logging
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Literal
from pathlib import Path

import h5py
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from curator.data import AseDataset, properties
from curator.data._uncertainty import UncertaintyModule
from .utils import find_layer_by_name_recursive
try:
    from torch_scatter import scatter_add, scatter_mean
except ImportError:
    from curator.utils import scatter_add, scatter_mean
try:
    from tqdm import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm
except ImportError:
    tqdm = None
    logging_redirect_tqdm = None

logger = logging.getLogger(__name__)


def _assign_non_module_attr(module: nn.Module, name: str, value) -> None:
    # Keep runtime callback handles out of nn.Module child registration.
    if name in getattr(module, "_modules", {}):
        del module._modules[name]
    object.__setattr__(module, name, value)

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
Reduction = Literal["mean", "sum"]


def normalize_kernel(kernel: str) -> str:
    aliases = {
        'full-g': 'full-gradient',
        'll-g': 'll-gradient',
        'local-full-g': 'local_full-gradient',
        'local_full-g': 'local_full-gradient',
        'local-ll-g': 'local_ll-gradient',
        'local_ll-g': 'local_ll-gradient',
        'local-gnn': 'local_gnn',
    }
    return aliases.get(kernel, kernel)


class FeatureExtractor(nn.Module):
    """Extract features from neural networks."""

    def __init__(
        self,
        repr_callback: Optional[Callable] = None,
        model_outputs: Optional[List[str]] = None,
        target_layer: str = 'readout_mlp',
        target_domain: Optional[Union[str, int]] = None,
    ) -> None:
        """Extract features from neural networks.

        Args:
            repr_callback: pytorch nn.Module
        """
        super().__init__()
        _assign_non_module_attr(self, "repr_callback", repr_callback)
        self._features = []
        self._grads = []
        self.hooks = []
        self.model_outputs = model_outputs if model_outputs is not None else ["feature", "gradient"]
        self.target_layer = target_layer
        self.target_domain = target_domain
        self._linear_types = self._resolve_linear_types()

        if self.repr_callback is not None:
            self.add_hooks()

    def save_feats_hook(self, _, in_feat):
        new_feat = torch.cat(
            (in_feat[0].detach().clone(), torch.ones_like(in_feat[0][:, 0:1])),
            dim=-1,
        )
        self._features.append(new_feat)

    def save_grads_hook(self, _, __, grad_output):
        self._grads.append(grad_output[0].detach().clone())

    def unhook(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def attach(self, repr_callback: nn.Module) -> None:
        if self.hooks:
            self.unhook()
        _assign_non_module_attr(self, "repr_callback", repr_callback)
        self.add_hooks()

    def detach(self) -> None:
        self.unhook()
        _assign_non_module_attr(self, "repr_callback", None)

    def register_repr_callback(self, repr_callback: nn.Module) -> None:
        self.attach(repr_callback)

    def add_hooks(self) -> None:
        if self.repr_callback is None:
            raise RuntimeError("repr_callback is not set.")
        search_root = self.repr_callback
        readout = find_layer_by_name_recursive(self.repr_callback, "readout")
        domain_modules = getattr(readout, "domain_modules", None)
        if self.target_domain is not None:
            if domain_modules is None:
                raise ValueError("target_domain is set but model has no domain_modules.")
            dom = str(self.target_domain)
            if dom not in domain_modules:
                raise ValueError(f"target_domain '{dom}' not found in model domain_modules.")
            search_root = domain_modules[dom]
        elif domain_modules is not None and len(domain_modules) > 1:
            logger.warning(
                "Multiple domains detected in readout; FeatureExtractor defaults to the first discovered domain. "
                "Set target_domain to select explicitly."
            )

        layer = find_layer_by_name_recursive(search_root, self.target_layer)
        assert layer is not None, f"Target layer {self.target_layer} is not found!"
        linear_modules = [m for m in layer.modules() if isinstance(m, self._linear_types)]
        if not linear_modules:
            logger.warning("No linear-like submodules found under target layer %s", self.target_layer)
        for child in linear_modules:
            self.hooks.append(child.register_forward_pre_hook(self.save_feats_hook))
            self.hooks.append(child.register_backward_hook(self.save_grads_hook))

    def forward(self, data: properties.Type, predict: bool = False) -> properties.Type:
        if torch.jit.is_scripting():
            return data
        # repr_callback may modify the original data in place, so we need to make a copy of the data
        new_data = data.copy()
        if predict:
            new_data = self.repr_callback(new_data)
        data[properties.feature] = self._features
        data[properties.gradient] = self._grads[::-1]
        self._reset()

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(target_layer={self.target_layer}, target_domain={self.target_domain})'

    def _reset(self) -> None:
        self._features = []
        self._grads = []

    @staticmethod
    def _resolve_linear_types() -> Sequence[type]:
        types: List[type] = [nn.Linear]
        if util.find_spec("e3nn.o3"):
            from e3nn import o3

            types.append(o3.Linear)
        try:
            from curator.layer._cuequivariance_wrapper import Linear as CueqLinear

            types.append(CueqLinear)
        except Exception:
            # If the cuequivariant Linear is unavailable, skip it.
            pass
        return tuple(types)


class FeatureProjector(nn.Module):
    """Base class for feature projection strategies."""

    def __init__(
        self,
        num_features: int,
        target_layer: str = 'readout_mlp',
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.target_layer = target_layer

    @property
    def in_feat_proj(self) -> List[torch.Tensor]:
        raise NotImplementedError

    @property
    def out_grad_proj(self) -> List[torch.Tensor]:
        raise NotImplementedError


class RandomProjections(FeatureProjector):
    """Random projection module with Gaussian distributions, storing projection matrices as buffers."""

    def __init__(
        self,
        module: nn.Module,
        num_features: int,
        dtype: Optional[torch.dtype] = None,
        target_layer: str = 'readout_mlp',
    ) -> None:
        super().__init__(num_features=num_features, target_layer=target_layer)

        self.in_feat_proj_buffers = []
        self.out_grad_proj_buffers = []
        device = next(module.parameters()).device
        if dtype is None:
            dtype = next(module.parameters()).dtype
        linear_types = FeatureExtractor._resolve_linear_types()

        if self.num_features > 0:
            layer = find_layer_by_name_recursive(module, target_layer)
            linear_modules = [m for m in layer.modules() if isinstance(m, linear_types)]
            if not linear_modules:
                raise ValueError(f"No linear-like submodules found under target layer {target_layer}")

            for i, l in enumerate(linear_modules):
                if hasattr(l, "in_features"):
                    in_dim = l.in_features + 1
                elif hasattr(l, "irreps_in"):
                    in_dim = l.irreps_in.dim + 1
                else:
                    raise AttributeError("Linear-like layer missing input dimension attributes.")

                in_feat_proj = torch.randn(in_dim, self.num_features, dtype=dtype, device=device)
                self.register_buffer(f'in_feat_proj_{i}', in_feat_proj)
                self.in_feat_proj_buffers.append(f'in_feat_proj_{i}')

                if hasattr(l, "out_features"):
                    out_dim = l.out_features
                elif hasattr(l, "irreps_out"):
                    out_dim = l.irreps_out.dim
                else:
                    raise AttributeError("Linear-like layer missing output dimension attributes.")

                out_grad_proj = torch.randn(out_dim, self.num_features, dtype=dtype, device=device)
                self.register_buffer(f'out_grad_proj_{i}', out_grad_proj)
                self.out_grad_proj_buffers.append(f'out_grad_proj_{i}')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_features={self.num_features})'

    @property
    def in_feat_proj(self) -> List[torch.Tensor]:
        return [getattr(self, name) for name in self.in_feat_proj_buffers]

    @property
    def out_grad_proj(self) -> List[torch.Tensor]:
        return [getattr(self, name) for name in self.out_grad_proj_buffers]


class FeatureKernel:
    """Compute features from (feat, grad) tuples using a fixed kernel/projector."""

    def __init__(self, kernel: str, projector: FeatureProjector) -> None:
        self.kernel = self._normalize_kernel(kernel)
        self.projector = projector

    def compute(
        self,
        feat_grad: Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]],
    ) -> torch.Tensor:
        image_idx, feats, grads = feat_grad
        local = self.kernel.startswith('local_')
        kernel = self.kernel[len('local_'):] if local else self.kernel

        if kernel == 'full-gradient':
            # Sum over all layers: (feat @ in_proj) * (grad @ out_proj)
            if self.projector.num_features == 0:
                raise ValueError("full-gradient requires random projections.")
            if not grads:
                raise RuntimeError("full-gradient requires gradient features.")
            atomic = torch.zeros(
                (image_idx.shape[0], self.projector.num_features),
                device=image_idx.device,
            )
            for feat, grad, in_proj, out_proj in zip(
                feats,
                grads,
                self.projector.in_feat_proj,
                self.projector.out_grad_proj,
            ):
                atomic += (feat @ in_proj) * (grad @ out_proj)
        elif kernel == 'll-gradient':
            # Last layer only.
            if self.projector.num_features != 0:
                if not grads:
                    raise RuntimeError("ll-gradient requires gradient features.")
                atomic = (feats[-1] @ self.projector.in_feat_proj[-1]) * (
                    grads[-1] @ self.projector.out_grad_proj[-1]
                )
            else:
                atomic = feats[-1][:, :-1]
        elif kernel == 'gnn':
            # First layer only.
            if self.projector.num_features != 0:
                if not grads:
                    raise RuntimeError("gnn with random projections requires gradient features.")
                atomic = (feats[0] @ self.projector.in_feat_proj[0]) * (
                    grads[0] @ self.projector.out_grad_proj[0]
                )
            else:
                atomic = feats[0][:, :-1]
        else:
            raise RuntimeError(f"Unknown kernel '{self.kernel}'")

        if local:
            return atomic
        return scatter_add(atomic, image_idx, dim=0)

    @staticmethod
    def _normalize_kernel(kernel: str) -> str:
        return normalize_kernel(kernel)


class FeatureCalculator(UncertaintyModule):
    """Thin orchestrator for feature extraction and kernel computation."""

    def __init__(
        self,
        extractor: Optional[FeatureExtractor] = None,
        kernels: Optional[
            Sequence[Union[FeatureKernel, Tuple[KernelName, Union[FeatureProjector, int]]]]
        ] = None,
        kernel_calculators: Optional[List[FeatureKernel]] = None,
        output_features: bool = True,
        compute_maha_dist: bool = False,
        dataset: Optional[Union[torch.utils.data.Dataset, str, Path]] = None,
        distance_kernel: Optional[str] = None,
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
        _assign_non_module_attr(self, "repr_callback", None)
        self.output_features = output_features
        self.model_outputs = [properties.feature] if self.output_features else []
        self.compute_maha_dist = compute_maha_dist
        self.dataset = dataset
        self._resolved_distance_kernel = ""
        self._distance_kernel = ""
        self.distance_kernel = distance_kernel
        self.use_node_features_direct = self._can_use_node_features_direct()
        self.max_dataset_size = max_dataset_size
        self.streaming = streaming
        self.regularization = regularization
        # Prevent recursive forward when compute(predict=True) calls the model.
        self._skip_forward = False
        self.update_uncertainty_outputs()

    def extract(
        self,
        data: properties.Type,
        predict: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        if self.use_node_features_direct:
            return self._extract_from_node_features(data, predict=predict)
        image_idx = data[properties.image_idx]
        feature_data = self.extractor(data, predict=predict)
        feats = feature_data[properties.feature]
        grads = feature_data[properties.gradient]
        return image_idx, feats, grads

    def register_repr_callback(self, repr_callback: nn.Module) -> None:
        _assign_non_module_attr(self, "repr_callback", repr_callback)
        self.kernels = self._build_kernels(
            self.kernels,
            repr_callback=repr_callback,
            target_layer=self.extractor.target_layer,
            target_domain=self.extractor.target_domain,
        )
        self.use_node_features_direct = self._can_use_node_features_direct()
        if self.use_node_features_direct:
            self.extractor.detach()
        else:
            self.extractor.attach(repr_callback)
        self._resolved_distance_kernel = ""
        if self.compute_maha_dist and self.dataset is not None:
            self.fit_distance(self.dataset, kernel=self.distance_kernel)

    def update_uncertainty_outputs(self) -> None:
        scalar_uncertainty_keys = [properties.maha_dist] if self.compute_maha_dist else []
        per_atom_uncertainty_keys = []
        if self.compute_maha_dist and self._resolve_distance_kernel().startswith("local_"):
            per_atom_uncertainty_keys = [properties.maha_dist_per_atom]
        self.set_uncertainty_outputs(
            scalar_keys=scalar_uncertainty_keys,
            per_atom_keys=per_atom_uncertainty_keys,
        )
        managed_outputs = {properties.maha_dist, properties.maha_dist_per_atom}
        self.model_outputs = [key for key in self.model_outputs if key not in managed_outputs]
        for key in [*self.uncertainty_keys, *self.per_atom_uncertainty_keys]:
            if key not in self.model_outputs:
                self.model_outputs.append(key)

    def compute(
        self,
        data: properties.Type,
        predict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if predict:
            # Avoid re-entrancy when repr_callback routes back into this module.
            self._skip_forward = True
            feat_grad = self.extract(data, predict=predict)
            self._skip_forward = False
        else:
            feat_grad = self.extract(data, predict=predict)
        if not self.kernels:
            raise RuntimeError("FeatureCalculator kernels are not initialized.")
        if any(not isinstance(kc, FeatureKernel) for kc in self.kernels):
            raise RuntimeError(
                "FeatureCalculator kernels require initialized projectors; "
                "call register_repr_callback or pass FeatureProjector."
            )
        if len(self.kernels) == 1:
            return self.kernels[0].compute(feat_grad)
        return {
            kc.kernel: kc.compute(feat_grad)
            for kc in self.kernels
        }

    def forward(self, data: properties.Type, predict: bool = False) -> properties.Type:
        if torch.jit.is_scripting():
            if self.use_node_features_direct:
                if properties.node_feat not in data:
                    raise RuntimeError("Node features are not available in model data.")
                if properties.image_idx in data:
                    image_idx = data[properties.image_idx]
                else:
                    image_idx = torch.zeros(
                        data[properties.node_feat].shape[0],
                        dtype=torch.long,
                        device=data[properties.node_feat].device,
                    )
                computed = data[properties.node_feat]
                kernel = normalize_kernel(self.distance_kernel if self.distance_kernel else "gnn")
                if not kernel.startswith("local_"):
                    computed = scatter_add(computed, image_idx, dim=0)
                if self.output_features:
                    data[properties.feature] = computed
                if self.compute_maha_dist:
                    if not hasattr(self, "precision") or not hasattr(self, "feature_mean"):
                        raise RuntimeError("Mahalanobis statistics are not initialized.")
                    feats = (computed - self.feature_mean) / self.feature_std
                    dist_sq = torch.einsum("bi,ij,bj->b", feats, self.precision, feats)
                    distances = torch.sqrt(torch.clamp(dist_sq, min=0.0))
                    if kernel.startswith("local_"):
                        data[properties.maha_dist_per_atom] = distances
                        data[properties.maha_dist] = scatter_mean(distances, image_idx, dim=0)
                    else:
                        data[properties.maha_dist] = distances
                return data
            raise RuntimeError("Hook-based FeatureCalculator runtime is not TorchScript-safe.")
        return self._forward_python(data, predict=predict)

    @torch.jit.unused
    def _forward_python(self, data, predict=False):
        if self._skip_forward:
            return data
        computed = self.compute(data, predict=predict)
        if self.output_features:
            data[properties.feature] = computed
        if self.compute_maha_dist:
            if not hasattr(self, "precision") or not hasattr(self, "feature_mean"):
                raise RuntimeError("Mahalanobis statistics are not initialized.")
            kernel = self._resolve_distance_kernel()
            if isinstance(computed, dict):
                if kernel not in computed:
                    raise RuntimeError(f"Kernel '{kernel}' is not available for Mahalanobis distance.")
                feats = computed[kernel]
            else:
                feats = computed
            feats = (feats - self.feature_mean) / self.feature_std
            dist_sq = torch.einsum("bi,ij,bj->b", feats, self.precision, feats)
            distances = torch.sqrt(torch.clamp(dist_sq, min=0.0))
            if kernel.startswith("local_"):
                data[properties.maha_dist_per_atom] = distances
                data[properties.maha_dist] = scatter_mean(distances, data[properties.image_idx], dim=0)
            else:
                data[properties.maha_dist] = distances
        return data

    # function to fit the covariance matrix for computing the Mahalanobis distance
    def fit_distance(
        self,
        dataset: Optional[Union[torch.utils.data.Dataset, str, Path]] = None,
        kernel: Optional[str] = None,
    ) -> None:
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
        dataset_size = len(dataset) if hasattr(dataset, "__len__") else None
        image_idx = None
        reduction = None
        if kernel.startswith("local_"):
            image_idx = self._build_image_idx(dataset)
            reduction = "mean"
        stats_batch_size = 8
        if hasattr(dataset, "__len__"):
            stats_batch_size = max(1, min(len(dataset), stats_batch_size))
        logger.info(
            "Fitting Mahalanobis reference statistics: kernel=%s structures=%s batch_size=%d streaming=%s",
            kernel,
            dataset_size if dataset_size is not None else "unknown",
            stats_batch_size,
            self.streaming,
        )
        stats = FeatureStatistics(
            models=[self.repr_callback],
            dataset=dataset,
            calculators=[self],
            batch_size=stats_batch_size,
            device=str(next(self.repr_callback.parameters()).device),
        )
        metrics = DistanceMetrics(regularization=self.regularization)
        metrics.fit_from_stats(
            stats,
            kernel,
            image_idx=image_idx,
            reduction=reduction,
            streaming=self.streaming,
        )
        device = next(self.repr_callback.parameters()).device
        self.register_buffer("feature_mean", metrics.mean.to(device))
        self.register_buffer("feature_std", metrics.std.to(device))
        self.register_buffer("precision", metrics.precision.to(device))
        if metrics.reference_distances is None:
            raise RuntimeError("DistanceMetrics did not compute reference distances.")
        self.register_buffer("maha_dist", metrics.reference_distances.to(device))
        self.distance_kernel = kernel
        logger.info("Mahalanobis reference statistics ready: kernel=%s", kernel)

    def _extract_from_node_features(
        self,
        data: properties.Type,
        predict: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        feature_data = data.copy()
        if predict:
            if self.repr_callback is None:
                raise ValueError("repr_callback must be set before computing features.")
            if hasattr(self.repr_callback, "input_modules") and hasattr(self.repr_callback, "representation"):
                for module in self.repr_callback.input_modules:
                    feature_data = module(feature_data)
                feature_data = self.repr_callback.representation(feature_data)
            else:
                feature_data = self.repr_callback(feature_data)
        if properties.node_feat not in feature_data:
            raise RuntimeError("Node features are not available in model data.")
        node_feat = feature_data[properties.node_feat]
        if properties.image_idx in feature_data:
            image_idx = feature_data[properties.image_idx]
        else:
            image_idx = torch.zeros(
                node_feat.shape[0],
                dtype=torch.long,
                device=node_feat.device,
            )
        bias = torch.ones(
            (node_feat.shape[0], 1),
            dtype=node_feat.dtype,
            device=node_feat.device,
        )
        return image_idx, [torch.cat((node_feat, bias), dim=-1)], []

    def _can_use_node_features_direct(self) -> bool:
        if not self.kernels:
            return False
        for kernel in self.kernels:
            if isinstance(kernel, FeatureKernel):
                if kernel.kernel not in {"gnn", "local_gnn"}:
                    return False
                if kernel.projector.num_features != 0:
                    return False
                continue
            if not isinstance(kernel, tuple) or len(kernel) != 2:
                return False
            kernel_name, projector = kernel
            if FeatureKernel._normalize_kernel(str(kernel_name)) not in {"gnn", "local_gnn"}:
                return False
            if not isinstance(projector, int) or projector != 0:
                return False
        return True

    @staticmethod
    def _build_image_idx(dataset: torch.utils.data.Dataset) -> torch.Tensor:
        total = len(dataset)
        counts: List[int] = []
        iterator = range(total)
        use_tqdm = tqdm is not None and sys.stderr.isatty()
        log_every = max(1, total // 10) if total > 0 else None
        if use_tqdm:
            iterator = tqdm(iterator, desc="build-image-idx", total=total)
        elif log_every is not None:
            logger.info("Building local-kernel image index: 0/%d structures", total)
        get_n_atoms = FeatureCalculator._resolve_n_atoms_getter(dataset)
        for i in iterator:
            if not use_tqdm and log_every is not None and (
                i == 0 or (i + 1) == total or (i + 1) % log_every == 0
            ):
                logger.info("Building local-kernel image index: %d/%d structures", i + 1, total)
            counts.append(get_n_atoms(i))
        if counts:
            return torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(counts)])
        return torch.empty((0,), dtype=torch.long)

    @staticmethod
    def _resolve_n_atoms_getter(dataset: torch.utils.data.Dataset):
        getter = getattr(dataset, "get_n_atoms", None)
        if callable(getter):
            return getter
        if isinstance(dataset, Subset):
            parent_getter = FeatureCalculator._resolve_n_atoms_getter(dataset.dataset)
            return lambda i: parent_getter(dataset.indices[i])

        def generic_get_n_atoms(i: int) -> int:
            sample = dataset[i]
            if hasattr(sample, "to_dict"):
                sample = sample.to_dict()
            n_atoms = sample[properties.n_atoms]
            if torch.is_tensor(n_atoms):
                return int(n_atoms.item())
            return int(n_atoms)

        return generic_get_n_atoms

    def _resolve_distance_kernel(self, kernel: Optional[str] = None) -> str:
        if kernel is None and self._resolved_distance_kernel:
            return self._resolved_distance_kernel
        if kernel is not None:
            kernel = normalize_kernel(str(kernel))
            logger.info("Distance kernel override: %s", kernel)
        if kernel is None:
            if self.distance_kernel:
                kernel = normalize_kernel(str(self.distance_kernel))
                logger.info("Distance kernel from config: %s", kernel)
            elif self.kernels:
                first = self.kernels[0]
                kernel = first.kernel if isinstance(first, FeatureKernel) else str(first[0])
                logger.info("Distance kernel from first kernel: %s", kernel)
            else:
                kernel = _DEFAULT_KERNEL
                logger.info("Distance kernel fallback: %s", kernel)
        self._resolved_distance_kernel = normalize_kernel(str(kernel))
        return self._resolved_distance_kernel

    @property
    def distance_kernel(self) -> str:
        return self._distance_kernel

    @distance_kernel.setter
    def distance_kernel(self, kernel: Optional[str]) -> None:
        self._distance_kernel = "" if kernel is None else str(kernel)
        self._resolved_distance_kernel = ""

    def _resolve_dataset(
        self,
        dataset: Union[torch.utils.data.Dataset, str, Path],
    ) -> torch.utils.data.Dataset:
        if isinstance(dataset, (str, Path)):
            cutoff = find_layer_by_name_recursive(self.repr_callback, "cutoff") if self.repr_callback else None
            return AseDataset(dataset, cutoff=cutoff or 5.0)
        return dataset

    @staticmethod
    def _build_kernels(
        kernels: Optional[Sequence[Union[FeatureKernel, Tuple[KernelName, Union[FeatureProjector, int]]]]],
        repr_callback: Optional[nn.Module],
        target_layer: str = 'readout_mlp',
        target_domain: Optional[Union[str, int]] = None,
    ) -> List[Union[FeatureKernel, Tuple[KernelName, int]]]:
        if kernels is None:
            kernels = [(_DEFAULT_KERNEL, _DEFAULT_N_RANDOM_FEATURES)]
        built: List[Union[FeatureKernel, Tuple[KernelName, int]]] = []
        for item in kernels:
            if isinstance(item, FeatureKernel):
                built.append(item)
                continue
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("kernels must be FeatureKernel or (kernel, projector/n_random_features).")
            kernel, projector = item
            if isinstance(projector, FeatureProjector):
                built.append(FeatureKernel(kernel, projector))
            elif isinstance(projector, int):
                if repr_callback is None:
                    built.append((kernel, projector))
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
                            RandomProjections(
                                module,
                                projector,
                                target_layer=target_layer,
                            ),
                        )
                    )
            else:
                raise ValueError("kernels projector must be FeatureProjector or int.")
        return built


class H5Feature:
    """Lightweight HDF5 feature store with append support."""

    def __init__(
        self,
        path: Union[str, Path],
        num_models: int,
        kernels: Optional[Sequence[KernelName]] = None,
        dataset_size: Optional[int] = None,
        compression: Optional[str] = None,
        chunk_rows: Optional[int] = None,
    ) -> None:
        self.path = Path(path)
        self.num_models = int(num_models)
        self.kernels = list(kernels) if kernels is not None else None
        self.dataset_size = int(dataset_size) if dataset_size is not None else None
        self.compression = compression
        self.chunk_rows = chunk_rows
        if self.kernels is not None:
            self.ensure(self.kernels, dataset_size=self.dataset_size)

    def ensure(
        self,
        kernels: Optional[Sequence[KernelName]] = None,
        dataset_size: Optional[int] = None,
    ) -> List[str]:
        if kernels is None:
            kernels = self.kernels
        if kernels is None:
            raise ValueError("kernels are required.")
        kernels_list = [str(k) for k in kernels]
        with h5py.File(self.path, "a") as handle:
            existing = handle.attrs.get("kernels")
            if existing is None:
                handle.attrs["kernels"] = kernels_list
            else:
                stored = [
                    k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
                    for k in existing
                ]
                if stored != kernels_list:
                    raise ValueError("HDF5 kernels do not match.")
            existing_models = handle.attrs.get("num_models")
            if existing_models is None:
                handle.attrs["num_models"] = self.num_models
            elif int(existing_models) != self.num_models:
                raise ValueError("HDF5 num_models does not match.")
            if dataset_size is not None:
                existing_size = handle.attrs.get("dataset_size")
                if existing_size is None:
                    handle.attrs["dataset_size"] = int(dataset_size)
                elif int(existing_size) != int(dataset_size):
                    raise ValueError("HDF5 dataset_size does not match.")
        self.kernels = kernels_list
        if dataset_size is not None:
            self.dataset_size = int(dataset_size)
        return kernels_list

    def count(self, kernel: KernelName, model_idx: int) -> int:
        if model_idx < 0 or model_idx >= self.num_models:
            raise ValueError("model_idx is out of range.")
        with h5py.File(self.path, "a") as handle:
            group = handle.get(f"features/{kernel}")
            if group is None or "counts" not in group:
                return 0
            return int(group["counts"][model_idx])

    def append(
        self,
        kernel: KernelName,
        model_idx: int,
        feats: torch.Tensor,
        image_idx: Optional[torch.Tensor] = None,
    ) -> None:
        if model_idx < 0 or model_idx >= self.num_models:
            raise ValueError("model_idx is out of range.")
        if not torch.is_tensor(feats):
            raise TypeError("feats must be a torch.Tensor.")
        if feats.dim() != 2:
            raise ValueError("feats must be 2D (N, P).")
        if feats.numel() == 0:
            return
        if image_idx is not None:
            if not torch.is_tensor(image_idx):
                raise TypeError("image_idx must be a torch.Tensor.")
            if image_idx.dim() != 1:
                raise ValueError("image_idx must be 1D.")
            if image_idx.shape[0] != feats.shape[0]:
                raise ValueError("image_idx length must match feats rows.")

        with h5py.File(self.path, "a") as handle:
            # write features for a given kernel, data is (num_models, num_structures, num_features)
            group = handle.require_group(f"features/{kernel}")
            data = group.get("data")
            if data is None:
                chunks = True
                if self.chunk_rows is not None:
                    chunks = (1, self.chunk_rows, feats.shape[1])
                data = group.create_dataset(
                    "data",
                    shape=(self.num_models, 0, feats.shape[1]),
                    maxshape=(self.num_models, None, feats.shape[1]),
                    chunks=chunks,
                    compression=self.compression,
                )
            elif data.shape[0] != self.num_models or data.shape[2] != feats.shape[1]:
                raise ValueError("HDF5 data shape does not match.")

            # write counts to restore feature computation
            counts = group.get("counts")
            if counts is None:
                counts = group.create_dataset(
                    "counts",
                    data=[0] * self.num_models,
                    dtype="i8",
                )
            current = int(counts[model_idx])
            new_total = current + feats.shape[0]
            if new_total > data.shape[1]:
                data.resize((self.num_models, new_total, data.shape[2]))
            data[model_idx, current:new_total, :] = feats.detach().cpu().numpy()
            counts[model_idx] = new_total

            # write image_idx
            if image_idx is not None:
                idx = group.get("image_idx")
                if idx is None:
                    idx = group.create_dataset(
                        "image_idx",
                        shape=(self.num_models, 0),
                        maxshape=(self.num_models, None),
                        chunks=True,
                        dtype="i8",
                    )
                old = idx.shape[1]
                if new_total > old:
                    idx.resize((self.num_models, new_total))
                idx[model_idx, current:new_total] = image_idx.detach().cpu().numpy()

    def load(self, kernel: KernelName) -> torch.Tensor:
        with h5py.File(self.path, "r") as handle:
            group = handle.get(f"features/{kernel}")
            if group is None or "data" not in group:
                return torch.empty((self.num_models, 0, 0))
            return torch.from_numpy(group["data"][()])

    def load_with_counts(self, kernel: KernelName) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        with h5py.File(self.path, "r") as handle:
            group = handle.get(f"features/{kernel}")
            if group is None or "data" not in group:
                data = torch.empty((self.num_models, 0, 0))
                counts = torch.zeros((self.num_models,), dtype=torch.long)
                return data, counts, None
            data = torch.from_numpy(group["data"][()])
            if "counts" in group:
                counts = torch.from_numpy(group["counts"][()])
            else:
                counts = torch.full((self.num_models,), data.shape[1], dtype=torch.long)
            image_idx = None
            if "image_idx" in group:
                image_idx = torch.from_numpy(group["image_idx"][()])
            return data, counts, image_idx

    def load_image_idx(self, kernel: KernelName) -> Optional[torch.Tensor]:
        with h5py.File(self.path, "r") as handle:
            group = handle.get(f"features/{kernel}")
            if group is None or "image_idx" not in group:
                return None
            return torch.from_numpy(group["image_idx"][()])


class FeatureStatistics:
    """Compute features and ensemble stats for a dataset."""

    def __init__(
        self,
        models: List[nn.Module],
        dataset: torch.utils.data.Dataset,
        kernels: Optional[Sequence[Union[KernelName, Tuple[KernelName, int]]]] = None,
        calculators: Optional[List[FeatureCalculator]] = None,
        n_random_features: int = 500,
        target_layer: str = 'readout_mlp',
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
        self.n_random_features = n_random_features
        self.target_layer = target_layer
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

    def get_ens_stats(
        self,
        dataset: Optional[torch.utils.data.Dataset] = None,
    ) -> Dict[str, torch.Tensor]:
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
        size_value = size if size is not None else "?"
        desc = f"ensemble size={size_value} bs={self.batch_size}"
        log_ctx = logging_redirect_tqdm() if logging_redirect_tqdm is not None else contextlib.nullcontext()
        with log_ctx:
            logger.info("Computing ensemble stats")
            for batch in self._iter_batches(self.dataset, dtype=model_dtype, desc=desc):
                out = self._ensemble(batch)
                outputs.append({k: v.detach().cpu() for k, v in out.items()})
        self._ens_stats = {
            k: torch.cat([o[k] for o in outputs])
            for k in outputs[0].keys()
        }
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
            extractor = FeatureExtractor(repr_callback=model, target_layer=self.target_layer)
            calculators.append(FeatureCalculator(extractor=extractor, kernels=kernel_specs))
        kernel_names = [FeatureKernel._normalize_kernel(k) for k, _ in kernel_specs]
        return calculators, kernel_names

    def iter_kernel_features(self, kernel: KernelName):
        calculators, kernel_names = self._resolve_calculators()
        norm_kernel = FeatureKernel._normalize_kernel(str(kernel))
        if norm_kernel not in kernel_names:
            raise ValueError(f"Kernel '{norm_kernel}' is not available in FeatureStatistics.")
        size = len(self.dataset) if hasattr(self.dataset, "__len__") else None
        size_value = size if size is not None else "?"
        model_dtype = next(self.models[0].parameters()).dtype
        desc = f"stream-kernel={norm_kernel} size={size_value} bs={self.batch_size}"
        log_ctx = logging_redirect_tqdm() if logging_redirect_tqdm is not None else contextlib.nullcontext()
        with log_ctx:
            logger.info("Streaming features for kernel %s", norm_kernel)
            for batch in self._iter_batches(self.dataset, dtype=model_dtype, desc=desc):
                per_model = []
                for calculator in calculators:
                    computed = calculator.compute(batch, predict=True)
                    results = self._as_dict(computed, kernel_names)
                    per_model.append(results[norm_kernel])
                feats = per_model[0] if len(per_model) == 1 else torch.stack(per_model).mean(dim=0)
                yield feats

    def _resolve_kernel_specs(self) -> List[Tuple[str, int]]:
        if self.kernels is None:
            kernels: List[Union[str, Tuple[str, int]]] = [_DEFAULT_KERNEL]
        else:
            kernels = list(self.kernels)
        specs: List[Tuple[str, int]] = []
        seen: set[str] = set()
        for item in kernels:
            if isinstance(item, tuple):
                if len(item) != 2 or not isinstance(item[1], int):
                    raise ValueError("kernels tuple must be (kernel, n_random_features).")
                kernel, n_features = item
            else:
                kernel, n_features = item, self.n_random_features
            norm = FeatureKernel._normalize_kernel(str(kernel))
            if norm in seen:
                raise ValueError(f"Duplicate kernel '{norm}'.")
            seen.add(norm)
            specs.append((str(kernel), int(n_features)))
        return specs

    @staticmethod
    def _kernel_names_from_calculators(
        calculators: List[FeatureCalculator],
    ) -> List[str]:
        kernels = calculators[0].kernels
        names = [
            kc.kernel if isinstance(kc, FeatureKernel) else FeatureKernel._normalize_kernel(kc[0])
            for kc in kernels
        ]
        for calc in calculators[1:]:
            other = [
                kc.kernel if isinstance(kc, FeatureKernel) else FeatureKernel._normalize_kernel(kc[0])
                for kc in calc.kernels
            ]
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

        yield from iter_batches(
            dataset=dataset,
            batch_size=self.batch_size,
            device=self.device,
            dtype=dtype,
            desc=desc,
        )

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
        """Compute features into in-memory cache when no H5Feature store is used."""
        cache: Dict[str, List[List[torch.Tensor]]] = {
            k: [list() for _ in self.models] for k in kernel_names
        }
        size = len(self.dataset) if hasattr(self.dataset, "__len__") else None
        size_value = size if size is not None else "?"
        for model_idx, (model, calculator) in enumerate(zip(self.models, calculators)):
            model_dtype = next(model.parameters()).dtype
            desc = (
                f"model={model.__class__.__name__} kernels={len(kernel_names)} "
                f"size={size_value} bs={self.batch_size}"
            )
            log_ctx = logging_redirect_tqdm() if logging_redirect_tqdm is not None else contextlib.nullcontext()
            with log_ctx:
                logger.info("Computing features for model %s", model.__class__.__name__)
                for b, batch in enumerate(self._iter_batches(self.dataset, dtype=model_dtype, desc=desc)):
                    computed = calculator.compute(batch, predict=True)
                    results = self._as_dict(computed, kernel_names)
                    for kernel in kernel_names:
                        cache[kernel][model_idx].append(results[kernel].cpu())
                    if self.checkpoint_interval > 0 and (b + 1) % self.checkpoint_interval == 0:
                        if self.save_path is not None:
                            torch.save(self._stack_features(cache, kernel_names), self.save_path)
        return cache

    def _compute_store(
        self,
        calculators: List[FeatureCalculator],
        kernel_names: List[str],
    ) -> None:
        """Compute features and append to H5Feature store for checkpointing/resume."""
        size = len(self.dataset) if hasattr(self.dataset, "__len__") else None
        size_value = size if size is not None else "?"
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
            desc = (
                f"model={model.__class__.__name__} kernels={len(kernel_names)} "
                f"size={size_value} bs={self.batch_size}"
            )
            log_ctx = logging_redirect_tqdm() if logging_redirect_tqdm is not None else contextlib.nullcontext()
            with log_ctx:
                if offset > 0:
                    logger.info(
                        "Resuming feature store for model %s at index %d/%s",
                        model.__class__.__name__,
                        offset,
                        size_value,
                    )
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
                            image_idx = local_idx
                            if batch_start < offset:
                                feats = feats[local_mask]
                                image_idx = image_idx[local_mask]
                        else:
                            image_idx = global_idx
                            if batch_start < offset:
                                feats = feats[global_cut:]
                                image_idx = image_idx[global_cut:]
                        self.store.append(kernel, model_idx, feats, image_idx)
                    global_index = batch_end

    @staticmethod
    def _as_dict(
        computed: Union[torch.Tensor, Dict[str, torch.Tensor]],
        kernel_names: List[str],
    ) -> Dict[str, torch.Tensor]:
        if isinstance(computed, dict):
            return computed
        return {kernel_names[0]: computed}

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
            per_model = []
            for batches in cache[kernel]:
                per_model.append(torch.cat(batches) if batches else torch.empty((0, 0)))
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


class DistanceMetrics:
    """Compute distances from feature tensors."""

    def __init__(
        self,
        regularization: float = 1e-6,
        reduction: Optional[Reduction] = None,
    ) -> None:
        self.regularization = regularization
        self.reduction = reduction
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.precision: Optional[torch.Tensor] = None
        self.reference_distances: Optional[torch.Tensor] = None

    def fit(
        self,
        features: torch.Tensor,
        image_idx: Optional[torch.Tensor] = None,
        reduction: Optional[Reduction] = None,
    ) -> None:
        feats = self._prepare_features(features)
        mean = torch.mean(feats, dim=0)
        std = torch.std(feats, dim=0)
        std = torch.where(std == 0, torch.ones_like(std), std)
        norm = (feats - mean) / std
        denom = max(norm.shape[0] - 1, 1)
        covariance = norm.T @ norm / denom
        eye = torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
        covariance = covariance + self.regularization * eye
        self.mean = mean
        self.std = std
        self.precision = torch.linalg.inv(covariance)
        dist_sq = torch.einsum("bi,ij,bj->b", norm, self.precision, norm)
        distances = torch.sqrt(torch.clamp(dist_sq, min=0.0))
        self.reference_distances = self._reduce(distances, image_idx, reduction)

    def fit_from_stats(
        self,
        stats: FeatureStatistics,
        kernel: KernelName,
        image_idx: Optional[torch.Tensor] = None,
        reduction: Optional[Reduction] = None,
        streaming: bool = False,
    ) -> None:
        if streaming:
            self.fit_from_stats_streaming(stats, kernel, image_idx=image_idx, reduction=reduction)
            return
        features = stats.get_features(normalize=False)
        if kernel not in features:
            raise ValueError(f"Kernel '{kernel}' is not available in FeatureStatistics.")
        self.fit(features[kernel], image_idx=image_idx, reduction=reduction)

    def fit_from_stats_streaming(
        self,
        stats: FeatureStatistics,
        kernel: KernelName,
        image_idx: Optional[torch.Tensor] = None,
        reduction: Optional[Reduction] = None,
    ) -> None:
        logger.info("Streaming stats pass 1/2: mean/std/precision")
        mean, std, precision = self._streaming_stats(stats, kernel)
        logger.info("Streaming stats pass 2/2: reference distances")
        distances = self._streaming_distances(stats, kernel, mean, std, precision)
        self.mean = mean
        self.std = std
        self.precision = precision
        self.reference_distances = self._reduce(distances, image_idx, reduction)

    def _streaming_stats(
        self,
        stats: FeatureStatistics,
        kernel: KernelName,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count = 0
        sum_x = None
        sum_x2 = None
        sum_xxT = None
        for feats in stats.iter_kernel_features(kernel):
            feats = feats.detach().cpu()
            feats = self._prepare_features(feats)
            if feats.numel() == 0:
                continue
            batch_n = feats.shape[0]
            batch_sum = feats.sum(dim=0)
            batch_sum2 = feats.pow(2).sum(dim=0)
            batch_sum_xxT = feats.T @ feats
            if sum_x is None:
                sum_x = batch_sum
                sum_x2 = batch_sum2
                sum_xxT = batch_sum_xxT
            else:
                sum_x += batch_sum
                sum_x2 += batch_sum2
                sum_xxT += batch_sum_xxT
            count += batch_n
        if sum_x is None or sum_x2 is None or sum_xxT is None or count == 0:
            raise RuntimeError("No features available for streaming statistics.")
        denom = max(count - 1, 1)
        mean = sum_x / count
        var = (sum_x2 - count * mean.pow(2)) / denom
        std = torch.sqrt(var)
        std = torch.where(std == 0, torch.ones_like(std), std)
        covariance = (sum_xxT - count * torch.outer(mean, mean)) / denom
        eye = torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
        covariance = covariance + self.regularization * eye
        precision = torch.linalg.inv(covariance)
        return mean, std, precision

    def _streaming_distances(
        self,
        stats: FeatureStatistics,
        kernel: KernelName,
        mean: torch.Tensor,
        std: torch.Tensor,
        precision: torch.Tensor,
    ) -> torch.Tensor:
        distances = []
        for feats in stats.iter_kernel_features(kernel):
            feats = feats.detach().cpu()
            feats = self._prepare_features(feats)
            if feats.numel() == 0:
                continue
            norm = (feats - mean) / std
            dist_sq = torch.einsum("bi,ij,bj->b", norm, precision, norm)
            distances.append(torch.sqrt(torch.clamp(dist_sq, min=0.0)))
        if not distances:
            raise RuntimeError("No features available for streaming distances.")
        return torch.cat(distances, dim=0)

    def score(
        self,
        features: torch.Tensor,
        image_idx: Optional[torch.Tensor] = None,
        reduction: Optional[Reduction] = None,
    ) -> torch.Tensor:
        if self.mean is None or self.std is None or self.precision is None:
            raise RuntimeError("DistanceMetrics must be fit before scoring.")
        feats = self._prepare_features(features)
        norm = (feats - self.mean) / self.std
        dist_sq = torch.einsum("bi,ij,bj->b", norm, self.precision, norm)
        distances = torch.sqrt(torch.clamp(dist_sq, min=0.0))
        return self._reduce(distances, image_idx, reduction)

    def score_euclidean(
        self,
        features: torch.Tensor,
        image_idx: Optional[torch.Tensor] = None,
        reduction: Optional[Reduction] = None,
    ) -> torch.Tensor:
        feats = self._prepare_features(features)
        distances = torch.norm(feats, dim=1)
        return self._reduce(distances, image_idx, reduction)

    def score_cosine(
        self,
        features: torch.Tensor,
        image_idx: Optional[torch.Tensor] = None,
        reduction: Optional[Reduction] = None,
    ) -> torch.Tensor:
        feats = self._prepare_features(features)
        norms = torch.norm(feats, dim=1)
        distances = 1.0 - torch.sum(feats, dim=1) / torch.clamp(norms, min=1e-12)
        return self._reduce(distances, image_idx, reduction)

    def score_from_stats(
        self,
        stats: FeatureStatistics,
        kernel: KernelName,
        image_idx: Optional[torch.Tensor] = None,
        reduction: Optional[Reduction] = None,
    ) -> torch.Tensor:
        features = stats.get_features(normalize=False)
        if kernel not in features:
            raise ValueError(f"Kernel '{kernel}' is not available in FeatureStatistics.")
        return self.score(features[kernel], image_idx=image_idx, reduction=reduction)

    @staticmethod
    def _prepare_features(features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 3:
            features = features.mean(dim=0)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if features.dim() != 2:
            raise ValueError("Features must be 2D or 3D.")
        return features

    def _reduce(
        self,
        distances: torch.Tensor,
        image_idx: Optional[torch.Tensor],
        reduction: Optional[Reduction],
    ) -> torch.Tensor:
        reduction = reduction or self.reduction
        if reduction is None:
            return distances
        if image_idx is None:
            raise ValueError("image_idx is required for distance reduction.")
        if reduction == "mean":
            return scatter_mean(distances, image_idx, dim=0)
        if reduction == "sum":
            return scatter_add(distances, image_idx, dim=0)
        raise ValueError(f"Unsupported reduction '{reduction}'.")
