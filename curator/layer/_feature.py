from curator.data import properties
try:
    from torch_scatter import scatter_add, scatter_mean
except ImportError:
    from curator.utils import scatter_add, scatter_mean
from importlib import util
from torch import nn
import torch
from typing import Dict, Optional, Callable, List, Union, Sequence
import itertools
import logging
from .utils import find_layer_by_name_recursive

logger = logging.getLogger(__name__)

class FeatureExtractor(nn.Module):
    """Extract features from neural networks"""
    def __init__(
        self,
        repr_callback: Optional[Callable] = None,
        model_outputs: List[str] = ['feature', 'gradient'],
        target_layer: str = 'readout_mlp',
    ) -> None:
        """Extract features from neural networks
        
        Args:
            repr_callback: pytorch nn.Module
        """
        super().__init__()
        self.repr_callback = repr_callback       # use callback mechanism
        self._features = []
        self._grads = []
        self.hooks = []
        self.model_outputs = model_outputs
        self.target_layer = target_layer
        self._linear_types = self._resolve_linear_types()

        if self.repr_callback is not None:
            self.add_hooks()
        
    def save_feats_hook(self, _, in_feat):
        new_feat = torch.cat((in_feat[0].detach().clone(), torch.ones_like(in_feat[0][:, 0:1])), dim=-1)
        self._features.append(new_feat)
    
    def save_grads_hook(self, _, __, grad_output):
        self._grads.append(grad_output[0].detach().clone())
    
    def unhook(self):
        for hook in self.hooks:
            hook.remove()
    
    def register_repr_callback(self):
        self.add_hooks()

    def add_hooks(self):
        layer = find_layer_by_name_recursive(self.repr_callback, self.target_layer)
        assert layer is not None, f"Target layer {self.target_layer} is not found!"
        # # Avoid direct imports; use class name string comparison instead for efficiency
        # if self.repr_callback.__class__.__name__ == 'MACE':
        #     layer = layer[-1]
        linear_modules = [m for m in layer.modules() if isinstance(m, self._linear_types)]
        if not linear_modules:
            logger.warning(f"No linear-like submodules found under target layer {self.target_layer}")
        for child in linear_modules:
            self.hooks.append(child.register_forward_pre_hook(self.save_feats_hook))
            self.hooks.append(child.register_backward_hook(self.save_grads_hook))

    def forward(self, data: properties.Type, predict: bool=False) -> properties.Type:
        # repr_callback may modify the original data in place, so we need to make a copy of the data
        new_data = data.copy()
        if predict: # in predict mode, modify the data in place
            new_data = self.repr_callback(new_data)
        data[properties.feature] = self._features
        data[properties.gradient] = self._grads[::-1]
        self._reset()

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(target_layer={self.target_layer})'

    def _reset(self) -> None:
        self._features = []
        self._grads = []

    @staticmethod
    def _resolve_linear_types() -> Sequence[type]:
        types: List[type] = [nn.Linear]
        if util.find_spec("e3nn.o3"):
            from e3nn import o3
            types.append(o3.Linear)
        # Add cuequivariance Linear if available
        try:
            import cuequivariance_torch as cuet
            types.append(cuet.Linear)
        except Exception:
            pass
        return tuple(types)

class RandomProjections(nn.Module):
    """Random projection module with Gaussian distributions, storing projection matrices as buffers."""
    def __init__(
        self,
        module: nn.Module,
        num_features: int,
        dtype = torch.get_default_dtype(),
        target_layer: str = 'readout_mlp',
    ):
        super(RandomProjections, self).__init__()

        self.num_features = num_features
        self.in_feat_proj_buffers = []  # Store references to projection matrices for later use
        self.out_grad_proj_buffers = []
        device = next(module.parameters()).device
        linear_types = FeatureExtractor._resolve_linear_types()

        if self.num_features > 0:
            # Calculate normalization constant once
            # normalization = torch.sqrt(torch.tensor(self.num_features, dtype=dtype, device=device))
            layer = find_layer_by_name_recursive(module, target_layer)
            linear_modules = [m for m in layer.modules() if isinstance(m, linear_types)]
            if not linear_modules:
                raise ValueError(f"No linear-like submodules found under target layer {target_layer}")

            # Input feature and output gradient projection matrices (in_features + 1 for bias term)
            for i, l in enumerate(linear_modules):
                if hasattr(l, "in_features"):
                    in_dim = l.in_features + 1
                elif hasattr(l, "irreps_in"):
                    in_dim = l.irreps_in.dim + 1
                else:
                    raise AttributeError("Linear-like layer missing input dimension attributes.")

                in_feat_proj = torch.randn(in_dim, self.num_features, dtype=dtype, device=device)
                self.register_buffer(f'in_feat_proj_{i}', in_feat_proj)
                self.in_feat_proj_buffers.append(f'in_feat_proj_{i}')  # Store buffer names for access

                if hasattr(l, "out_features"):
                    out_dim = l.out_features
                elif hasattr(l, "irreps_out"):
                    out_dim = l.irreps_out.dim
                else:
                    raise AttributeError("Linear-like layer missing output dimension attributes.")

                out_grad_proj = torch.randn(out_dim, self.num_features, dtype=dtype, device=device)
                self.register_buffer(f'out_grad_proj_{i}', out_grad_proj)
                self.out_grad_proj_buffers.append(f'out_grad_proj_{i}')  # Store buffer names for access

    def __repr__(self):
        return f'{self.__class__.__name__}(num_features={self.num_features})'

    @property
    def in_feat_proj(self) -> List[torch.Tensor]:
        return [getattr(self, name) for name in self.in_feat_proj_buffers]

    @property
    def out_grad_proj(self) -> List[torch.Tensor]:
        return [getattr(self, name) for name in self.out_grad_proj_buffers]
        
class FeatureCalculator(nn.Module):
    def __init__(
        self,
        repr_callback: Optional[nn.Module] = None,   # which module to extract features from
        kernel: str = 'local-full-g',    # select from full-gradient, ll-gradient, gnn, local-full-g, local-ll-g, local-gnn
        n_random_features: int = 500,
        model_outputs = ['feature'],
        target_layer: str = 'readout_mlp',
        dataset: Union[torch.utils.data.Dataset, str, None] = None,
        compute_maha_dist: bool = False,
        precision: Optional[torch.Tensor] = None,
        feature_mean: Optional[torch.Tensor] = None,
        max_dataset_size: Union[int, str, None] = None,  # None or "all" means use entire dataset
        batch_size: int = 8,  # batch size for computing covariance matrix, keep small to avoid OOM
    ) -> None:
        super().__init__()
        self.n_random_features = n_random_features
        self.kernel = kernel
        self.model_outputs = model_outputs
        self.repr_callback = repr_callback
        self.target_layer = target_layer
        self.dataset = dataset
        self.compute_maha_dist = compute_maha_dist
        # Handle max_dataset_size: None or "all" means use entire dataset
        if isinstance(max_dataset_size, str) and max_dataset_size.lower() == "all":
            max_dataset_size = None
        self.max_dataset_size = max_dataset_size
        self.batch_size = batch_size
        self._skip_forward: bool = False  # avoid re-entrancy when repr_callback routes back here

        if self.compute_maha_dist and properties.maha_dist not in self.model_outputs:
            self.model_outputs.append(properties.maha_dist)

        if repr_callback is not None:
            self.initialize_feature_components(
                repr_callback=repr_callback,
                precision=precision,
                feature_mean=feature_mean,
                dataset=self.dataset,
            )
        elif self.compute_maha_dist:
            if self.dataset is not None:
                logger.warning('Module repr_callback is not provided. To calculate mahalanobis, the provided dataset will be used to calculate precision matrix afterwards.')
            else:
                raise ValueError("repr_callback is required when compute_maha_dist is enabled.")

    def initialize_feature_components(
        self,
        repr_callback: Optional[nn.Module] = None,
        precision: Optional[torch.Tensor] = None,
        feature_mean: Optional[torch.Tensor] = None,
        dataset: Union[torch.utils.data.Dataset, str, None] = None,
    ):
        new_repr = repr_callback or self.repr_callback
        if new_repr is None:
            raise ValueError("repr_callback must be provided to initialize feature components.")

        if dataset is not None:
            self.dataset = dataset

        same_repr = new_repr is self.repr_callback and hasattr(self, "feature_extractor") and hasattr(self, "random_projections")
        self.repr_callback = new_repr

        if not same_repr:
            self.feature_extractor = FeatureExtractor(self.repr_callback, target_layer=self.target_layer)
            self.random_projections = RandomProjections(self.repr_callback, self.n_random_features, target_layer=self.target_layer)
            self._sync_device_with_repr()

        should_compute_cov = self.compute_maha_dist and (
            (precision is not None and feature_mean is not None)
            or not hasattr(self, 'precision')
            or dataset is not None
        )
        if should_compute_cov:
            self.get_covariance_matrix(
                precision=precision,
                feature_mean=feature_mean,
                dataset=dataset or self.dataset,
            )
    
    def register_repr_callback(self, repr_callback: nn.Module):
        self.initialize_feature_components(
            repr_callback=repr_callback,
            dataset=self.dataset,
        )

    def _sync_device_with_repr(self):
        if self.repr_callback is None:
            return
        device = next(self.repr_callback.parameters()).device
        self.to(device)
        for name, buf in self.named_buffers(recurse=False):
            if buf.device != device:
                setattr(self, name, buf.to(device))

    def forward(self, data: properties.Type, predict: bool=False) -> properties.Type:
        if self._skip_forward:
            return data
        data = self._compute_feature(data, predict=predict)
        if self.compute_maha_dist:
            data = self.mahalanobis_distance(data)
        return data

    def get_covariance_matrix(
            self,
            precision: Optional[torch.Tensor] = None,
            feature_mean: Optional[torch.Tensor] = None,
            dataset: Union[torch.utils.data.Dataset, str, None] = None,
        ):
        if not self.compute_maha_dist:
            return

        if self.repr_callback is None:
            raise ValueError("repr_callback is required to compute covariance and Mahalanobis distance.")

        if precision is not None and feature_mean is not None:
            logger.info('Loading precision matrix and feature mean from provided values.')
            device = next(self.repr_callback.parameters()).device if self.repr_callback is not None else torch.device('cpu')
            self.register_buffer('precision', precision.to(device))
            self.register_buffer('feature_mean', feature_mean.to(device))
            return
            
        if dataset is None:
            dataset = self.dataset
        if dataset is None:
            raise ValueError("Mahalanobis distance requested but no dataset or precision/feature_mean provided.")
        if isinstance(dataset, str):
            from curator.data import AseDataset, MatScipyNeighborList
            logger.info(f'Calculating features from provided dataset <{dataset}>.')
            cutoff = find_layer_by_name_recursive(self.repr_callback, 'cutoff')
            # Include neighbor list transform with cell_displacements to match training data preprocessing
            transforms = [MatScipyNeighborList(cutoff=cutoff, return_cell_displacements=True)]
            dataset = AseDataset(dataset, cutoff=cutoff, transforms=transforms)
            logger.info(f'Calculating precision matrix from {len(dataset)} structures.')

        # collect features
        if hasattr(self.repr_callback, 'model_outputs'):
            self.repr_callback.model_outputs.append('all')
        features = []
        image_idx = [] if 'local' in self.kernel else None
        device = next(self.repr_callback.parameters()).device
        from curator.data import collate_atomsdata
        from torch.utils.data import DataLoader, Subset
        
        # Limit dataset size if max_dataset_size is specified
        if self.max_dataset_size is not None and len(dataset) > self.max_dataset_size:
            indices = torch.randperm(len(dataset))[:self.max_dataset_size].tolist()
            dataset = Subset(dataset, indices)
            logger.info(f"Subsampled dataset to {self.max_dataset_size} structures for covariance calculation.")

        # Use self.batch_size for DataLoader, capped by dataset length
        batch_size = min(self.batch_size, len(dataset))
        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_atomsdata,
        )
        iterator = DataLoader(dataset, **loader_kwargs)

        try:
            self._skip_forward = True  # prevent recursive call when repr_callback is the parent model
            for i, batch in enumerate(iterator):
                if i % 100 == 0:
                    logger.info(f"Processing batch {i+1}/{len(iterator)}")
                batch = {k: v.to(device) for k, v in batch.items()}
                feats = self._compute_feature(batch, predict=True)[properties.feature].to('cpu')  # use cpu to save memory
                features.append(feats)
                # Handle image_idx appropriately for global vs local features
                use_local = 'local' in self.kernel if hasattr(self, 'kernel') else False

                if use_local:
                    # Local: Atom-level, construct image_idx per atom
                    n_atoms = batch[properties.n_atoms]
                    if torch.is_tensor(n_atoms):
                        batch_img_idx = []
                        for img_j, n in enumerate(n_atoms):
                            batch_img_idx.append(torch.full((n,), i * len(n_atoms) + img_j, dtype=torch.long))
                        batch_img_idx = torch.cat(batch_img_idx)
                    else:
                        batch_img_idx = torch.ones(batch[properties.n_atoms], dtype=torch.long) * i
                    image_idx.append(batch_img_idx)
                else:
                    # Global: One entry per image (batch)
                    # image_idx should be a tensor of image indices, one per structure in batch
                    n_images = batch[properties.n_atoms].shape[0] if torch.is_tensor(batch[properties.n_atoms]) else len(batch[properties.n_atoms])
                    batch_img_idx = torch.arange(i * n_images, i * n_images + n_images, dtype=torch.long)
                    image_idx.append(batch_img_idx)
        finally:
            self._skip_forward = False

        # calculate inverse covariance matrix
        features = torch.cat(features)
        if image_idx is not None:
            image_idx = torch.cat(image_idx)
        # normalization for numerical stability
        mean = features.mean(dim=0)
        std = features.std(dim=0)
        std = torch.where(std == 0, torch.ones_like(std), std)
        features = (features - mean) / std
        cov_matrix = torch.cov(features.T)
        precision = torch.inverse(cov_matrix + torch.eye(cov_matrix.size(0)) * 1e-3) # add a regularization term

        # calculate 95th percentile for uncertainty threshold
        maha_dist = torch.sqrt(torch.einsum("ij,jk,ik->i", features, precision, features))
        if image_idx is not None:
            maha_dist = scatter_mean(maha_dist, image_idx, dim=0)

        device = next(self.repr_callback.parameters()).device if self.repr_callback is not None else torch.device('cpu')
        self.register_buffer('feature_mean', mean.to(device))
        self.register_buffer('feature_std', std.to(device))
        self.register_buffer('cov_matrix', cov_matrix.to(device))
        self.register_buffer('precision', precision.to(device))
        self.register_buffer('maha_dist', maha_dist.to(device))

    def mahalanobis_distance(self, data: properties.Type) -> properties.Type:
        if not hasattr(self, 'precision') or not hasattr(self, 'feature_mean'):
            raise RuntimeError("Cannot compute Mahalanobis distance without a covariance matrix.")

        if properties.feature in data:
            feats = (data[properties.feature] - self.feature_mean) / self.feature_std
            # Clamp to avoid negative values due to numerical precision issues
            maha_sq = torch.einsum("ij,jk,ik->i", feats, self.precision, feats).clamp(min=0.0)
            maha_dist = torch.sqrt(maha_sq)
            if 'local' in self.kernel:
                maha_dist = scatter_mean(maha_dist, data[properties.image_idx], dim=0)
            data[properties.maha_dist] = maha_dist
        return data

    def _compute_feature(self, data: properties.Type, predict: bool=False) -> properties.Type:
        if not hasattr(self, 'feature_extractor') or not hasattr(self, 'random_projections'):
            raise RuntimeError("Feature extractor is not initialized; ensure repr_callback is provided.")
        data = self.feature_extractor(data, predict=predict)       # this will modify the data in place if in predict mode
        feats, grads = data[properties.feature], data[properties.gradient]
        if not feats or not grads:
            raise RuntimeError(
                "FeatureExtractor did not capture features/gradients. "
                "Ensure the repr_callback forward/backward runs with hooks attached before calling FeatureCalculator, "
                "or call with predict=True so repr_callback executes within FeatureCalculator."
            )
        in_feat_projs = [getattr(self.random_projections, name) for name in self.random_projections.in_feat_proj_buffers]
        out_grad_projs = [getattr(self.random_projections, name) for name in self.random_projections.out_grad_proj_buffers]
        if not in_feat_projs or not out_grad_projs:
            raise RuntimeError(
                "RandomProjections has no projection buffers; ensure the target layer contains linear-like modules "
                "and n_random_features > 0."
            )

        if 'local' in self.kernel:
            if self.kernel == 'local-full-g':
                atomic_feat = torch.zeros(
                    data[properties.n_atoms].sum().item(),
                    self.n_random_features, 
                    dtype=data[properties.edge_diff].dtype,
                    device=data[properties.edge_diff].device,
                )
                for feat, grad, in_proj, out_proj in zip(
                    feats,
                    grads,
                    in_feat_projs,
                    out_grad_projs,
                ):
                    atomic_feat += (feat @ in_proj) * (grad @ out_proj)
            elif self.kernel == 'local-ll-g':
                if self.n_random_features != 0:
                    atomic_feat = (feats[-1] @ in_feat_projs[-1]) * (grads[-1] @ out_grad_projs[-1])
                else:
                    atomic_feat = feats[-1][:, :-1]    # remove bias
            elif self.kernel == 'local-gnn':
                if self.n_random_features != 0:
                    atomic_feat = (feats[0] @ in_feat_projs[0]) * (grads[0] @ out_grad_projs[0])
                else:
                    atomic_feat = feats[0][:, :-1]    # remove bias

            atoms_feat = atomic_feat
        else:
            if self.kernel == 'full-gradient':
                atoms_feat = torch.zeros(
                    data[properties.n_atoms].shape[0], 
                    self.n_random_features, 
                    dtype=data[properties.positions].dtype,
                    device=data[properties.positions].device,
                )
                for feat, grad, in_proj, out_proj in zip(
                    feats,
                    grads,
                    in_feat_projs,
                    out_grad_projs,
                ):
                    print(feat.shape, grad.shape, in_proj.shape, out_proj.shape)
                    atoms_feat += (feat @ in_proj) * (grad @ out_proj)
            elif self.kernel == 'll-gradient':
                if self.n_random_features != 0:
                    atoms_feat = (feats[-1] @ in_feat_projs[-1]) * (grads[-1] @ out_grad_projs[-1])
                else:
                    atoms_feat = feats[-1][:, :-1]    # remove bias
            elif self.kernel == 'gnn':
                if self.n_random_features != 0:
                    atoms_feat = (feats[0] @ in_feat_projs[0]) * (grads[0] @ out_grad_projs[0])
                else:
                    atoms_feat = feats[0][:, :-1]    # remove bias
        
        data[properties.feature] = atoms_feat

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}(kernel={self.kernel}, n_random_features={self.n_random_features}, '
            f'compute_maha_dist={self.compute_maha_dist})')
