from curator.data import properties
from curator.utils import scatter_add, scatter_mean
from torch import nn
import torch
from typing import Dict, Optional, Callable, List, Union
import logging

logger = logging.getLogger(__name__)

def find_layer_by_name_recursive(module, target_name):
    """
    Recursively search for a layer with the specified name within a PyTorch module.
    
    Args:
        module (nn.Module): The module to search in.
        target_name (str): The name of the layer to search for.
    
    Returns:
        nn.Module or None: The layer with the specified name if found, else None.
    """
    # Check if the current module has a direct submodule with the target name
    if hasattr(module, target_name):
        return getattr(module, target_name)

    # Recursively check all child modules
    for child_module in module.children():
        found_module = find_layer_by_name_recursive(child_module, target_name)
        if found_module is not None:
            return found_module

    # If not found, return None
    return None

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
        for child in layer.children():
            if isinstance(child, nn.Linear):
                self.hooks.append(child.register_forward_pre_hook(self.save_feats_hook))
                self.hooks.append(child.register_backward_hook(self.save_grads_hook))

    def forward(self, data: properties.Type, predict: bool=False) -> properties.Type:
        if predict:
            data = self.repr_callback(data)
        data[properties.feature] = self._features
        data[properties.gradient] = self._grads[::-1]
        self._features = []
        self._grads = []
                          
        return data
    
    def __repr__(self):
        return f'{self.__class__.__name__}(target_layer={self.target_layer})'

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

        if self.num_features > 0:
            # Calculate normalization constant once
            # normalization = torch.sqrt(torch.tensor(self.num_features, dtype=dtype, device=device))
            layer = find_layer_by_name_recursive(module, target_layer)
            # Input feature projection matrices (in_features + 1 for bias term)
            for i, l in enumerate(layer.children()):
                if isinstance(l, nn.Linear):
                    in_feat_proj = torch.randn(l.in_features + 1, self.num_features, dtype=dtype, device=device)
                    # Register the buffer by name
                    self.register_buffer(f'in_feat_proj_{i}', in_feat_proj)
                    self.in_feat_proj_buffers.append(f'in_feat_proj_{i}')  # Store buffer names for access

            # Output gradient projection matrices
            for i, l in enumerate(layer.children()):
                if isinstance(l, nn.Linear):
                    out_grad_proj = torch.randn(l.out_features, self.num_features, dtype=dtype, device=device)
                    # Register the buffer by name
                    self.register_buffer(f'out_grad_proj_{i}', out_grad_proj)
                    self.out_grad_proj_buffers.append(f'out_grad_proj_{i}')  # Store buffer names for access

    def __repr__(self):
        return f'{self.__class__.__name__}(num_features={self.num_features})'
        
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
    ) -> None:
        super().__init__()
        self.n_random_features = n_random_features
        self.kernel = kernel
        self.model_outputs = model_outputs
        self.repr_callback = repr_callback
        self.target_layer = target_layer
        self.dataset = dataset
        self.compute_maha_dist = compute_maha_dist

        if self.repr_callback is not None:
            self._initialize_feature_components()

            if self.compute_maha_dist:
                assert self.dataset is not None or hasattr(self, 'precision'), "Mahalanobis distance can not be calculated without precision matrix or provided reference dataset."
                self.get_covariance_matrix(precision, feature_mean, dataset)

    def _initialize_feature_components(self, repr_callback: Optional[nn.Module] = None):
        repr_callback = repr_callback or self.repr_callback
        self.feature_extractor = FeatureExtractor(repr_callback, target_layer=self.target_layer)
        self.random_projections = RandomProjections(repr_callback, self.n_random_features, target_layer=self.target_layer)
    
    def register_repr_callback(self, repr_callback: Optional[nn.Module] = None):
        self._initialize_feature_components(repr_callback)
        if self.compute_maha_dist:
            assert self.dataset is not None or hasattr(self, 'precision'), "Mahalanobis distance can not be calculated without precision matrix or provided reference dataset."
            self.get_covariance_matrix()

    def forward(self, data: properties.Type, predict: bool=False) -> properties.Type:
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
        if precision is not None and feature_mean is not None:
            logger.info('Loading precision matrix and feature mean from provided values.')
            self.register_buffer('precision', precision)
            self.register_buffer('feature_mean', feature_mean)
            return
            
        if dataset is None:
            dataset = self.dataset
        if isinstance(dataset, str):
            from curator.data import AseDataset
            logger.info(f'Calculating features from provided dataset <{dataset}>.')
            dataset = AseDataset(dataset, cutoff=find_layer_by_name_recursive(self.repr_callback, 'cutoff'))
            logger.info(f'Calculating precision matrix from {len(dataset)} structures.')

        # collect features
        if hasattr(self.repr_callback, 'model_outputs'):
            self.repr_callback.model_outputs.append('all')
        features = []
        image_idx = []
        device = next(self.repr_callback.parameters()).device
        for i, sample in enumerate(dataset):
            sample = {k: v.to(device) for k, v in sample.items()}
            features.append(self._compute_feature(sample, predict=True)[properties.feature].to('cpu'))   # use cpu to save memory
            image_idx.append(torch.ones(sample[properties.n_atoms], dtype=torch.long) * i)
        if hasattr(self.repr_callback, 'model_outputs'):
            self.repr_callback.model_outputs.remove('all')

        # calculate inverse covariance matrix
        features = torch.cat(features)
        image_idx = torch.cat(image_idx)
        # normalization for numerical stability
        mean = features.mean(dim=0)
        std = features.std(dim=0)
        features = (features - mean) / std
        cov_matrix = torch.cov(features.T)
        precision = torch.inverse(cov_matrix + torch.eye(cov_matrix.size(0)) * 1e-3) # add a regularization term

        # calculate 95th percentile for uncertainty threshold
        maha_dist = torch.sqrt(torch.einsum("ij,jk,ik->i", features, precision, features))
        maha_dist = scatter_mean(maha_dist, image_idx, dim=0)

        self.register_buffer('feature_mean', mean)
        self.register_buffer('feature_std', std)
        self.register_buffer('cov_matrix', cov_matrix)
        self.register_buffer('precision', precision)
        self.register_buffer('maha_dist', maha_dist)
    
    def mahalanobis_distance(self, data: properties.Type) -> properties.Type:
        if properties.feature in data:
            feats = (data[properties.feature] - self.feature_mean) / self.feature_std
            maha_dist = torch.sqrt(torch.einsum("ij,jk,ik->i", feats, self.precision, feats))
            if 'local' in self.kernel:
                maha_dist = scatter_mean(maha_dist, data[properties.image_idx], dim=0)
            data[properties.maha_dist] = maha_dist
        return data

    def _compute_feature(self, data: properties.Type, predict: bool=False) -> properties.Type:
        data = self.feature_extractor(data, predict=predict)
        feats, grads = data[properties.feature], data[properties.gradient]
        in_feat_projs = [getattr(self.random_projections, name) for name in self.random_projections.in_feat_proj_buffers]
        out_grad_projs = [getattr(self.random_projections, name) for name in self.random_projections.out_grad_proj_buffers]

        if 'local' not in self.kernel:
            if self.kernel == 'full-gradient':
                atomic_feat = torch.zeros(
                    data[properties.image_idx].shape[0], 
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
                    atomic_feat += (feat @ in_proj) * (grad @ out_proj)
            elif self.kernel == 'll-gradient':
                if self.n_random_features != 0:
                    atomic_feat = (feats[-1] @ in_feat_projs[-1]) * (grads[-1] @ out_grad_projs[-1])
                else:
                    atomic_feat = feats[-1][:, :-1]    # remove bias
            elif self.kernel == 'gnn':
                if self.n_random_features != 0:
                    atomic_feat = (feats[0] @ in_feat_projs[0]) * (grads[0] @ out_grad_projs[0])
                else:
                    atomic_feat = feats[0][:, :-1]    # remove bias

            atoms_feat = scatter_add(atomic_feat, data[properties.image_idx], 0)
        else:
            if self.kernel == 'local-full-g':
                atoms_feat = torch.zeros(
                    data[properties.image_idx].shape[0], 
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
                    atoms_feat += (feat @ in_proj) * (grad @ out_proj)
            elif self.kernel == 'local-ll-g':
                if self.n_random_features != 0:
                    atoms_feat = (feats[-1] @ in_feat_projs[-1]) * (grads[-1] @ out_grad_projs[-1])
                else:
                    atoms_feat = feats[-1][:, :-1]    # remove bias
            elif self.kernel == 'local-gnn':
                if self.n_random_features != 0:
                    atoms_feat = (feats[0] @ in_feat_projs[0]) * (grads[0] @ out_grad_projs[0])
                else:
                    atoms_feat = feats[0][:, :-1]    # remove bias
        
        data[properties.feature] = atoms_feat

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}(kernel={self.kernel}, n_random_features={self.n_random_features}, '
            f'compute_maha_dist={self.compute_maha_dist})')