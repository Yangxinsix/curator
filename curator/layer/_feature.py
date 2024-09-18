from curator.data import properties
from torch import nn
import torch
from typing import Dict, Optional, Callable, List

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
        predict: bool = False,
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
        self.predict = predict
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

    def forward(self, data: properties.Type) -> properties.Type:
        if self.predict:
            data = self.repr_callback(data)
        data[properties.feature] = self._features
        data[properties.gradient] = self._grads[::-1]
        self._features = []
        self._grads = []
                          
        return data
    
    def __repr__(self):
        return f'{self.__class__.__name__}(target_layer={self.target_layer}, predict={self.predict})'

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
            normalization = torch.sqrt(torch.tensor(self.num_features, dtype=dtype, device=device))
            layer = find_layer_by_name_recursive(module, target_layer)
            # Input feature projection matrices (in_features + 1 for bias term)
            for i, l in enumerate(layer.children()):
                if isinstance(l, nn.Linear):
                    in_feat_proj = torch.randn(l.in_features + 1, self.num_features, dtype=dtype, device=device) / normalization
                    # Register the buffer by name
                    self.register_buffer(f'in_feat_proj_{i}', in_feat_proj)
                    self.in_feat_proj_buffers.append(f'in_feat_proj_{i}')  # Store buffer names for access

            # Output gradient projection matrices
            for i, l in enumerate(layer.children()):
                if isinstance(l, nn.Linear):
                    out_grad_proj = torch.randn(l.out_features, self.num_features, dtype=dtype, device=device) / normalization
                    # Register the buffer by name
                    self.register_buffer(f'out_grad_proj_{i}', out_grad_proj)
                    self.out_grad_proj_buffers.append(f'out_grad_proj_{i}')  # Store buffer names for access

    def __repr__(self):
        return f'{self.__class__.__name__}(num_features={self.num_features})'
        
class FeatureCalculator(nn.Module):
    def __init__(
        self,
        repr_callback: Optional[nn.Module] = None,   # which module to extract features from
        kernel: str = 'full-gradient',    # select from full-gradient, ll-gradient, gnn
        n_random_features: int = 500,
        model_outputs = ['feature'],
        predict: bool = False,
        target_layer: str = 'readout_mlp',
    ) -> None:
        super().__init__()
        self.n_random_features = n_random_features
        self.kernel = kernel
        self.model_outputs = model_outputs
        self.repr_callback = repr_callback
        self.predict = predict
        self.target_layer = target_layer

        if self.repr_callback is not None:
            self._initialize_feature_components()

    def _initialize_feature_components(self, repr_callback: Optional[nn.Module] = None):
        repr_callback = repr_callback or self.repr_callback
        self.feature_extractor = FeatureExtractor(repr_callback, target_layer=self.target_layer, predict=self.predict)
        self.random_projections = RandomProjections(repr_callback, self.n_random_features, target_layer=self.target_layer)
    
    def register_repr_callback(self, repr_callback: Optional[nn.Module] = None):
        self._initialize_feature_components(repr_callback)

    def forward(self, data: properties.Type) -> properties.Type:
        data = self.feature_extractor(data)
        feats, grads = data[properties.feature], data[properties.gradient]
        in_feat_projs = [getattr(self.random_projections, name) for name in self.random_projections.in_feat_proj_buffers]
        out_grad_projs = [getattr(self.random_projections, name) for name in self.random_projections.out_grad_proj_buffers]
        atomic_feat = torch.zeros(
            data[properties.image_idx].shape[0], 
            self.n_random_features, 
            dtype=data[properties.positions].dtype,
            device=data[properties.positions].device,
        )

        if self.kernel == 'full-gradient':
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

        atoms_feat = torch.zeros(
            (data[properties.n_atoms].shape[0], atomic_feat.shape[1]),
            dtype = atomic_feat.dtype,
            device = atomic_feat.device,
        ).index_add(0, data[properties.image_idx], atomic_feat)
        
        data[properties.feature] = atoms_feat

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(kernel={self.kernel}, n_random_features={self.n_random_features})'