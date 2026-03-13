from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn

from ..utils import find_layer_by_name_recursive
from .extractor import FeatureExtractor


class FeatureProjector(nn.Module):
    """Base class for feature projection strategies."""

    def __init__(self, num_features: int, target_layer: str = "readout_mlp") -> None:
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
    """Gaussian random projections for compressed gradient features."""

    def __init__(
        self,
        module: nn.Module,
        num_features: int,
        dtype: Optional[torch.dtype] = None,
        target_layer: str = "readout_mlp",
    ) -> None:
        super().__init__(num_features=num_features, target_layer=target_layer)
        self.in_feat_proj_buffers: List[str] = []
        self.out_grad_proj_buffers: List[str] = []
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
                self.register_buffer(f"in_feat_proj_{i}", in_feat_proj)
                self.in_feat_proj_buffers.append(f"in_feat_proj_{i}")

                if hasattr(l, "out_features"):
                    out_dim = l.out_features
                elif hasattr(l, "irreps_out"):
                    out_dim = l.irreps_out.dim
                else:
                    raise AttributeError("Linear-like layer missing output dimension attributes.")

                out_grad_proj = torch.randn(out_dim, self.num_features, dtype=dtype, device=device)
                self.register_buffer(f"out_grad_proj_{i}", out_grad_proj)
                self.out_grad_proj_buffers.append(f"out_grad_proj_{i}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_features={self.num_features})"

    @property
    def in_feat_proj(self) -> List[torch.Tensor]:
        return [getattr(self, name) for name in self.in_feat_proj_buffers]

    @property
    def out_grad_proj(self) -> List[torch.Tensor]:
        return [getattr(self, name) for name in self.out_grad_proj_buffers]
