from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn

from ._ops import ScalarLinear
from .norm import safe_norm


class PainnUpdate(nn.Module):
    """PaiNN intraatomic scalar/vector mixing block."""

    def __init__(
        self,
        num_features: int,
        activation: Optional[nn.Module] = None,
        inner_product_scale: float = 1.0,
        scalar_update_scale: float = 1.0,
        eps: float = 1e-8,
        vector_bias: bool = False,
        scalar_first: bool = False,
        resnet: bool = True,
        linear_initializer: Optional[Callable[[nn.Module], None]] = None,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.inner_product_scale = float(inner_product_scale)
        self.scalar_update_scale = float(scalar_update_scale)
        self.eps = float(eps)
        self.resnet = resnet
        self.scalar_first = bool(scalar_first)

        self.update_U = ScalarLinear(
            num_features,
            num_features,
            bias=vector_bias,
        )
        self.update_V = ScalarLinear(
            num_features,
            num_features,
            bias=vector_bias,
        )
        self.update_mlp = nn.Sequential(
            ScalarLinear(2 * num_features, num_features, bias=True),
            nn.SiLU() if activation is None else activation,
            ScalarLinear(num_features, 3 * num_features, bias=True),
        )

        if linear_initializer is not None:
            linear_initializer(self.update_U)
            linear_initializer(self.update_V)
            linear_initializer(self.update_mlp[0])
            linear_initializer(self.update_mlp[2])

    def forward(self, node_feat: torch.Tensor) -> torch.Tensor:
        node_scalar, node_vector = torch.split(
            node_feat,
            [self.num_features, 3 * self.num_features],
            dim=1,
        )
        node_vector = node_vector.reshape(-1, 3, self.num_features)

        Uv = self.update_U(node_vector)
        Vv = self.update_V(node_vector)
        vector_norm = safe_norm(Vv, dim=1, eps=self.eps)
        update_input = (
            torch.cat((node_scalar, vector_norm), dim=1)
            if self.scalar_first
            else torch.cat((vector_norm, node_scalar), dim=1)
        )
        mlp_output = self.update_mlp(update_input)
        a_vv, a_sv, a_ss = torch.split(
            mlp_output,
            self.num_features,
            dim=1,
        )

        delta_v = a_vv.unsqueeze(1) * Uv
        inner_product = (
            torch.sum(Uv * Vv, dim=1) * self.inner_product_scale
        )
        delta_s = (
            a_sv * inner_product + a_ss
        ) * self.scalar_update_scale
        residual = torch.cat(
            [delta_s, delta_v.reshape(-1, 3 * self.num_features)],
            dim=1,
        )
        return node_feat + residual if self.resnet else residual


__all__ = ["PainnUpdate"]
