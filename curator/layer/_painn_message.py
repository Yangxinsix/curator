from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import nn

from ._interaction import Interaction
from ._ops import ScalarLinear

try:
    from torch_scatter import scatter_add
except ImportError:
    from curator.utils import scatter_add


class PainnMessage(Interaction):
    """PaiNN interatomic message block.

    The block returns message deltas when ``resnet=False`` so the representation
    can compose its residual policy without changing the PaiNN message equations.
    """

    def __init__(
        self,
        num_features: int,
        num_basis: int,
        activation: Optional[nn.Module] = None,
        scalar_norm: Optional[nn.Module] = None,
        state_vector_scale: float = 1.0,
        message_vector_scale: float = 1.0,
        resnet: bool = True,
        linear_initializer: Optional[Callable[[nn.Module], None]] = None,
    ) -> None:
        super().__init__()
        self.num_basis = num_basis
        self.num_features = num_features
        self.resnet = resnet
        self.state_vector_scale = float(state_vector_scale)
        self.message_vector_scale = float(message_vector_scale)
        self.scalar_norm = nn.Identity() if scalar_norm is None else scalar_norm
        self.scalar_message_mlp = nn.Sequential(
            ScalarLinear(num_features, num_features, bias=True),
            nn.SiLU() if activation is None else activation,
            ScalarLinear(num_features, num_features * 3, bias=True),
        )
        self.filter_layer = ScalarLinear(num_basis, num_features * 3, bias=True)

        if linear_initializer is not None:
            linear_initializer(self.scalar_message_mlp[0])
            linear_initializer(self.scalar_message_mlp[2])
            linear_initializer(self.filter_layer)

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_idx: torch.Tensor,
        edge_dist: torch.Tensor,
        edge_diff: torch.Tensor,
        edge_embedding: torch.Tensor,
        lammps_data: Optional[Any] = None,
        n_local: Optional[int] = None,
        n_ghost: Optional[int] = None,
    ) -> torch.Tensor:
        filter_weight = self.filter_layer(edge_embedding)
        node_scalar, node_vector = torch.split(
            node_feat,
            [self.num_features, 3 * self.num_features],
            dim=-1,
        )
        node_scalar = self.scalar_message_mlp(self.scalar_norm(node_scalar))

        exchanged = self.exchange_info(
            torch.cat([node_scalar, node_vector], dim=-1),
            lammps_data,
            n_ghost,
        )
        node_scalar, node_vector = exchanged.split(
            [3 * self.num_features, 3 * self.num_features],
            dim=-1,
        )

        filter_out = filter_weight * node_scalar[edge_idx[:, 1]]
        gate_state_vector, gate_edge_vector, message_scalar = torch.split(
            filter_out,
            self.num_features,
            dim=1,
        )
        message_vector = (
            node_vector[edge_idx[:, 1]].reshape(-1, 3, self.num_features)
            * (gate_state_vector * self.state_vector_scale).unsqueeze(1)
        )
        edge_vector = gate_edge_vector.unsqueeze(1) * (
            edge_diff / edge_dist.unsqueeze(-1)
        ).unsqueeze(-1)
        message_vector = (
            (message_vector + edge_vector) * self.message_vector_scale
        ).reshape(-1, 3 * self.num_features)

        residual_scalar = scatter_add(
            message_scalar,
            edge_idx[:, 0],
            dim=0,
            dim_size=node_scalar.shape[0],
        )
        residual_vector = scatter_add(
            message_vector,
            edge_idx[:, 0],
            dim=0,
            dim_size=node_vector.shape[0],
        )
        residual = self.truncate_ghost(
            torch.cat([residual_scalar, residual_vector], dim=-1),
            n_local,
        )
        if not self.resnet:
            return residual
        return self.truncate_ghost(node_feat, n_local) + residual


__all__ = ["PainnMessage"]
