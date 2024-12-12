import torch
from torch import nn
from .cutoff import CosineCutoff
from .radial_basis import SineBasis
from typing import Optional
from curator.data import properties
try:
    from torch_scatter import scatter_add
except ImportError:
    from curator.utils import scatter_add

class PainnMessage(nn.Module):
    """Message function"""
    def __init__(
        self, 
        num_features: int,
        num_basis: int,
        cutoff: float,
        cutoff_fn: Optional[nn.Module] = None,
        radial_basis: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.num_basis = num_basis
        self.num_features = num_features
        self.cutoff = cutoff
        
        if cutoff_fn is None:
            cutoff_fn = CosineCutoff(cutoff=self.cutoff)
        if radial_basis is None:
            radial_basis = SineBasis(cutoff=self.cutoff, num_basis=self.num_basis)
        self.cutoff_fn = cutoff_fn
        self.radial_basis = radial_basis
        
        self.scalar_message_mlp = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 3),
        )
        
        self.filter_layer = nn.Linear(num_basis, num_features * 3)
        
    # def forward(self, node_scalar, node_vector, edge, edge_diff, edge_dist):
    def forward(self, data: properties.Type) -> properties.Type:
        # remember to use v_j, s_j but not v_i, s_i
        edge = data[properties.edge_idx]
        edge_dist = data[properties.edge_dist]
        edge_diff = data[properties.edge_diff]
        node_scalar = data[properties.node_feat]
        node_vector = data[properties.node_vect]
        
        filter_weight = self.filter_layer(self.radial_basis(edge_dist))
        filter_weight = filter_weight * self.cutoff_fn(edge_dist).unsqueeze(-1)
        scalar_out = self.scalar_message_mlp(node_scalar)        
        filter_out = filter_weight * scalar_out[edge[:, 1]]
        
        gate_state_vector, gate_edge_vector, message_scalar = torch.split(
            filter_out, 
            self.num_features,
            dim = 1,
        )
        
        # num_pairs * 3 * num_features, num_pairs * num_features
        message_vector =  data[properties.node_vect][edge[:, 1]] * gate_state_vector.unsqueeze(1) 
        edge_vector = gate_edge_vector.unsqueeze(1) * (edge_diff / edge_dist.unsqueeze(-1)).unsqueeze(-1)
        message_vector = message_vector + edge_vector
        
        # sum message
        residual_scalar = torch.zeros_like(node_scalar)
        residual_vector = torch.zeros_like(node_vector)
        message_scalar = message_scalar.to(residual_scalar.dtype)
        message_vector = message_vector.to(residual_vector.dtype)

        residual_scalar.index_add_(0, edge[:, 0], message_scalar)
        residual_vector.index_add_(0, edge[:, 0], message_vector)
        
        # new node state
        data[properties.node_feat] = node_scalar + residual_scalar
        data[properties.node_vect] = node_vector + residual_vector
        
        return data