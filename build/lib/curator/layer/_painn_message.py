import torch
from torch import nn
from ._interaction import Interaction
from .cutoff import CosineCutoff
from .radial_basis import SineBasis
from typing import Optional, Any
from curator.data import properties
try:
    from torch_scatter import scatter_add
except ImportError:
    from curator.utils import scatter_add

class PainnMessage(Interaction):
    """Message function"""
    def __init__(
        self, 
        num_features: int,
        num_basis: int,
        cutoff: float,
        cutoff_fn: Optional[nn.Module] = None,
        radial_basis: Optional[nn.Module] = None,
        resnet: bool = True,
    ):
        super().__init__()
        
        self.num_basis = num_basis
        self.num_features = num_features
        self.cutoff = cutoff
        self.resnet = resnet
        
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
    def forward(
        self, 
        node_feat,
        edge_idx,
        edge_dist,
        edge_diff,
        lammps_data: Optional[Any] = None,
        n_local: Optional[int] = None,
        n_ghost: Optional[int] = None,
    ) -> properties.Type:
        # edge distance embedding
        filter_weight = self.filter_layer(self.radial_basis(edge_dist))
        filter_weight = filter_weight * self.cutoff_fn(edge_dist).unsqueeze(-1)

        node_scalar, node_vector = torch.split(node_feat, [self.num_features * 1, self.num_features * 3], dim=-1)
        node_scalar = node_scalar.clone()
        node_scalar = self.scalar_message_mlp(node_scalar)

        # exchange node features between processors
        new_node_feat = torch.cat([node_scalar, node_vector], dim=-1)
        new_node_feat = self.exchange_info(new_node_feat, lammps_data, n_ghost)
        node_scalar, node_vector = new_node_feat.split([self.num_features * 3, self.num_features * 3], dim=-1)

        filter_out = filter_weight * node_scalar[edge_idx[:, 1]]
        gate_state_vector, gate_edge_vector, message_scalar = torch.split(
            filter_out, 
            self.num_features,
            dim=1,
        )
        
        # num_pairs * 3 * num_features, num_pairs * num_features
        message_vector = node_vector[edge_idx[:, 1]].reshape(-1, 3, self.num_features) * gate_state_vector.unsqueeze(1) 
        edge_vector = gate_edge_vector.unsqueeze(1) * (edge_diff / edge_dist.unsqueeze(-1)).unsqueeze(-1)
        message_vector = (message_vector + edge_vector).reshape(-1, self.num_features * 3)
        
        # sum message
        residual_scalar = scatter_add(message_scalar, edge_idx[:, 0], dim=0)
        residual_vector = scatter_add(message_vector, edge_idx[:, 0], dim=0)
        
        # new node state
        residual_node_feat = torch.cat([residual_scalar, residual_vector], dim=-1)
        residual_node_feat = self.truncate_ghost(residual_node_feat, n_local)
        if not self.resnet:
            return residual_node_feat

        node_feat = self.truncate_ghost(node_feat, n_local)
        node_feat += residual_node_feat
        return node_feat