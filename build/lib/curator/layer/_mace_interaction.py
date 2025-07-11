import torch
from torch import nn
from curator.data import properties
from e3nn.util.jit import compile_mode
from e3nn import o3
from ._cuequivariance_wrapper import (
    Linear,
    TensorProduct,
    FullyConnectedTensorProduct,
    SymmetricContractionWrapper,
)
from ._interaction import Interaction
from e3nn.nn import FullyConnectedNet
from .utils import (
    tp_out_irreps_with_instructions,
    linear_out_irreps,
    reshape_irreps,
)
try:
    from torch_scatter import scatter_add
except ImportError:
    from curator.utils import scatter_add
from typing import Optional, Callable, Tuple, Any

@compile_mode("script")
class EquivariantProductBasisBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        correlation: int,
        use_sc: bool = True,
        num_elements: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.use_sc = use_sc
        self.symmetric_contractions = SymmetricContractionWrapper(
            irreps_in=node_feats_irreps,
            irreps_out=target_irreps,
            correlation=correlation,
            num_elements=num_elements,
        )
        # Update linear
        self.linear = Linear(
            target_irreps,
            target_irreps,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        sc: Optional[torch.Tensor],
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        node_feats = self.symmetric_contractions(node_feats, node_attrs)
        if self.use_sc and sc is not None:
            return self.linear(node_feats) + sc

        return self.linear(node_feats)


@compile_mode("script")
class RealAgnosticInteractionBlock(Interaction):
    def __init__(
        self,
        irreps_in,
        target_irreps,
        radial_MLP,
        avg_num_neighbors: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.target_irreps = target_irreps
        self.radial_MLP = radial_MLP
        self._initialized = True if avg_num_neighbors is not None else False
        avg_num_neighbors = torch.ones((1,)) if avg_num_neighbors is None else torch.tensor([avg_num_neighbors])
        self.register_buffer("avg_num_neighbors", avg_num_neighbors)
        
        # First linear
        self.linear_up = Linear(
            self.irreps_in[properties.node_feat],
            self.irreps_in[properties.node_feat],
            internal_weights=True,
            shared_weights=True,
        )
        
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.irreps_in[properties.node_feat],
            self.irreps_in[properties.edge_diff_embedding],
            self.target_irreps,
        )
        
        self.conv_tp = TensorProduct(
            self.irreps_in[properties.node_feat],
            self.irreps_in[properties.edge_diff_embedding],
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        
        # Convolution weights
        input_dim = self.irreps_in[properties.edge_dist_embedding].num_irreps
        self.conv_tp_weights = FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )
        
        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )
        
        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.irreps_in[properties.node_attr], 
            self.irreps_out,
        )
        self.reshape = reshape_irreps(self.irreps_out)

    def forward(
        self,       
        node_feat, 
        node_attr,
        edge_idx, 
        edge_dist_embedding,
        edge_diff_embedding,
        lammps_data: Optional[Any] = None,
        n_local: Optional[int] = None,
        n_ghost: Optional[int] = None,
        is_first_layer: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        node_feat = self.linear_up(node_feat)
        
        tp_weights = self.conv_tp_weights(edge_dist_embedding)
        node_feat = self.exchange_info(node_feat, lammps_data, n_ghost, is_first_layer=is_first_layer)
        edge_feat = self.conv_tp(
            node_feat[edge_idx[:, 0]],
            edge_diff_embedding,
            tp_weights,
        ) # [n_edges, irreps]

        node_feat = scatter_add(edge_feat, edge_idx[:, 1], dim=0) # [n_nodes, irreps]， message
        node_feat = self.truncate_ghost(node_feat, n_local)
        node_attr = self.truncate_ghost(node_attr, n_local)

        node_feat = self.linear(node_feat)
        node_feat = node_feat / self.avg_num_neighbors
        node_feat = self.skip_tp(node_feat, node_attr)
        
        return (self.reshape(node_feat), None)
    
    def datamodule(self, _datamodule):
        if not self._initialized:
            avg_num_neigh = _datamodule._get_avg_num_neighbors()
            if avg_num_neigh is not None:
                self.avg_num_neighbors = torch.tensor([avg_num_neigh])


@compile_mode("script")
class RealAgnosticResidualInteractionBlock(Interaction):
    def __init__(
        self,
        irreps_in, 
        target_irreps,
        hidden_irreps,
        radial_MLP,
        avg_num_neighbors: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.irreps_in = irreps_in
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.radial_MLP = radial_MLP
        self._initialized = True if avg_num_neighbors is not None else False
        avg_num_neighbors = torch.ones((1,)) if avg_num_neighbors is None else torch.tensor([avg_num_neighbors])
        self.register_buffer("avg_num_neighbors", avg_num_neighbors) 

        # First linear
        self.linear_up = Linear(
            self.irreps_in[properties.node_feat],
            self.irreps_in[properties.node_feat],
            internal_weights=True,
            shared_weights=True,
        )
        
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.irreps_in[properties.node_feat],
            self.irreps_in[properties.edge_diff_embedding],
            self.target_irreps,
        )
        
        self.conv_tp = TensorProduct(
            self.irreps_in[properties.node_feat],
            self.irreps_in[properties.edge_diff_embedding],
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.irreps_in[properties.edge_dist_embedding].num_irreps
        self.conv_tp_weights = FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_in[properties.node_feat], 
            self.irreps_in[properties.node_attr],
            self.hidden_irreps,
        )
        self.reshape = reshape_irreps(self.irreps_out)

    def forward(
        self,
        node_feat, 
        node_attr,
        edge_idx, 
        edge_dist_embedding,
        edge_diff_embedding,
        lammps_data: Optional[Any] = None,
        n_local: Optional[int] = None,
        n_ghost: Optional[int] = None,
        is_first_layer: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:      
        sc = self.skip_tp(node_feat, node_attr)
        node_feat = self.linear_up(node_feat)
        tp_weights = self.conv_tp_weights(edge_dist_embedding)

        node_feat = self.exchange_info(node_feat, lammps_data, n_ghost, is_first_layer=is_first_layer)
        edge_feat = self.conv_tp(
            node_feat[edge_idx[:, 0]],
            edge_diff_embedding,
            tp_weights,
        )
        node_feat = scatter_add(edge_feat, edge_idx[:, 1], dim=0)
        node_feat = self.truncate_ghost(node_feat, n_local)
        node_attr = self.truncate_ghost(node_attr, n_local)
        sc = self.truncate_ghost(sc, n_local)
        node_feat = self.linear(node_feat) / self.avg_num_neighbors
        
        return (self.reshape(node_feat), sc)
    
    def datamodule(self, _datamodule):
        if not self._initialized:
            avg_num_neigh = _datamodule._get_avg_num_neighbors()
            if avg_num_neigh is not None:
                self.avg_num_neighbors = torch.tensor([avg_num_neigh])