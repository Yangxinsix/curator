import torch
from typing import Optional, Dict, Callable, Any
from curator.data import properties
from ._interaction import Interaction
from ._ops import (
    build_convnet_radial_mlp,
    FullyConnectedTensorProduct,
    Linear,
    TensorProduct,
)

from e3nn import o3

try:
    from torch_scatter import scatter_add
except ImportError:
    from curator.utils import scatter_add

class ConvNetLayer(Interaction):
    use_sc: bool

    def __init__(
        self,
        irreps_in,
        irreps_out,
        invariant_layers: Optional[int] = None,
        invariant_neurons: Optional[int] = None,
        radial_mlp_depth: Optional[int] = None,
        radial_mlp_width: Optional[int] = None,
        avg_num_neighbors: Optional[float] = None,
        use_sc: bool = True,
        is_first_layer: bool = False,
        self_connection_field: str = properties.node_attr,
        nonlinearity_scalars: Optional[Dict[int, Callable]] = None,
    ) -> None:
        """
        Convolution Block.

        :param irreps_in: Input irreps, including 
        :param irreps_out: Output irreps, in our case typically a single scalar
        :param radial_layers: Number of radial layers, default = 1
        :param radial_neurons: Number of hidden neurons in radial function, default = 8
        :param avg_num_neighbors: Number of neighbors to divide by, default None => no normalization.
        :param number_of_basis: Number or Basis function, default = 8
        :param irreps_in: Input Features, default = None
        :param use_sc: bool, use self-connection or not
        """
        super().__init__()
        if nonlinearity_scalars is None:
            nonlinearity_scalars = {"e": "ssp"}

        if radial_mlp_depth is None:
            radial_mlp_depth = invariant_layers if invariant_layers is not None else 1
        if radial_mlp_width is None:
            radial_mlp_width = invariant_neurons if invariant_neurons is not None else 128

        if avg_num_neighbors is not None:
            self._initialized = True
            avg_num_neigh = torch.tensor(avg_num_neighbors)
        else:
            self._initialized = False
            avg_num_neigh = torch.tensor(1.0)
        
        # self._initialized = True if avg_num_neighbors is not None else False
        # avg_num_neighbors = torch.ones((1,)) if avg_num_neighbors is None else torch.tensor([avg_num_neighbors])
        self.register_buffer("avg_num_neighbors", avg_num_neigh)
        self.use_sc = use_sc
        self.is_first_layer = is_first_layer
        self.self_connection_field = self_connection_field
        self.radial_mlp_depth = int(radial_mlp_depth)
        self.radial_mlp_width = int(radial_mlp_width)

        feature_irreps_in = irreps_in[properties.node_feat]
        feature_irreps_out = irreps_out
        edge_diff_irreps = irreps_in[properties.edge_diff_embedding]
        edge_dist_irreps = irreps_in[properties.edge_dist_embedding]

        # - Build modules -
        self.linear_1 = Linear(
            irreps_in=feature_irreps_in,
            irreps_out=feature_irreps_in,
            internal_weights=True,
            shared_weights=True,
        )

        irreps_mid = []
        instructions = []

        for i, (mul, ir_in) in enumerate(feature_irreps_in):
            for j, (_, ir_edge) in enumerate(edge_diff_irreps):
                for ir_out in ir_in * ir_edge:
                    if ir_out in feature_irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))

        # We sort the output irreps of the tensor product so that we can simplify them
        # when they are provided to the second o3.Linear
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        # Permute the output indexes of the instructions to match the sorted irreps:
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        tp = TensorProduct(
            feature_irreps_in,
            edge_diff_irreps,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # init_irreps already confirmed that the edge embeddding is all invariant scalars
        self.fc = build_convnet_radial_mlp(
            input_dim=edge_dist_irreps.num_irreps,
            output_dim=tp.weight_numel,
            hidden_layers_depth=self.radial_mlp_depth,
            hidden_layers_width=self.radial_mlp_width,
            nonlinearity=nonlinearity_scalars["e"],
        )

        self.tp = tp

        self.linear_2 = Linear(
            # irreps_mid has uncoallesed irreps because of the uvu instructions,
            # but there's no reason to treat them seperately for the Linear
            # Note that normalization of o3.Linear changes if irreps are coallesed
            # (likely for the better)
            irreps_in=irreps_mid.simplify(),
            irreps_out=feature_irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        self.sc = None
        if self.use_sc:
            self.sc = FullyConnectedTensorProduct(
                feature_irreps_in,
                irreps_in[self.self_connection_field],
                feature_irreps_out,
            )

    def forward(
        self,
        node_feat,
        node_attr,
        sc_attr,
        edge_idx,
        edge_dist_embedding,
        edge_diff_embedding,
        lammps_data: Optional[Any] = None,
        n_local: Optional[int] = None,
        n_ghost: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Evaluate interaction Block with ResNet (self-connection).

        :param node_input:
        :param node_attr:
        :param edge_src:
        :param edge_dst:
        :param edge_attr:
        :param edge_length_embedded:

        :return:
        """
        weight = self.fc(edge_dist_embedding)

        if self.sc is not None:
            sc = self.sc(node_feat, sc_attr)

        node_feat = self.linear_1(node_feat)
        node_feat = self.exchange_info(node_feat, lammps_data, n_ghost)
        edge_features = self.tp(
            node_feat[edge_idx[:, 1]], edge_diff_embedding, weight
        )
        out_feat = torch.zeros(
            (node_feat.shape[0], edge_features.shape[1]),
            device=edge_features.device,
            dtype=edge_features.dtype,
        )
        node_feat = scatter_add(edge_features, edge_idx[:, 0], dim=0, out=out_feat)

        node_feat = self.truncate_ghost(node_feat, n_local)
        # Necessary to get TorchScript to be able to type infer when its not None
        # avg_num_neigh: Optional[float] = self.avg_num_neighbors
        # if avg_num_neigh is not None:
        node_feat = node_feat.div(self.avg_num_neighbors**0.5)

        node_feat = self.linear_2(node_feat)

        if self.sc is not None:
            sc = self.truncate_ghost(sc, n_local)
            node_feat = node_feat + sc

        return node_feat
    
    def datamodule(self, _datamodule):
        if not self._initialized:
            avg_num_neigh = _datamodule._get_avg_num_neighbors()
            if avg_num_neigh is not None:
                self.avg_num_neighbors = torch.tensor(avg_num_neigh)

    def setup_from_datamodule(self, datamodule):
        return self.datamodule(datamodule)

    def setup_from_context(self, ctx):
        if not self._initialized and ctx.avg_num_neighbors is not None:
            self.avg_num_neighbors = torch.tensor(ctx.avg_num_neighbors)
