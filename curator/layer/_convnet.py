import torch
from typing import Optional, Dict, Callable
from curator.data import properties

from ._cuequivariance_wrapper import (
    Linear,
    TensorProduct,
    FullyConnectedTensorProduct,
)

from e3nn import o3
from e3nn.nn import FullyConnectedNet

try:
    from torch_scatter import scatter_add
except ImportError:
    from curator.utils import scatter_add
from .nonlinearities import ShiftedSoftPlus

class ConvNetLayer(torch.nn.Module):
    use_sc: bool

    def __init__(
        self,
        irreps_in,
        irreps_out,
        invariant_layers: int=1,
        invariant_neurons: int=8,
        avg_num_neighbors: Optional[float]=None,
        use_sc: bool=True,
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp"},
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

        if avg_num_neighbors is not None:
            self._initialized = True
            avg_num_neigh = torch.tensor([avg_num_neighbors])
        else:
            self._initialized = False
            avg_num_neigh = torch.ones((1,))
        
        # self._initialized = True if avg_num_neighbors is not None else False
        # avg_num_neighbors = torch.ones((1,)) if avg_num_neighbors is None else torch.tensor([avg_num_neighbors])
        self.register_buffer("avg_num_neighbors", avg_num_neigh)
        self.use_sc = use_sc

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
        self.fc = FullyConnectedNet(
            [edge_dist_irreps.num_irreps]
            + invariant_layers * [invariant_neurons]
            + [tp.weight_numel],
            {
                "ssp": ShiftedSoftPlus,
                "silu": torch.nn.functional.silu,
            }[nonlinearity_scalars["e"]],
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
                irreps_in[properties.node_attr],
                feature_irreps_out,
            )

    def forward(self, data: properties.Type) -> properties.Type:
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
        weight = self.fc(data[properties.edge_dist_embedding])

        x = data[properties.node_feat]
        edge_idx = data[properties.edge_idx]  # i, j index

        if self.sc is not None:
            sc = self.sc(x, data[properties.node_attr])

        x = self.linear_1(x)
        edge_features = self.tp(
            x[edge_idx[:, 1]], data[properties.edge_diff_embedding], weight
        )
        x = scatter_add(edge_features, edge_idx[:, 0], dim=0)

        # Necessary to get TorchScript to be able to type infer when its not None
        # avg_num_neigh: Optional[float] = self.avg_num_neighbors
        # if avg_num_neigh is not None:
        x = x.div(self.avg_num_neighbors**0.5)

        x = self.linear_2(x)

        if self.sc is not None:
            x = x + sc

        data[properties.node_feat] = x
        return data
    
    def datamodule(self, _datamodule):
        if not self._initialized:
            avg_num_neigh = _datamodule._get_avg_num_neighbors()
            if avg_num_neigh is not None:
                self.avg_num_neighbors = torch.tensor([avg_num_neigh])