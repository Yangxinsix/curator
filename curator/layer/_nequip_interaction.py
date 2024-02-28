# The implementation is revised based on https://github.com/mir-group/nequip/tree/main
# Some code was directly copied from original nequip model

import torch
from typing import Dict, Callable
from curator.data import properties

from e3nn.nn import Gate, NormActivation
from e3nn import o3

from ._convnet import ConvNetLayer
from .nonlinearities import ShiftedSoftPlus
from .utils import tp_path_exists



acts = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "ssp": ShiftedSoftPlus,
    "silu": torch.nn.functional.silu,
}

class InteractionLayer(torch.nn.Module):
    """
    Args:

    """

    resnet: bool

    def __init__(
        self,
        irreps_in,
        feature_irreps_hidden,
        convolution=ConvNetLayer,
        convolution_kwargs: dict = {},
        resnet: bool = False,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp", "o": "tanh"},
        nonlinearity_gates: Dict[int, Callable] = {"e": "ssp", "o": "abs"},
    ):
        super().__init__()
        # initialization
        assert nonlinearity_type in ("gate", "norm")
        # make the nonlin dicts from parity ints instead of convinience strs
        nonlinearity_scalars_dict = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }
        nonlinearity_gates_dict = {
            1: nonlinearity_gates["e"],
            -1: nonlinearity_gates["o"],
        }

        self.feature_irreps_hidden = o3.Irreps(feature_irreps_hidden)
        self.resnet = resnet
        self.irreps_out = irreps_in.copy()

        self.irreps_in = irreps_in
        edge_diff_irreps = self.irreps_in[properties.edge_diff_embedding]
        irreps_layer_out_prev = self.irreps_in[properties.node_feat]

        irreps_scalars = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.feature_irreps_hidden
                if ir.l == 0
                and tp_path_exists(irreps_layer_out_prev, edge_diff_irreps, ir)
            ]
        )

        irreps_gated = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.feature_irreps_hidden
                if ir.l > 0
                and tp_path_exists(irreps_layer_out_prev, edge_diff_irreps, ir)
            ]
        )

        irreps_layer_out = (irreps_scalars + irreps_gated).simplify()

        if nonlinearity_type == "gate":
            ir = (
                "0e"
                if tp_path_exists(irreps_layer_out_prev, edge_diff_irreps, "0e")
                else "0o"
            )
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            # TO DO, it's not that safe to directly use the
            # dictionary
            equivariant_nonlin = Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=[
                    acts[nonlinearity_scalars_dict[ir.p]] for _, ir in irreps_scalars
                ],
                irreps_gates=irreps_gates,
                act_gates=[acts[nonlinearity_gates_dict[ir.p]] for _, ir in irreps_gates],
                irreps_gated=irreps_gated,
            )

            conv_irreps_out = equivariant_nonlin.irreps_in.simplify()

        else:
            conv_irreps_out = irreps_layer_out.simplify()

            equivariant_nonlin = NormActivation(
                irreps_in=conv_irreps_out,
                # norm is an even scalar, so use nonlinearity_scalars[1]
                scalar_nonlinearity=acts[nonlinearity_scalars_dict[1]],
                normalize=True,
                epsilon=1e-8,
                bias=False,
            )

        self.equivariant_nonlin = equivariant_nonlin

        if irreps_layer_out == irreps_layer_out_prev and resnet:
            self.resnet = True
        else:
            self.resnet = False

        # # TODO: last convolution should go to explicit irreps out
        # logging.debug(
        #     f" parameters used to initialize {convolution.__name__}={convolution_kwargs}"
        # )

        # override defaults for irreps:
        convolution_kwargs.pop("irreps_in", None)
        convolution_kwargs.pop("irreps_out", None)
        self.conv = convolution(
            irreps_in=self.irreps_in,
            irreps_out=conv_irreps_out,
            nonlinearity_scalars=nonlinearity_scalars,
            **convolution_kwargs,
        )
        # output node feature irreps
        self.irreps_out[properties.node_feat] = self.equivariant_nonlin.irreps_out

    def forward(self, data: properties.Type) -> properties.Type:
        # save old features for resnet
        old_node_feat = data[properties.node_feat]
        # run convolution
        data = self.conv(data)
        # do nonlinearity
        data[properties.node_feat] = self.equivariant_nonlin(data[properties.node_feat])
        if self.resnet:
            data[properties.node_feat] += old_node_feat
        return data