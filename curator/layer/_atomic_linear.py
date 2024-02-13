from e3nn.o3 import Linear
from e3nn import o3
from e3nn.nn import Activation
from e3nn.util.jit import compile_mode
from typing import Optional, Callable
import torch
from curator.data import properties

class AtomwiseLinear(torch.nn.Module):
    def __init__(
        self,
        irreps_in: Optional[o3.Irreps]=None,
        irreps_out: Optional[o3.Irreps]=None,
        field: str=properties.node_feat,
        out_field: Optional[str]=None,
    ):
        super().__init__()
        self.irreps_in: Optional[o3.Irreps] = irreps_in
        if irreps_out is None:
            irreps_out = irreps_in
        self.irreps_out = irreps_out
        
        self.linear = Linear(
            irreps_in=self.irreps_in, irreps_out=self.irreps_out
        )
        self.field = field
        self.out_field = out_field if out_field is not None else self.field

    def forward(self, data: properties.Type) -> properties.Type:
        data[self.out_field] = self.linear(data[self.field])
        return data


@compile_mode("script") 
class AtomwiseNonLinear(torch.nn.Module):
    def __init__(
        self, 
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        irreps_out: o3.Irreps=o3.Irreps("1x0e"),
    ):
        super().__init__()
        self.MLP_irreps = MLP_irreps
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.MLP_irreps)
        self.non_linearity = Activation(irreps_in=self.MLP_irreps, acts=[gate])
        self.linear_2 = o3.Linear(
            irreps_in=self.MLP_irreps, irreps_out=irreps_out
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 1]