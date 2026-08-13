from ._ops import Linear
from e3nn import o3
from e3nn.nn import Activation
from e3nn.util.jit import compile_mode
from os import PathLike
from typing import Optional, Callable
import numpy as np
import torch
from curator.data import properties


class FeatureProjection(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        input_key: str = properties.node_final_feature,
        output_key: str = properties.node_feature_distill,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.input_key = input_key
        self.output_key = output_key
        self.model_outputs = [output_key]

    def forward(self, data: properties.Type) -> properties.Type:
        data[self.output_key] = self.linear(data[self.input_key])
        return data


class ProjectedElementEmbedding(torch.nn.Module):
    def __init__(
        self,
        embeddings,
        out_features: int,
        index_offset: int = 0,
    ):
        super().__init__()
        if isinstance(embeddings, (str, PathLike)):
            embeddings = np.load(embeddings)
        embeddings = torch.as_tensor(embeddings).detach().to(torch.get_default_dtype())
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [num_elements, num_features].")
        self.register_buffer("embeddings", embeddings)
        self.projection = torch.nn.Linear(embeddings.shape[1], out_features)
        self.index_offset = int(index_offset)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embeddings[indices + self.index_offset])


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
        self.linear_1 = Linear(irreps_in=irreps_in, irreps_out=self.MLP_irreps)
        self.non_linearity = Activation(irreps_in=self.MLP_irreps, acts=[gate])
        self.linear_2 = Linear(
            irreps_in=self.MLP_irreps, irreps_out=irreps_out
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 1]
