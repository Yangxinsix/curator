import torch
from e3nn.util.jit import compile_mode
from curator.data import properties, TypeMapper
from e3nn import o3
from typing import List, Optional

@compile_mode("script")
class OneHotAtomEncoding(torch.nn.Module):
    """Copmute a one-hot floating point encoding of atoms' discrete atom types.

    Args:
        set_features: If ``True`` (default), ``node_features`` will be set in addition to ``node_attrs``.
    """

    num_elements: int

    def __init__(
        self,
        num_elements: Optional[int] = None,
        species: Optional[List[str]] = None,
        set_features: bool=True,
    ):
        super().__init__()
        self.num_elements = num_elements
        self.set_features = set_features
        self.species = species
        
        if self.species is not None:
            self.type_mapper = TypeMapper(self.species)
            self.num_elements = len(self.species)
            
        # output node feature irreps
        self.irreps_out = {
            properties.node_attr: o3.Irreps([(self.num_elements, (0, 1))])
        }
        if self.set_features:
            self.irreps_out[properties.node_feat] = self.irreps_out[properties.node_attr]
            
    def forward(self, data: properties.Type) -> properties.Type:
        if properties.atomic_types not in data:
            data = self.type_mapper(data)
        onehot = torch.nn.functional.one_hot(
            data[properties.atomic_types], num_classes=self.num_elements
        ).to(device=data[properties.positions].device, dtype=data[properties.positions].dtype)
        
        data[properties.node_attr] = onehot
        if self.set_features:
            data[properties.node_feat] = onehot
        return data
    
    def datamodule(self, _datamodule):
        if self.species is None:
            self.species = _datamodule._get_species()
            self.type_mapper = TypeMapper(self.species)