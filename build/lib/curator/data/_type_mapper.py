from typing import Optional, List, Dict
from ase.data import atomic_numbers
import warnings
import torch
from . import properties
from ._transform import Transform

class TypeMapper(Transform):
    def __init__(
        self,
        species: Optional[List[str]]=None,
        symbol_to_type: Optional[Dict[str, int]]=None,
    ) -> None:
        super().__init__()
        if species is not None:
            if symbol_to_type is not None:
                raise TypeError("Cannot give both `species` and `symbol_to_type`")
            numbers = [atomic_numbers[s] for s in species]
            # sort chemical species
            species = [e[1] for e in sorted(zip(numbers, species))]
            symbol_to_type = {k: idx for idx, k in enumerate(species)}
        self.symbol_to_type = symbol_to_type
        
        if self.symbol_to_type is not None:
            for sym, type in self.symbol_to_type.items():
                assert sym in atomic_numbers, f"Invalid chemical symbol {sym}"
                assert 0 <= type, f"Invalid type number {type}"
            # 119 elements
            Z_to_index = torch.full(size=(119,), fill_value=-1, dtype=torch.long)
            for sym, type in self.symbol_to_type.items():
                Z_to_index[atomic_numbers[sym]] = type
            index_to_Z = torch.zeros(size=(len(self.symbol_to_type),), dtype=torch.long)
            for sym, type in self.symbol_to_type.items():
                index_to_Z[type] = atomic_numbers[sym]
        
            self.register_buffer("Z_to_index", Z_to_index)
            self.register_buffer("index_to_Z", index_to_Z)
        else:
            raise ValueError("`species` or `symbol_to_type` should be given!")
        
    def forward(self, data: properties.Type) -> properties.Type:
        if properties.atomic_types in data:
            warnings.warn("Data already contains mapped types. This will be overwrited.")
        
        data[properties.atomic_types] = self.transform(data[properties.Z])
        assert torch.all(data[properties.atomic_types] >= 0), "Provided data contains species not defined in TypeMapper!"
        return data
        
    def transform(self, numbers: torch.Tensor) -> torch.Tensor:
        if numbers.max() > 119 or numbers.min() < 1:
            raise ValueError("Provided atomic numbers are not in the periodic table!")
        types = self.Z_to_index[numbers]
        return types
    
    def untransform(self, types: torch.Tensor) -> torch.Tensor:
        return self.index_to_Z[types]
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(symbol_to_type={self.symbol_to_type}"
        )