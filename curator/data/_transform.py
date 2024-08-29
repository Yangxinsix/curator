import torch
import abc
from typing import Dict
from . import properties

class Transform(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super().__init__()
    
    @abc.abstractmethod
    def forward(self):
        raise NotImplementedError
    
class UnitTransform(Transform):
    def __init__(
        self,
        unit_dict: Dict[str, float]
    ) -> None:
        super().__init__()
        
        self.unit_dict = unit_dict
    
    def forward(self, data: properties.Type) -> properties.Type:
        for k, v in self.unit_dict:
            data[k] *= v
        
        return data   