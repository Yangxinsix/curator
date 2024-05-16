import torch
import abc

class Transform(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super().__init__()
    
    @abc.abstractmethod
    def forward(self):
        raise NotImplementedError