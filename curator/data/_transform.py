import torch

class Transform(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self):
        raise NotImplementedError