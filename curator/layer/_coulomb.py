from torch import nn
import torch

class CoulombEnergy(nn.Module):
    def __init__(
        self, 
        *args, 
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        