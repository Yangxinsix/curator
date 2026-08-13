import torch
from torch import nn


class ScaledSiLU(nn.Module):
    """SiLU with a fixed multiplicative output scale."""

    def __init__(self, scale: float = 1.0 / 0.6):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(x) * self.scale
