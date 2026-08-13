import torch
from torch import nn


class ResidualAdd(nn.Module):
    """Add a residual update and optionally scale the result."""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        # The scale is constructor configuration, not fitted state. Keeping it
        # non-persistent avoids adding keys to checkpoints created by models
        # that previously expressed residual addition directly in ``forward``.
        self.register_buffer(
            "scale",
            torch.tensor(float(scale)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        return (x + update) * self.scale
