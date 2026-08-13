"""Reusable gated scalar-vector equivariant layers."""

from typing import Callable, Optional, Tuple

import torch
from torch import nn

from .norm import safe_norm


class GatedEquivariantBlock(nn.Module):
    """Mix invariant scalars and equivariant vectors with scalar gates.

    The two vector projections deliberately have separate parameters:
    one produces rotation-invariant vector norms for the scalar network,
    while the other produces the equivariant output directions.
    """

    def __init__(
        self,
        scalar_in: int,
        vector_in: int,
        scalar_out: int,
        vector_out: int,
        invariant_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        activation: Optional[nn.Module] = None,
        scalar_activation: Optional[nn.Module] = None,
        norm_eps: float = 1e-8,
        linear_initializer: Optional[Callable[[nn.Module], None]] = None,
    ) -> None:
        super().__init__()
        if min(scalar_in, vector_in, scalar_out, vector_out) <= 0:
            raise ValueError("Input and output channel counts must be positive.")

        invariant_channels = (
            vector_out if invariant_channels is None else int(invariant_channels)
        )
        hidden_channels = (
            scalar_out + vector_out
            if hidden_channels is None
            else int(hidden_channels)
        )
        if invariant_channels <= 0 or hidden_channels <= 0:
            raise ValueError("Invariant and hidden channel counts must be positive.")

        self.scalar_out = scalar_out
        self.vector_out = vector_out
        self.norm_eps = float(norm_eps)
        self.invariant_vector_projection = nn.Linear(
            vector_in,
            invariant_channels,
            bias=False,
        )
        self.output_vector_projection = nn.Linear(
            vector_in,
            vector_out,
            bias=False,
        )
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_in + invariant_channels, hidden_channels),
            nn.SiLU() if activation is None else activation,
            nn.Linear(hidden_channels, scalar_out + vector_out),
        )
        self.scalar_activation = (
            nn.Identity() if scalar_activation is None else scalar_activation
        )
        if linear_initializer is not None:
            linear_initializer(self.invariant_vector_projection)
            linear_initializer(self.output_vector_projection)
            linear_initializer(self.scalar_net[0])
            linear_initializer(self.scalar_net[2])

    def forward(
        self,
        scalars: torch.Tensor,
        vectors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        invariant_vectors = self.invariant_vector_projection(vectors)
        output_vectors = self.output_vector_projection(vectors)
        invariants = safe_norm(
            invariant_vectors,
            dim=1,
            eps=self.norm_eps,
        )
        scalar_vector_gates = self.scalar_net(
            torch.cat((scalars, invariants), dim=-1)
        )
        scalar_out, gates = scalar_vector_gates.split(
            (self.scalar_out, self.vector_out),
            dim=-1,
        )
        return (
            self.scalar_activation(scalar_out),
            output_vectors * gates.unsqueeze(1),
        )


__all__ = ["GatedEquivariantBlock"]
