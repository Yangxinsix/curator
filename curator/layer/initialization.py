"""Small, reusable parameter-initialization helpers."""

from __future__ import annotations

from torch import nn


def reset_linear(linear: nn.Module) -> None:
    """Apply Xavier-uniform weights and zero bias to a linear-like module."""
    weight = getattr(linear, "weight", None)
    if weight is None:
        raise TypeError(f"{type(linear).__name__} does not expose a weight tensor.")
    nn.init.xavier_uniform_(weight)
    bias = getattr(linear, "bias", None)
    if bias is not None:
        nn.init.zeros_(bias)


__all__ = ["reset_linear"]
