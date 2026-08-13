from typing import Optional

import torch


def safe_norm(
    x: torch.Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute a norm with finite derivatives at zero."""

    if eps == 0.0:
        if dim is None:
            return torch.linalg.vector_norm(x)
        return torch.linalg.vector_norm(x, dim=[dim], keepdim=keepdim)
    squared = x.square()
    if dim is None:
        return torch.sqrt(torch.sum(squared) + eps)
    return torch.sqrt(torch.sum(squared, dim=dim, keepdim=keepdim) + eps)
