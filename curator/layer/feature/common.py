from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import torch

KernelName = Literal[
    "full-g",
    "ll-g",
    "local-full-g",
    "local_full-g",
    "local-ll-g",
    "local_ll-g",
    "local-gnn",
    "full-gradient",
    "ll-gradient",
    "gnn",
    "local_full-gradient",
    "local_ll-gradient",
    "local_gnn",
]
Reduction = Literal["mean", "sum"]

_DEFAULT_KERNEL = "full-g"
_DEFAULT_N_RANDOM_FEATURES = 500


def normalize_kernel(kernel: KernelName) -> str:
    aliases = {
        "full-g": "full-gradient",
        "ll-g": "ll-gradient",
        "local-full-g": "local_full-gradient",
        "local_full-g": "local_full-gradient",
        "local-ll-g": "local_ll-gradient",
        "local_ll-g": "local_ll-gradient",
        "local-gnn": "local_gnn",
    }
    return aliases.get(kernel, kernel)


@dataclass
class ExtractedFeatures:
    image_idx: torch.Tensor
    feats: List[torch.Tensor]
    grads: List[torch.Tensor]
    atomic_numbers: Optional[torch.Tensor] = None
    num_atoms: Optional[torch.Tensor] = None
