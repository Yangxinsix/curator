from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from curator.model.conversion import load_official_nequip_as_curator

from ..registry import ExternalModelSpec, register_adapter_loader


def _load_nequip(spec: ExternalModelSpec, device: Optional[torch.device]) -> nn.Module:
    compile_mode = spec.params.get("compile_mode", "eager")
    return load_official_nequip_as_curator(
        spec.resource,
        device=device or torch.device("cpu"),
        compile_mode=compile_mode,
    )


register_adapter_loader("nequip", _load_nequip)
