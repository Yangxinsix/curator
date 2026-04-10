from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn

from curator.model.conversion import load_official_mace_as_curator

from .utils import ExternalModelSpec, register_adapter_loader


def _parse_head(value: Optional[str]) -> Optional[Union[str, int]]:
    if value is None:
        return None
    token = str(value).strip()
    if token == "":
        return None
    try:
        return int(token)
    except ValueError:
        return token


def _load_mace(spec: ExternalModelSpec, device: Optional[torch.device]) -> nn.Module:
    return load_official_mace_as_curator(
        spec.resource,
        head=_parse_head(spec.params.get("head")),
        device=device or torch.device("cpu"),
    )


register_adapter_loader("mace", _load_mace)

__all__ = ["_load_mace"]
