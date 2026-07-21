from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Optional
from urllib.parse import parse_qs, urlencode
import sys

import torch
from torch import nn


@dataclass
class ExternalModelSpec:
    scheme: str
    resource: str
    params: Dict[str, str]


_ADAPTER_LOADERS: Dict[str, Callable[[ExternalModelSpec, Optional[torch.device]], nn.Module]] = {}


def ensure_local_nequip_source_on_path() -> None:
    nequip_src = Path.home() / "local" / "src" / "nequip"
    nequip_src_str = str(nequip_src)
    if nequip_src.exists() and nequip_src_str not in sys.path:
        sys.path.insert(0, nequip_src_str)


def parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


def parse_external_model_spec(raw: str) -> Optional[ExternalModelSpec]:
    if not isinstance(raw, str) or ":" not in raw:
        return None
    scheme, rest = raw.split(":", 1)
    if not scheme:
        return None
    scheme = scheme.strip().lower()
    resource, sep, query = rest.partition("?")
    resource = resource.strip()
    if not resource:
        return None
    params: Dict[str, str] = {}
    if sep:
        parsed = parse_qs(query, keep_blank_values=True)
        params = {k: v[-1] for k, v in parsed.items() if v}
    return ExternalModelSpec(scheme=scheme, resource=resource, params=params)


def register_adapter_loader(
    scheme: str,
    loader: Callable[[ExternalModelSpec, Optional[torch.device]], nn.Module],
) -> None:
    _ADAPTER_LOADERS[scheme.lower()] = loader


def is_external_model_spec(raw: str) -> bool:
    spec = parse_external_model_spec(raw)
    return spec is not None and spec.scheme in _ADAPTER_LOADERS


def format_external_model_spec(spec: ExternalModelSpec) -> str:
    query = urlencode(sorted(spec.params.items()))
    if query:
        return f"{spec.scheme}:{spec.resource}?{query}"
    return f"{spec.scheme}:{spec.resource}"


def load_external_model(raw: str, device: Optional[torch.device] = None) -> nn.Module:
    spec = parse_external_model_spec(raw)
    if spec is None:
        raise ValueError(f"Invalid external model spec: {raw}")
    loader = _ADAPTER_LOADERS.get(spec.scheme)
    if loader is None:
        known = ", ".join(sorted(_ADAPTER_LOADERS.keys()))
        raise ValueError(
            f"Unsupported external model scheme '{spec.scheme}'. Known schemes: {known}"
        )
    return loader(spec, device=device)


def build_representation(cutoff: float) -> SimpleNamespace:
    return SimpleNamespace(cutoff=float(cutoff))

__all__ = [
    "ExternalModelSpec",
    "ensure_local_nequip_source_on_path",
    "parse_bool",
    "parse_external_model_spec",
    "format_external_model_spec",
    "register_adapter_loader",
    "is_external_model_spec",
    "load_external_model",
    "build_representation",
]
