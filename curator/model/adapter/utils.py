from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, Optional
from urllib.parse import parse_qs
import sys

import torch
from torch import nn

from curator.layer.utils import find_layer_by_name_recursive
from curator.model.external_utils import (
    BatchStructure,
    batch_to_atoms,
    build_atomic_number_to_type_map,
    extract_cells,
    extract_pbc,
    infer_cell_offsets,
    map_atomic_numbers_to_types,
    normalize_edge_index,
    require_batch_fields,
    split_batch_structures,
)


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


def resolve_target_layer(
    model: nn.Module,
    target_layer: str,
    fallbacks: Iterable[str] = (),
) -> nn.Module:
    layer = find_layer_by_name_recursive(model, target_layer)
    if layer is not None:
        return layer
    for fallback in fallbacks:
        layer = find_layer_by_name_recursive(model, fallback)
        if layer is not None:
            return layer
    known = [target_layer, *fallbacks]
    raise ValueError(f"Cannot find target layer among: {known}")


def bind_target_layer_aliases(adapter: nn.Module, target_layer: str, module: nn.Module) -> None:
    adapter.readout_mlp = module
    adapter.final_layer = module
    if target_layer not in {"readout_mlp", "final_layer"}:
        setattr(adapter, target_layer, module)


def build_representation(cutoff: float) -> SimpleNamespace:
    return SimpleNamespace(cutoff=float(cutoff))
