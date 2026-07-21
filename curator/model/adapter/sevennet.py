from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn

from .ase import ASECalculatorAdapter
from .utils import ExternalModelSpec, parse_bool, register_adapter_loader


_SEVENNET_ALIASES = {
    "sevennet": "7net-omni",
    "sevennet-omni": "7net-omni",
    "sevennet-omni-i8": "7net-omni-i8",
    "sevennet-omni-i12": "7net-omni-i12",
    "sevennet-mf-ompa": "7net-mf-ompa",
    "sevennet-omat": "7net-omat",
    "sevennet-l3i5": "7net-l3i5",
    "sevennet-0": "7net-0",
    "sevennet-0-11jul2024": "7net-0_11Jul2024",
}


def _device_name(device: Optional[torch.device]) -> str:
    return str(device) if device is not None else "cpu"


def _resolve_model_name(resource: str) -> str:
    path = Path(resource).expanduser()
    if path.is_file():
        return str(path)
    return _SEVENNET_ALIASES.get(resource.strip().lower(), resource)


def _default_modal(model_name: str) -> Optional[str]:
    lowered = model_name.lower()
    if lowered in {
        "7net-mf-ompa",
        "sevennet-mf-ompa",
        "7net-omni",
        "sevennet-omni",
        "7net-omni-i8",
        "sevennet-omni-i8",
        "7net-omni-i12",
        "sevennet-omni-i12",
    }:
        return "mpa"
    return None


def _load_sevennet(spec: ExternalModelSpec, device: Optional[torch.device]) -> nn.Module:
    try:
        from sevenn.calculator import SevenNetCalculator
    except Exception:
        try:
            from sevenn.sevennet_calculator import SevenNetCalculator
        except Exception as exc:
            raise ModuleNotFoundError("SevenNet support requires the sevenn package.") from exc

    model_name = _resolve_model_name(spec.resource)
    kwargs = {"device": _device_name(device)}
    modal = spec.params.get("modal") or _default_modal(model_name)
    if modal:
        kwargs["modal"] = modal
    for key in ("enable_cueq", "enable_flash", "enable_oeq"):
        if key in spec.params:
            kwargs[key] = parse_bool(spec.params.get(key), False)

    try:
        calc = SevenNetCalculator(model=model_name, **kwargs)
    except TypeError:
        calc = SevenNetCalculator(model_name, **kwargs)
    return ASECalculatorAdapter(calc, cutoff=float(spec.params.get("cutoff", "0.0"))).eval()


register_adapter_loader("sevennet", _load_sevennet)

__all__ = ["_load_sevennet"]
