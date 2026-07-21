from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn

from .ase import ASECalculatorAdapter
from .utils import ExternalModelSpec, register_adapter_loader


_MATTERSIM_ALIASES = {
    "mattersim": "MatterSim-v1.0.0-1M.pth",
    "mattersim-v1": "MatterSim-v1.0.0-1M.pth",
    "mattersim-v1-1m": "MatterSim-v1.0.0-1M.pth",
    "mattersim-v1.0.0-1m": "MatterSim-v1.0.0-1M.pth",
    "mattersim-v1-5m": "MatterSim-v1.0.0-5M.pth",
    "mattersim-v1.0.0-5m": "MatterSim-v1.0.0-5M.pth",
    "1m": "MatterSim-v1.0.0-1M.pth",
    "5m": "MatterSim-v1.0.0-5M.pth",
}


def _device_name(device: Optional[torch.device]) -> str:
    return str(device) if device is not None else "cpu"


def _resolve_load_path(resource: str) -> str:
    path = Path(resource).expanduser()
    if path.is_file():
        return str(path)
    return _MATTERSIM_ALIASES.get(resource.strip().lower(), resource)


def _load_mattersim(spec: ExternalModelSpec, device: Optional[torch.device]) -> nn.Module:
    try:
        from mattersim.forcefield import MatterSimCalculator
    except Exception as exc:
        raise ModuleNotFoundError("MatterSim support requires the mattersim package.") from exc

    device_name = _device_name(device)
    load_path = _resolve_load_path(spec.resource)
    kwargs = {"device": device_name}
    if spec.params.get("load_path"):
        load_path = spec.params["load_path"]

    calc = None
    from_checkpoint = getattr(MatterSimCalculator, "from_checkpoint", None)
    if callable(from_checkpoint):
        checkpoint_name = load_path[:-4] if load_path.lower().endswith(".pth") else load_path
        try:
            calc = from_checkpoint(checkpoint_name, device=device_name)
        except Exception:
            calc = None

    if calc is None:
        try:
            calc = MatterSimCalculator(load_path=load_path, **kwargs)
        except TypeError:
            if spec.resource.strip().lower() in {"mattersim", "mattersim-v1", "mattersim-v1-1m", "1m"}:
                calc = MatterSimCalculator(**kwargs)
            else:
                raise

    return ASECalculatorAdapter(calc, cutoff=float(spec.params.get("cutoff", "0.0"))).eval()


register_adapter_loader("mattersim", _load_mattersim)

__all__ = ["_load_mattersim"]
