from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn

from .ase import ASECalculatorAdapter
from .utils import ExternalModelSpec, parse_bool, register_adapter_loader


_ORB_ALIASES = {
    "orb": "orb_v3_conservative_inf_omat",
    "orb-v2": "orb_v2",
    "orb-d3-v2": "orb_d3_v2",
    "orb-mptraj-only-v2": "orb_mptraj_only_v2",
    "orb-v3-conservative-inf-omat": "orb_v3_conservative_inf_omat",
    "orb-v3-direct-inf-omat": "orb_v3_direct_inf_omat",
    "orb-v3-conservative-20-omat": "orb_v3_conservative_20_omat",
    "orb-v3-conservative-120-omat": "orb_v3_conservative_20_omat",
    "orb-v3-direct-20-omat": "orb_v3_direct_20_omat",
    "orb-v3-direct-120-omat": "orb_v3_direct_20_omat",
    "orb-v3-conservative-inf-mpa": "orb_v3_conservative_inf_mpa",
    "orb-v3-direct-inf-mpa": "orb_v3_direct_inf_mpa",
    "orb-v3-conservative-20-mpa": "orb_v3_conservative_20_mpa",
    "orb-v3-conservative-120-mpa": "orb_v3_conservative_20_mpa",
    "orb-v3-direct-20-mpa": "orb_v3_direct_20_mpa",
    "orb-v3-direct-120-mpa": "orb_v3_direct_20_mpa",
    "orbmol-v1-conservative": "orbmol_v1_conservative",
    "orbmol-v1-direct": "orbmol_v1_direct",
    "orbmol-v2": "orbmol_v2",
}


def _device_name(device: Optional[torch.device]) -> str:
    return str(device) if device is not None else "cpu"


def _normalize_model_name(resource: str) -> str:
    lowered = resource.strip().lower()
    return _ORB_ALIASES.get(lowered, lowered.replace("-", "_"))


def _get_pretrained_loader(pretrained, resource: str):
    model_name = _normalize_model_name(resource)
    loader = getattr(pretrained, model_name, None)
    if callable(loader):
        return loader

    registry = getattr(pretrained, "ORB_PRETRAINED_MODELS", {}) or {}
    for key in (resource, resource.replace("_", "-"), model_name, model_name.replace("_", "-")):
        candidate = registry.get(key)
        if callable(candidate):
            return candidate

    known = sorted(str(key) for key in registry.keys())
    known_text = ", ".join(known[:25])
    if len(known) > 25:
        known_text += ", ..."
    raise ValueError(f"Unknown ORB pretrained model {resource!r}. Known models: {known_text}")


def _load_orb(spec: ExternalModelSpec, device: Optional[torch.device]) -> nn.Module:
    try:
        from orb_models.forcefield import pretrained
    except Exception as exc:
        raise ModuleNotFoundError("ORB support requires the orb-models package.") from exc

    try:
        from orb_models.forcefield.inference.calculator import ORBCalculator
    except Exception:
        try:
            from orb_models.forcefield.calculator import ORBCalculator
        except Exception as exc:
            raise ModuleNotFoundError(
                "ORB support requires orb_models.forcefield.inference.calculator.ORBCalculator."
            ) from exc

    device_name = _device_name(device)
    precision = spec.params.get("precision", "float32-high")
    weights_path = spec.params.get("weights_path")
    resource_path = Path(spec.resource).expanduser()
    if weights_path is None and resource_path.is_file():
        weights_path = str(resource_path)
    base_model = spec.params.get("base_model", spec.resource if weights_path is None else "orb-v3-conservative-inf-omat")
    loader = _get_pretrained_loader(pretrained, base_model)

    kwargs = {"device": device_name, "precision": precision}
    if weights_path is not None:
        kwargs["weights_path"] = weights_path
    compile_model = spec.params.get("compile")
    if compile_model is not None:
        kwargs["compile"] = parse_bool(compile_model, True)

    try:
        loaded = loader(**kwargs)
    except TypeError:
        kwargs.pop("compile", None)
        loaded = loader(**kwargs)

    if isinstance(loaded, tuple):
        if len(loaded) < 2:
            raise ValueError("ORB pretrained loader returned a tuple without an atoms adapter.")
        orbff, atoms_adapter = loaded[0], loaded[1]
    else:
        orbff, atoms_adapter = loaded, None

    if parse_bool(spec.params.get("d3"), False):
        try:
            from orb_models.forcefield.inference.d3_model import AlchemiDFTD3, D3SumModel
        except Exception as exc:
            raise ModuleNotFoundError("ORB D3 correction requires the ORB D3 inference module.") from exc
        functional = spec.params.get("functional", "PBE")
        damping = spec.params.get("damping", "BJ")
        d3_compile = parse_bool(spec.params.get("d3_compile"), True)
        orbff = D3SumModel(orbff, AlchemiDFTD3(functional=functional, damping=damping, compile=d3_compile))

    calc_kwargs = {"device": device_name}
    if atoms_adapter is not None:
        calc_kwargs["atoms_adapter"] = atoms_adapter
    calc = ORBCalculator(orbff, **calc_kwargs)
    return ASECalculatorAdapter(calc, cutoff=float(spec.params.get("cutoff", "0.0"))).eval()


register_adapter_loader("orb", _load_orb)

__all__ = ["_load_orb"]
