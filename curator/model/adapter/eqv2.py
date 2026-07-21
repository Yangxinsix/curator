from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn

from .ase import ASECalculatorAdapter
from .utils import ExternalModelSpec, parse_bool, register_adapter_loader


_EQV2_ALIASES = {
    "eqv2-s-oam": ("facebook/OMAT24", "eqV2_31M_omat_mp_salex.pt"),
    "eqv2-m-oam": ("facebook/OMAT24", "eqV2_86M_omat_mp_salex.pt"),
    "eqv2-l-oam": ("facebook/OMAT24", "eqV2_153M_omat_mp_salex.pt"),
    "eqv2-s-omat": ("facebook/OMAT24", "eqV2_31M_omat.pt"),
    "eqv2-m-omat": ("facebook/OMAT24", "eqV2_86M_omat.pt"),
    "eqv2-l-omat": ("facebook/OMAT24", "eqV2_153M_omat.pt"),
    "eqv2-s": ("facebook/OMAT24", "eqV2_31M_mp.pt"),
    "eqv2-s-dens": ("facebook/OMAT24", "eqV2_dens_31M_mp.pt"),
    "eqv2-m-dens": ("facebook/OMAT24", "eqV2_dens_86M_mp.pt"),
    "eqv2-l-dens": ("facebook/OMAT24", "eqV2_dens_153M_mp.pt"),
    "eqv2-31m-omat-mp-salex": ("facebook/OMAT24", "eqV2_31M_omat_mp_salex.pt"),
    "eqv2-86m-omat-mp-salex": ("facebook/OMAT24", "eqV2_86M_omat_mp_salex.pt"),
    "eqv2-153m-omat-mp-salex": ("facebook/OMAT24", "eqV2_153M_omat_mp_salex.pt"),
    "eqv2-31m-omat": ("facebook/OMAT24", "eqV2_31M_omat.pt"),
    "eqv2-86m-omat": ("facebook/OMAT24", "eqV2_86M_omat.pt"),
    "eqv2-153m-omat": ("facebook/OMAT24", "eqV2_153M_omat.pt"),
}


def _device_name(device: Optional[torch.device]) -> str:
    return str(device) if device is not None else "cpu"


def _resolve_checkpoint(spec: ExternalModelSpec) -> str:
    resource_path = Path(spec.resource).expanduser()
    if resource_path.is_file():
        return str(resource_path)

    filename = spec.params.get("filename")
    repo_id = spec.resource
    alias = _EQV2_ALIASES.get(spec.resource.strip().lower())
    if alias is not None:
        repo_id, alias_filename = alias
        filename = filename or alias_filename

    if filename:
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            raise ModuleNotFoundError("eqV2 Hugging Face checkpoints require huggingface_hub.") from exc
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=spec.params.get("revision"),
        )

    return spec.resource


def _load_with_ocp_calculator(checkpoint_path: str, device: Optional[torch.device]):
    try:
        from fairchem.core import OCPCalculator
    except Exception as exc:
        raise ModuleNotFoundError("eqV2 support requires fairchem.core.OCPCalculator.") from exc

    cpu = _device_name(device).startswith("cpu")
    kwargs = {"checkpoint_path": checkpoint_path, "cpu": cpu}
    try:
        return OCPCalculator(**kwargs)
    except TypeError:
        kwargs.pop("cpu", None)
        kwargs["device"] = _device_name(device)
        try:
            return OCPCalculator(**kwargs)
        except TypeError:
            kwargs.pop("device", None)
            return OCPCalculator(**kwargs)


def _load_with_fairchem_calculator(
    checkpoint_or_name: str,
    spec: ExternalModelSpec,
    device: Optional[torch.device],
):
    try:
        from fairchem.core import FAIRChemCalculator
    except Exception:
        try:
            from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
        except Exception as exc:
            raise ModuleNotFoundError("eqV2 named model support requires FAIRChemCalculator.") from exc
    try:
        from fairchem.core import pretrained_mlip
    except Exception:
        try:
            from fairchem.core.calculate import pretrained_mlip
        except Exception as exc:
            raise ModuleNotFoundError("eqV2 named model support requires fairchem pretrained_mlip.") from exc

    inference_settings = spec.params.get("inference_settings", "default")
    task_name = spec.params.get("task", "omat")
    path = Path(checkpoint_or_name).expanduser()
    if path.is_file():
        try:
            from fairchem.core.units.mlip_unit import load_predict_unit
        except Exception as exc:
            raise ModuleNotFoundError("Loading local eqV2 predict units requires fairchem.core.units.mlip_unit.") from exc
        predictor = load_predict_unit(
            str(path),
            inference_settings=inference_settings,
            device=_device_name(device),
        )
    else:
        predictor = pretrained_mlip.get_predict_unit(
            checkpoint_or_name,
            inference_settings=inference_settings,
            device=_device_name(device),
        )
    return FAIRChemCalculator(predictor, task_name=task_name)


def _load_eqv2(spec: ExternalModelSpec, device: Optional[torch.device]) -> nn.Module:
    checkpoint_or_name = _resolve_checkpoint(spec)
    prefer_ocp = parse_bool(spec.params.get("prefer_ocp"), True)
    if prefer_ocp:
        try:
            calc = _load_with_ocp_calculator(checkpoint_or_name, device)
        except Exception as ocp_exc:
            try:
                calc = _load_with_fairchem_calculator(checkpoint_or_name, spec, device)
            except Exception as fairchem_exc:
                raise RuntimeError(
                    "Failed to load eqV2 with both OCPCalculator and FAIRChemCalculator. "
                    f"OCPCalculator error: {ocp_exc.__class__.__name__}: {ocp_exc}; "
                    f"FAIRChemCalculator error: {fairchem_exc.__class__.__name__}: {fairchem_exc}"
                ) from fairchem_exc
    else:
        try:
            calc = _load_with_fairchem_calculator(checkpoint_or_name, spec, device)
        except Exception as fairchem_exc:
            try:
                calc = _load_with_ocp_calculator(checkpoint_or_name, device)
            except Exception as ocp_exc:
                raise RuntimeError(
                    "Failed to load eqV2 with both FAIRChemCalculator and OCPCalculator. "
                    f"FAIRChemCalculator error: {fairchem_exc.__class__.__name__}: {fairchem_exc}; "
                    f"OCPCalculator error: {ocp_exc.__class__.__name__}: {ocp_exc}"
                ) from ocp_exc

    return ASECalculatorAdapter(calc, cutoff=float(spec.params.get("cutoff", "0.0"))).eval()


register_adapter_loader("eqv2", _load_eqv2)

__all__ = ["_load_eqv2"]
