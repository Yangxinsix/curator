from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from curator.model.conversion import load_official_nequip_as_curator

from .utils import (
    ExternalModelSpec,
    ensure_local_nequip_source_on_path,
    parse_bool,
    register_adapter_loader,
)
from .nequip_net import resolve_nequip_net_artifact


def _load_nequip(spec: ExternalModelSpec, device: Optional[torch.device]) -> nn.Module:
    compile_mode = spec.params.get("compile_mode", "eager")
    return load_official_nequip_as_curator(
        spec.resource,
        device=device or torch.device("cpu"),
        compile_mode=compile_mode,
    )


def _load_nequip_hf(spec: ExternalModelSpec, device: Optional[torch.device]) -> nn.Module:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise ModuleNotFoundError(
            "NequIP Hugging Face model support requires huggingface_hub."
        ) from exc

    filename = spec.params.get("filename")
    if not filename:
        raise ValueError("nequip_hf specs require '?filename=...' for the packaged model artifact.")
    revision = spec.params.get("revision")
    model_path = hf_hub_download(
        repo_id=spec.resource,
        filename=filename,
        revision=revision,
    )
    compile_mode = spec.params.get("compile_mode", "eager")
    return load_official_nequip_as_curator(
        model_path,
        device=device or torch.device("cpu"),
        compile_mode=compile_mode,
    )


def _load_nequip_net(spec: ExternalModelSpec, device: Optional[torch.device]) -> nn.Module:
    ensure_local_nequip_source_on_path()
    try:
        import nequip.model.saved_models.load_utils  # noqa: F401
    except Exception as exc:
        raise ModuleNotFoundError(
            "NequIP nequip.net model support requires the NequIP saved-model loader."
        ) from exc

    version = spec.params.get("version", "0.1")
    resource = resolve_nequip_net_artifact(
        spec.resource,
        version=version,
        cache_dir=spec.params.get("cache_dir"),
        download=parse_bool(spec.params.get("download"), True),
        timeout_sec=int(spec.params.get("timeout_sec", "300")),
    )
    compile_mode = spec.params.get("compile_mode", "eager")
    return load_official_nequip_as_curator(
        resource,
        device=device or torch.device("cpu"),
        compile_mode=compile_mode,
    )


register_adapter_loader("nequip", _load_nequip)
register_adapter_loader("nequip_hf", _load_nequip_hf)
register_adapter_loader("nequip_net", _load_nequip_net)

__all__ = ["_load_nequip", "_load_nequip_hf", "_load_nequip_net"]
