from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn

from .config import WrapperConfig, get_wrapper_config, set_wrapper_config
from .utils import infer_module_device_dtype, is_elora_parameter, temporary_default_dtype


def _attach_wrapper_config(model: nn.Module, wrapper_cfg: WrapperConfig) -> None:
    setattr(model, "_wrapper_config", wrapper_cfg)


@contextmanager
def temporary_wrapper_config(wrapper_cfg: WrapperConfig) -> Iterator[WrapperConfig]:
    previous = get_wrapper_config()
    set_wrapper_config(**wrapper_cfg.to_dict())
    try:
        yield wrapper_cfg
    finally:
        set_wrapper_config(**previous.to_dict())


def get_model_wrapper_config(model: nn.Module) -> WrapperConfig:
    attached = getattr(model, "_wrapper_config", None)
    if isinstance(attached, WrapperConfig):
        return attached
    rep = getattr(model, "representation", model)
    defaults = WrapperConfig().to_dict()
    backend = str(getattr(rep, "backend", "") or "").strip().lower()
    adapter = str(getattr(rep, "adapter", "") or "").strip().lower()
    if not backend or backend == "default":
        backend = "e3nn"
    if not adapter or adapter == "default":
        adapter = "none"
    for module in model.modules():
        module_name = type(module).__name__.lower()
        module_path = type(module).__module__.lower()
        if backend == "e3nn" and (
            "cueq" in module_name or "cuequivariance" in module_path
        ):
            backend = "cueq"
        if backend == "e3nn" and (
            module_name.startswith("oeq")
            or module_path.startswith("curator.layer.wrappers.oeq")
            or "openequivariance" in module_path
        ):
            backend = "oeq"
        if adapter == "none" and "lora" in module_name:
            adapter = "lora"
    if adapter == "none":
        for name, parameter in model.named_parameters():
            if is_elora_parameter(name, parameter):
                adapter = "lora"
                break
    payload = {
        "backend": backend,
        "adapter": adapter,
        "lora_rank": int(getattr(rep, "lora_rank", defaults["lora_rank"])),
        "lora_alpha": float(getattr(rep, "lora_alpha", defaults["lora_alpha"])),
        "lora_freeze_base": bool(
            getattr(rep, "lora_freeze_base", defaults["lora_freeze_base"])
        ),
    }
    from .config import resolve_wrapper_config

    return resolve_wrapper_config(**payload)


def _has_cueq_math_dtype_mismatch(
    model: nn.Module,
    *,
    target_dtype: torch.dtype | None,
) -> bool:
    if target_dtype is None:
        return False
    for module in model.modules():
        module_path = type(module).__module__
        if not module_path.startswith("cuequivariance_torch."):
            continue
        math_dtype = getattr(module, "math_dtype", None)
        if isinstance(math_dtype, torch.dtype) and math_dtype != target_dtype:
            return True
    return False


def apply_wrappers(
    model: nn.Module,
    wrapper_cfg: WrapperConfig,
    *,
    target_dtype: torch.dtype | None = None,
) -> nn.Module:
    from curator.model.base import NeuralNetworkPotential

    if not isinstance(model, NeuralNetworkPotential):
        raise TypeError(
            f"apply_wrappers expects NeuralNetworkPotential, got {type(model)!r}."
        )

    device, source_dtype = infer_module_device_dtype(model)
    build_dtype = target_dtype if target_dtype is not None else source_dtype
    current_wrapper_cfg = get_model_wrapper_config(model)
    if (
        current_wrapper_cfg == wrapper_cfg
        and (target_dtype is None or source_dtype == target_dtype)
        and not _has_cueq_math_dtype_mismatch(model, target_dtype=target_dtype)
    ):
        _attach_wrapper_config(model, wrapper_cfg)
        return model

    export_fn = getattr(model.representation, "export_init_kwargs", None)
    if not callable(export_fn):
        raise TypeError(
            f"{model.representation.__class__.__name__} does not expose export_init_kwargs() "
            "for wrapper rebuilds."
        )
    rep_kwargs = export_fn()
    if not isinstance(rep_kwargs, dict):
        raise TypeError(
            f"{model.representation.__class__.__name__}.export_init_kwargs() must return a dict "
            f"for wrapper rebuilds, got {type(rep_kwargs)!r}."
        )
    with temporary_wrapper_config(wrapper_cfg), temporary_default_dtype(build_dtype):
        patched_rep = model.representation.__class__(**rep_kwargs)
    patched_model = model.clone_with_representation(patched_rep)

    patched_model.to(device=device)
    if build_dtype is not None:
        patched_model.to(dtype=build_dtype)
    patched_model.train(model.training)
    patched_model._initialized = getattr(model, "_initialized", False)
    _attach_wrapper_config(patched_model, wrapper_cfg)
    return patched_model


def collect_adapter_parameter_groups(
    model: nn.Module,
    *,
    require_grad: bool = True,
    group_name: str = "adapter",
    weight_decay: float = 0.0,
) -> list[Any]:
    wrapper_cfg = get_model_wrapper_config(model)
    if wrapper_cfg.adapter != "lora":
        return []
    seen: set[int] = set()
    params = []
    for name, param in model.named_parameters():
        if not isinstance(param, nn.Parameter):
            continue
        if not is_elora_parameter(name, param):
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))
        params.append(param)
    if require_grad:
        params = [param for param in params if param.requires_grad]
    if not params:
        return []
    from curator.model.base import ParameterGroup

    return [
        ParameterGroup(
            name=group_name,
            params=params,
            defaults={"weight_decay": float(weight_decay)},
        )
    ]


def merge_model_wrappers(model: nn.Module) -> int:
    merged = 0
    for submodule in model.modules():
        merge_fn = getattr(submodule, "merge_elora", None)
        if callable(merge_fn):
            merge_fn()
            merged += 1
    return merged


__all__ = [
    "temporary_wrapper_config",
    "apply_wrappers",
    "get_model_wrapper_config",
    "collect_adapter_parameter_groups",
    "merge_model_wrappers",
]
