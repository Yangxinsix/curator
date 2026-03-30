from __future__ import annotations

import inspect
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

import torch
from torch import nn

from .config import WrapperConfig, get_wrapper_config, resolve_wrapper_config, set_wrapper_config
from .utils import collect_named_parameters, is_elora_parameter

_WRAPPER_CFG_KEYS = frozenset(
    {
        "use_cueq",
        "use_elora",
        "wrapper_stack",
        "elora_rank",
        "elora_alpha",
        "elora_freeze_base",
    }
)


def _attach_wrapper_config(model: nn.Module, wrapper_cfg: WrapperConfig) -> None:
    setattr(model, "_wrapper_config", wrapper_cfg)
    rep = getattr(model, "representation", None)
    if rep is not None:
        setattr(rep, "_wrapper_config", wrapper_cfg)


def _normalize_wrapper_config(
    wrapper_cfg: WrapperConfig | Mapping[str, Any] | None = None,
    **updates: Any,
) -> WrapperConfig:
    if wrapper_cfg is None:
        payload: dict[str, Any] = get_wrapper_config().to_dict()
    elif isinstance(wrapper_cfg, WrapperConfig):
        payload = wrapper_cfg.to_dict()
    elif isinstance(wrapper_cfg, Mapping):
        payload = {key: value for key, value in wrapper_cfg.items() if key in _WRAPPER_CFG_KEYS}
    else:
        raise TypeError(
            f"Unsupported wrapper_cfg type: {type(wrapper_cfg)!r}. "
            "Expected WrapperConfig, mapping, or None."
        )

    unknown = set(updates) - _WRAPPER_CFG_KEYS
    if unknown:
        raise ValueError(f"Unknown wrapper config fields: {sorted(unknown)}")
    payload.update(updates)
    payload = {key: value for key, value in payload.items() if key in _WRAPPER_CFG_KEYS}
    return resolve_wrapper_config(**payload)


def _callable_kwargs(target: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _infer_model_dtype(model: nn.Module) -> torch.dtype:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    for buffer in model.buffers():
        if buffer.is_floating_point():
            return buffer.dtype
    return torch.get_default_dtype()


def _infer_model_device(model: nn.Module) -> torch.device:
    for parameter in model.parameters():
        return parameter.device
    for buffer in model.buffers():
        return buffer.device
    return torch.device("cpu")


def _canonical_wrapper_state_key(key: str) -> str:
    return key.replace(".base.", ".")


def _coerce_matching_state_tensor(
    source_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
) -> torch.Tensor | None:
    if source_tensor.shape == target_tensor.shape:
        return source_tensor
    if (
        source_tensor.dim() + 1 == target_tensor.dim()
        and target_tensor.shape[0] == 1
        and source_tensor.shape == target_tensor.shape[1:]
    ):
        return source_tensor.unsqueeze(0)
    if (
        source_tensor.dim() == target_tensor.dim() + 1
        and source_tensor.shape[0] == 1
        and source_tensor.shape[1:] == target_tensor.shape
    ):
        return source_tensor.squeeze(0)
    return None


def _load_matching_state(target: nn.Module, source_state: Mapping[str, torch.Tensor]) -> int:
    target_state = target.state_dict()
    source_aliases: dict[str, torch.Tensor] = {}
    for source_key, source_tensor in source_state.items():
        source_aliases.setdefault(source_key, source_tensor)
        source_aliases.setdefault(_canonical_wrapper_state_key(source_key), source_tensor)
    matched: dict[str, torch.Tensor] = {}
    for key, target_tensor in target_state.items():
        canonical_key = _canonical_wrapper_state_key(key)
        for source_tensor in (
            source_state.get(key),
            source_state.get(canonical_key),
            source_aliases.get(canonical_key),
        ):
            if source_tensor is None:
                continue
            compatible = _coerce_matching_state_tensor(source_tensor, target_tensor)
            if compatible is None:
                continue
            matched[key] = compatible
            break
    if matched:
        target.load_state_dict(matched, strict=False)
    return len(matched)


def _representation_init_kwargs(model: nn.Module) -> dict[str, Any]:
    rep = getattr(model, "representation", None)
    if rep is None:
        raise TypeError("Expected a model with a `representation` attribute.")
    export_fn = getattr(rep, "export_init_kwargs", None)
    if callable(export_fn):
        try:
            payload = export_fn()
        except NotImplementedError:
            payload = None
        if payload is not None:
            if not isinstance(payload, Mapping):
                raise TypeError(
                    f"{rep.__class__.__name__}.export_init_kwargs() must return a mapping, "
                    f"got {type(payload)!r}."
                )
            return dict(payload)

    from curator.utils import get_representation_config

    payload = get_representation_config(model)
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"get_representation_config({rep.__class__.__name__}) returned {type(payload)!r}, "
            "expected a mapping."
        )
    return dict(payload)


def _clone_model_with_representation(model: nn.Module, representation: nn.Module) -> nn.Module:
    clone_fn = getattr(model, "clone_with_representation", None)
    if callable(clone_fn):
        return clone_fn(representation)

    model_kwargs = _callable_kwargs(
        model.__class__,
        {
            "input_modules": list(getattr(model, "input_modules", [])),
            "output_modules": list(getattr(model, "output_modules", [])),
            "representation": representation,
            "model_outputs": list(getattr(model, "model_outputs", [])),
            "heads": getattr(model, "heads", None),
        },
    )
    return model.__class__(**model_kwargs)


@contextmanager
def temporary_wrapper_config(
    wrapper_cfg: WrapperConfig | Mapping[str, Any] | None = None,
    **updates: Any,
) -> Iterator[WrapperConfig]:
    previous = get_wrapper_config()
    next_cfg = _normalize_wrapper_config(wrapper_cfg, **updates)
    set_wrapper_config(**next_cfg.to_dict())
    try:
        yield next_cfg
    finally:
        set_wrapper_config(**previous.to_dict())


def get_model_wrapper_config(model: nn.Module) -> WrapperConfig:
    attached = getattr(model, "_wrapper_config", None)
    if isinstance(attached, WrapperConfig):
        return attached
    rep = getattr(model, "representation", model)
    attached = getattr(rep, "_wrapper_config", None)
    if isinstance(attached, WrapperConfig):
        return attached
    defaults = get_wrapper_config().to_dict()
    use_cueq = bool(getattr(rep, "use_cueq", False))
    use_elora = bool(getattr(rep, "use_elora", False))
    if not (use_cueq or use_elora):
        for module in model.modules():
            module_name = type(module).__name__.lower()
            module_path = type(module).__module__.lower()
            if "lora" in module_name:
                use_elora = True
            if "cueq" in module_name or "cuequivariance_torch" in module_path:
                use_cueq = True
        if not use_elora:
            for name, parameter in model.named_parameters():
                if is_elora_parameter(name, parameter):
                    use_elora = True
                    break
    payload = {
        "use_cueq": use_cueq,
        "use_elora": use_elora,
        "wrapper_stack": getattr(rep, "wrapper_stack", defaults["wrapper_stack"]),
        "elora_rank": int(getattr(rep, "elora_rank", defaults["elora_rank"])),
        "elora_alpha": float(getattr(rep, "elora_alpha", defaults["elora_alpha"])),
        "elora_freeze_base": bool(
            getattr(rep, "elora_freeze_base", defaults["elora_freeze_base"])
        ),
    }
    return resolve_wrapper_config(**payload)


def export_wrapper_config(model: nn.Module) -> dict[str, Any]:
    return get_model_wrapper_config(model).to_dict()


def apply_wrappers(
    model: nn.Module,
    wrapper_cfg: WrapperConfig | Mapping[str, Any] | None = None,
    **updates: Any,
) -> nn.Module:
    if not hasattr(model, "representation"):
        raise TypeError("apply_wrappers expects a model with a `representation` attribute.")

    source_cfg = get_model_wrapper_config(model)
    target_cfg = _normalize_wrapper_config(wrapper_cfg, **updates)
    if source_cfg.to_dict() == target_cfg.to_dict():
        _attach_wrapper_config(model, target_cfg)
        return model

    from curator.utils import load_cueq_weights, load_e3nn_weights

    rep_cfg = _representation_init_kwargs(model)

    dtype = _infer_model_dtype(model)
    old_dtype = torch.get_default_dtype()
    if dtype != old_dtype:
        torch.set_default_dtype(dtype)
    try:
        with temporary_wrapper_config(target_cfg):
            rep_kwargs = _callable_kwargs(model.representation.__class__, rep_cfg)
            patched_rep = model.representation.__class__(**rep_kwargs)
            patched_model = _clone_model_with_representation(model, patched_rep)
    finally:
        if dtype != old_dtype:
            torch.set_default_dtype(old_dtype)

    if bool(source_cfg.use_cueq) != bool(target_cfg.use_cueq):
        if target_cfg.use_cueq:
            load_e3nn_weights(model, patched_model)
        else:
            load_cueq_weights(model, patched_model)

    _load_matching_state(patched_model, model.state_dict())
    patched_model.to(device=_infer_model_device(model))
    patched_model.train(model.training)
    _attach_wrapper_config(patched_model, target_cfg)
    return patched_model


def collect_addon_parameter_groups(
    model: nn.Module,
    *,
    require_grad: bool = True,
    group_name: str = "elora",
    weight_decay: float = 0.0,
) -> list[Any]:
    wrapper_cfg = get_model_wrapper_config(model)
    if not wrapper_cfg.use_elora:
        return []
    params = collect_named_parameters(
        model,
        include=is_elora_parameter,
    )
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


def merge_model_elora(model: nn.Module) -> int:
    return merge_model_wrappers(model)


__all__ = [
    "temporary_wrapper_config",
    "apply_wrappers",
    "get_model_wrapper_config",
    "export_wrapper_config",
    "collect_addon_parameter_groups",
    "merge_model_wrappers",
    "merge_model_elora",
]
