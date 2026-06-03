from __future__ import annotations

from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class WrapperConfig:
    backend: str = "e3nn"
    adapter: str = "none"
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_freeze_base: bool = False
    lora_target_groups: tuple[str, ...] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


_WRAPPER_FIELDS = (
    "backend",
    "adapter",
    "lora_rank",
    "lora_alpha",
    "lora_freeze_base",
    "lora_target_groups",
)


def _normalize_backend(backend: Optional[str]) -> str:
    normalized = str(backend or "e3nn").strip().lower()
    if not normalized or normalized == "default":
        normalized = "e3nn"
    if normalized in {"oeq", "open_equivariance"}:
        normalized = "oeq"
    if normalized in {"cuequivariance", "cuequivariance_torch"}:
        normalized = "cueq"
    if normalized == "openequivariance":
        normalized = "oeq"
    if normalized not in {"e3nn", "cueq", "oeq"}:
        raise ValueError(
            f"Unsupported backend={backend!r}. Expected one of "
            f"['e3nn', 'cueq', 'cuequivariance', 'oeq', 'openequivariance']."
        )
    return normalized


def _normalize_adapter(adapter: Optional[str]) -> str:
    normalized = str(adapter or "none").strip().lower()
    if not normalized or normalized == "default":
        normalized = "none"
    if normalized not in {"none", "lora"}:
        raise ValueError(
            f"Unsupported adapter={adapter!r}. Expected one of ['none', 'lora']."
        )
    return normalized


def _normalize_lora_target_groups(
    groups: Optional[Sequence[Any] | str],
) -> tuple[str, ...] | None:
    if groups is None:
        return None
    if isinstance(groups, str):
        groups = [groups]
    normalized = []
    for group in groups:
        token = str(group).strip().lower()
        if not token or token == "none":
            continue
        if token not in normalized:
            normalized.append(token)
    return tuple(normalized) if normalized else None


def _to_payload(
    config_like: DictConfig | Mapping[str, Any] | WrapperConfig | None,
) -> Optional[dict]:
    if config_like is None:
        return None
    if isinstance(config_like, WrapperConfig):
        return config_like.to_dict()
    if isinstance(config_like, DictConfig):
        payload = OmegaConf.to_container(config_like, resolve=False)
    elif isinstance(config_like, Mapping):
        payload = dict(config_like)
    else:
        raise TypeError(
            f"Expected DictConfig, mapping, or wrapper config dataclass, got {type(config_like)!r}."
        )
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected wrapper config payload to resolve to a dict, got {type(payload)!r}."
        )
    return payload


def resolve_wrapper_config(
    *,
    backend: str = "e3nn",
    adapter: str = "none",
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
    lora_freeze_base: bool = False,
    lora_target_groups: Optional[Sequence[Any] | str] = None,
) -> WrapperConfig:
    backend = _normalize_backend(backend)
    adapter = _normalize_adapter(adapter)
    if lora_rank <= 0:
        adapter = "none"
    return WrapperConfig(
        backend=backend,
        adapter=adapter,
        lora_rank=int(lora_rank),
        lora_alpha=float(lora_alpha),
        lora_freeze_base=bool(lora_freeze_base),
        lora_target_groups=_normalize_lora_target_groups(lora_target_groups),
    )


def get_config_wrapper_config(
    config_like: DictConfig | Mapping[str, Any] | WrapperConfig | None
) -> Optional[WrapperConfig]:
    payload = _to_payload(config_like)
    if payload is None:
        return None
    present_fields = [field for field in _WRAPPER_FIELDS if field in payload and payload[field] is not None]
    if not present_fields:
        return None

    def _value(name: str, default: Any) -> Any:
        value = payload.get(name, default)
        return default if value is None else value

    return resolve_wrapper_config(
        backend=_value("backend", "e3nn"),
        adapter=_value("adapter", "none"),
        lora_rank=_value("lora_rank", 16),
        lora_alpha=_value("lora_alpha", 16.0),
        lora_freeze_base=_value("lora_freeze_base", False),
        lora_target_groups=_value("lora_target_groups", None),
    )


_DEFAULT_CONFIG = WrapperConfig()
_WRAPPER_CONFIG: ContextVar[WrapperConfig] = ContextVar(
    "curator_wrapper_config",
    default=_DEFAULT_CONFIG,
)
def get_wrapper_config() -> WrapperConfig:
    return _WRAPPER_CONFIG.get()


def set_wrapper_config(
    *,
    backend: str = "e3nn",
    adapter: str = "none",
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
    lora_freeze_base: bool = False,
    lora_target_groups: Optional[Sequence[Any] | str] = None,
) -> WrapperConfig:
    config = resolve_wrapper_config(
        backend=backend,
        adapter=adapter,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_freeze_base=lora_freeze_base,
        lora_target_groups=lora_target_groups,
    )
    _WRAPPER_CONFIG.set(config)
    return config
