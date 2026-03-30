from __future__ import annotations

from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from typing import Optional, Sequence, Union


WrapperStack = Union[str, Sequence[str], None]


@dataclass(frozen=True)
class WrapperConfig:
    use_cueq: bool = False
    use_elora: bool = False
    wrapper_stack: Optional[str] = None
    elora_rank: int = 16
    elora_alpha: float = 16.0
    elora_freeze_base: bool = True

    @property
    def resolved_stack(self) -> str:
        parts = []
        if self.use_cueq:
            parts.append("cueq")
        if self.use_elora:
            parts.append("elora")
        return "+".join(parts) if parts else "e3nn"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["wrapper_stack"] = self.wrapper_stack or self.resolved_stack
        return payload


def _normalize_stack(wrapper_stack: WrapperStack) -> Optional[str]:
    if wrapper_stack is None:
        return None
    if isinstance(wrapper_stack, str):
        raw = wrapper_stack
    else:
        raw = "+".join(str(item) for item in wrapper_stack if str(item).strip())
    normalized = raw.strip().lower().replace(",", "+")
    if not normalized or normalized == "default":
        return None
    parts = [part.strip() for part in normalized.split("+") if part.strip()]
    if not parts:
        return None
    if any(part not in {"cueq", "elora", "e3nn"} for part in parts):
        raise ValueError(
            f"Unsupported wrapper_stack={wrapper_stack!r}. Expected a combination of "
            f"'cueq' and 'elora'."
        )
    if "e3nn" in parts and len(parts) > 1:
        raise ValueError("wrapper_stack='e3nn' cannot be combined with other wrappers.")
    if parts == ["e3nn"]:
        return "e3nn"
    ordered = []
    for part in ("cueq", "elora"):
        if part in parts:
            ordered.append(part)
    return "+".join(ordered) if ordered else "e3nn"


def resolve_wrapper_config(
    *,
    use_cueq: bool = False,
    use_elora: bool = False,
    wrapper_stack: WrapperStack = None,
    elora_rank: int = 16,
    elora_alpha: float = 16.0,
    elora_freeze_base: bool = True,
) -> WrapperConfig:
    normalized_stack = _normalize_stack(wrapper_stack)
    if normalized_stack is not None:
        use_cueq = "cueq" in normalized_stack
        use_elora = "elora" in normalized_stack
    if elora_rank <= 0:
        use_elora = False
    return WrapperConfig(
        use_cueq=bool(use_cueq),
        use_elora=bool(use_elora),
        wrapper_stack=normalized_stack,
        elora_rank=int(elora_rank),
        elora_alpha=float(elora_alpha),
        elora_freeze_base=bool(elora_freeze_base),
    )


_DEFAULT_CONFIG = WrapperConfig()
_WRAPPER_CONFIG: ContextVar[WrapperConfig] = ContextVar(
    "curator_wrapper_config",
    default=_DEFAULT_CONFIG,
)


def get_wrapper_config() -> WrapperConfig:
    return _WRAPPER_CONFIG.get()


def get_wrapper_config_dict() -> dict:
    return get_wrapper_config().to_dict()


def set_wrapper_config(
    *,
    use_cueq: bool = False,
    use_elora: bool = False,
    wrapper_stack: WrapperStack = None,
    elora_rank: int = 16,
    elora_alpha: float = 16.0,
    elora_freeze_base: bool = True,
) -> WrapperConfig:
    config = resolve_wrapper_config(
        use_cueq=use_cueq,
        use_elora=use_elora,
        wrapper_stack=wrapper_stack,
        elora_rank=elora_rank,
        elora_alpha=elora_alpha,
        elora_freeze_base=elora_freeze_base,
    )
    _WRAPPER_CONFIG.set(config)
    return config


def update_wrapper_config(**kwargs) -> WrapperConfig:
    current = get_wrapper_config()
    payload = current.to_dict()
    payload.update(kwargs)
    return set_wrapper_config(**payload)


def clone_wrapper_config(
    config: Optional[WrapperConfig] = None,
    **updates,
) -> WrapperConfig:
    base = config or get_wrapper_config()
    return replace(base, **updates)
