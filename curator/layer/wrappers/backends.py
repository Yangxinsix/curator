from __future__ import annotations

from .config import get_wrapper_config
from .cueq import (
    IS_CUET_AVAILABLE,
    make_e3nn_symmetric_contraction,
    make_fully_connected_tensor_product as make_cueq_fctp,
    make_linear as make_cueq_linear,
    make_symmetric_contraction as make_cueq_symmetric_contraction,
    make_tensor_product as make_cueq_tensor_product,
)
from .cueq_elora import (
    CueqLoRAChannelWiseTensorProduct,
    CueqLoRAFullyConnectedTensorProduct,
    CueqLoRALinear,
    CueqLoRASymmetricContraction,
)
from .elora import (
    ELoRAFullyConnectedTensorProduct,
    ELoRALinear,
    ELoRATensorProduct,
)
from .mlp import (
    build_convnet_radial_mlp as build_scalar_convnet_radial_mlp,
    build_fully_connected_net as build_scalar_fully_connected_net,
)
from .scalar import build_scalar_linear
from .utils import ensure_torch_serialization_compat

ensure_torch_serialization_compat()

from e3nn import o3


class E3NNBackend:
    name = "e3nn"

    def make_linear(self, *args, **kwargs):
        return o3.Linear(*args, **kwargs)

    def make_tensor_product(self, *args, **kwargs):
        return o3.TensorProduct(*args, **kwargs)

    def make_fully_connected_tensor_product(self, *args, **kwargs):
        return o3.FullyConnectedTensorProduct(*args, **kwargs)

    def make_symmetric_contraction(self, *args, **kwargs):
        return make_e3nn_symmetric_contraction(*args, **kwargs)

    def make_fully_connected_net(self, *args, **kwargs):
        return build_scalar_fully_connected_net(*args, use_elora=False, **kwargs)

    def make_convnet_radial_mlp(self, *args, **kwargs):
        return build_scalar_convnet_radial_mlp(*args, use_elora=False, **kwargs)

    def make_scalar_linear(self, *args, **kwargs):
        return build_scalar_linear(*args, use_elora=False, **kwargs)


class CueqBackend(E3NNBackend):
    name = "cueq"

    def make_linear(self, *args, **kwargs):
        return make_cueq_linear(*args, **kwargs)

    def make_tensor_product(self, *args, **kwargs):
        return make_cueq_tensor_product(*args, **kwargs)

    def make_fully_connected_tensor_product(self, *args, **kwargs):
        return make_cueq_fctp(*args, **kwargs)

    def make_symmetric_contraction(self, *args, **kwargs):
        return make_cueq_symmetric_contraction(*args, **kwargs)


class _ConfiguredELoRABackendMixin:
    def __init__(self, *, rank: int, alpha: float, freeze_base: bool = True) -> None:
        self.elora_rank = int(rank)
        self.elora_alpha = float(alpha)
        self.elora_freeze_base = bool(freeze_base)

    def make_fully_connected_net(self, *args, **kwargs):
        return build_scalar_fully_connected_net(
            *args,
            use_elora=True,
            rank=self.elora_rank,
            alpha=self.elora_alpha,
            elora_freeze_base=self.elora_freeze_base,
            **kwargs,
        )

    def make_convnet_radial_mlp(self, *args, **kwargs):
        return build_scalar_convnet_radial_mlp(
            *args,
            use_elora=True,
            rank=self.elora_rank,
            alpha=self.elora_alpha,
            elora_freeze_base=self.elora_freeze_base,
            **kwargs,
        )

    def make_scalar_linear(self, *args, **kwargs):
        return build_scalar_linear(
            *args,
            use_elora=True,
            rank=self.elora_rank,
            alpha=self.elora_alpha,
            freeze_base=self.elora_freeze_base,
            **kwargs,
        )


class ELoRABackend(_ConfiguredELoRABackendMixin, E3NNBackend):
    name = "elora"

    def make_linear(self, *args, **kwargs):
        return ELoRALinear(
            *args,
            elora_rank=self.elora_rank,
            elora_alpha=self.elora_alpha,
            elora_freeze_base=self.elora_freeze_base,
            **kwargs,
        )

    def make_tensor_product(self, *args, **kwargs):
        if kwargs.get("internal_weights", False):
            return ELoRATensorProduct(
                *args,
                elora_rank=self.elora_rank,
                elora_alpha=self.elora_alpha,
                elora_freeze_base=self.elora_freeze_base,
                **kwargs,
            )
        return o3.TensorProduct(*args, **kwargs)

    def make_fully_connected_tensor_product(self, *args, **kwargs):
        return ELoRAFullyConnectedTensorProduct(
            *args,
            elora_rank=self.elora_rank,
            elora_alpha=self.elora_alpha,
            elora_freeze_base=self.elora_freeze_base,
            **kwargs,
        )

class CueqELoRABackend(_ConfiguredELoRABackendMixin, CueqBackend):
    name = "cueq+elora"

    def make_linear(self, *args, **kwargs):
        return CueqLoRALinear(
            *args,
            rank=self.elora_rank,
            alpha=self.elora_alpha,
            elora_freeze_base=self.elora_freeze_base,
            **kwargs,
        )

    def make_tensor_product(self, *args, **kwargs):
        if kwargs.get("internal_weights", False):
            return CueqLoRAChannelWiseTensorProduct(
                *args,
                rank=self.elora_rank,
                alpha=self.elora_alpha,
                elora_freeze_base=self.elora_freeze_base,
                **kwargs,
            )
        return make_cueq_tensor_product(*args, **kwargs)

    def make_fully_connected_tensor_product(self, *args, **kwargs):
        return CueqLoRAFullyConnectedTensorProduct(
            *args,
            rank=self.elora_rank,
            alpha=self.elora_alpha,
            elora_freeze_base=self.elora_freeze_base,
            **kwargs,
        )

    def make_symmetric_contraction(self, *args, **kwargs):
        return CueqLoRASymmetricContraction(
            *args,
            rank=self.elora_rank,
            alpha=self.elora_alpha,
            elora_freeze_base=self.elora_freeze_base,
            **kwargs,
        )


def _resolved_stack(
    *,
    use_cueq: bool = True,
    use_elora: bool | None = None,
) -> tuple[str, ...]:
    config = get_wrapper_config()
    resolved_stack = config.resolved_stack
    if resolved_stack in ("", "e3nn"):
        resolved = []
    else:
        resolved = [part for part in resolved_stack.split("+") if part]
    if not use_cueq:
        resolved = [item for item in resolved if item != "cueq"]
    if use_elora is False:
        resolved = [item for item in resolved if item != "elora"]
    elif use_elora is True and "elora" not in resolved:
        resolved.append("elora")
    if "cueq" in resolved and not IS_CUET_AVAILABLE:
        resolved = [item for item in resolved if item != "cueq"]
    return tuple(resolved)


def get_backend(
    *,
    use_cueq: bool = True,
    use_elora: bool | None = None,
):
    config = get_wrapper_config()
    stack = _resolved_stack(use_cueq=use_cueq, use_elora=use_elora)
    if stack == ("cueq", "elora"):
        return CueqELoRABackend(
            rank=config.elora_rank,
            alpha=config.elora_alpha,
            freeze_base=config.elora_freeze_base,
        )
    if stack == ("elora",):
        return ELoRABackend(
            rank=config.elora_rank,
            alpha=config.elora_alpha,
            freeze_base=config.elora_freeze_base,
        )
    if stack == ("cueq",):
        return CueqBackend()
    return E3NNBackend()


def get_backend_name(*, use_cueq: bool = True, use_elora: bool | None = None) -> str:
    return get_backend(use_cueq=use_cueq, use_elora=use_elora).name


__all__ = [
    "E3NNBackend",
    "CueqBackend",
    "ELoRABackend",
    "CueqELoRABackend",
    "get_backend",
    "get_backend_name",
]
