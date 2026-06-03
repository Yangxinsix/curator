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
)
from .elora import (
    ELoRAFullyConnectedTensorProduct,
    ELoRALinear,
    ELoRATensorProduct,
)
from .oeq import (
    OeqFullyConnectedTensorProduct,
    OeqLinear,
    OeqSymmetricContraction,
    OeqTensorProduct,
    ensure_oeq_runtime_available,
)
from .oeq_elora import (
    OeqLoRAFullyConnectedTensorProduct,
    OeqLoRALinear,
    OeqLoRATensorProduct,
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


class OeqBackend(E3NNBackend):
    name = "oeq"

    def make_linear(self, *args, **kwargs):
        return OeqLinear(*args, **kwargs)

    def make_tensor_product(self, *args, **kwargs):
        return OeqTensorProduct(*args, **kwargs)

    def make_fully_connected_tensor_product(self, *args, **kwargs):
        return OeqFullyConnectedTensorProduct(*args, **kwargs)

    def make_symmetric_contraction(self, *args, **kwargs):
        return OeqSymmetricContraction(*args, **kwargs)


class _ConfiguredLoRABackendMixin:
    def __init__(self, *, rank: int, alpha: float, freeze_base: bool = True) -> None:
        self.lora_rank = int(rank)
        self.lora_alpha = float(alpha)
        self.lora_freeze_base = bool(freeze_base)

    def make_fully_connected_net(self, *args, **kwargs):
        return build_scalar_fully_connected_net(
            *args,
            use_elora=True,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            elora_freeze_base=self.lora_freeze_base,
            **kwargs,
        )

    def make_convnet_radial_mlp(self, *args, **kwargs):
        return build_scalar_convnet_radial_mlp(
            *args,
            use_elora=True,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            elora_freeze_base=self.lora_freeze_base,
            **kwargs,
        )

    def make_scalar_linear(self, *args, **kwargs):
        return build_scalar_linear(
            *args,
            use_elora=True,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            freeze_base=self.lora_freeze_base,
            **kwargs,
        )


class LoRABackend(_ConfiguredLoRABackendMixin, E3NNBackend):
    name = "lora"

    def make_linear(self, *args, **kwargs):
        return ELoRALinear(
            *args,
            elora_rank=self.lora_rank,
            elora_alpha=self.lora_alpha,
            elora_freeze_base=self.lora_freeze_base,
            **kwargs,
        )

    def make_tensor_product(self, *args, **kwargs):
        if kwargs.get("internal_weights", False):
            return ELoRATensorProduct(
                *args,
                elora_rank=self.lora_rank,
                elora_alpha=self.lora_alpha,
                elora_freeze_base=self.lora_freeze_base,
                **kwargs,
            )
        return o3.TensorProduct(*args, **kwargs)

    def make_fully_connected_tensor_product(self, *args, **kwargs):
        return ELoRAFullyConnectedTensorProduct(
            *args,
            elora_rank=self.lora_rank,
            elora_alpha=self.lora_alpha,
            elora_freeze_base=self.lora_freeze_base,
            **kwargs,
        )

class CueqLoRABackend(_ConfiguredLoRABackendMixin, CueqBackend):
    name = "cueq+lora"

    def make_linear(self, *args, **kwargs):
        return CueqLoRALinear(
            *args,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            elora_freeze_base=self.lora_freeze_base,
            **kwargs,
        )

    def make_tensor_product(self, *args, **kwargs):
        if kwargs.get("internal_weights", False):
            reference_instructions = kwargs.get("instructions")
            if reference_instructions is None and len(args) >= 4:
                reference_instructions = args[3]
            return CueqLoRAChannelWiseTensorProduct(
                *args,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                elora_freeze_base=self.lora_freeze_base,
                reference_instructions=reference_instructions,
                **kwargs,
            )
        return make_cueq_tensor_product(*args, **kwargs)

    def make_fully_connected_tensor_product(self, *args, **kwargs):
        return CueqLoRAFullyConnectedTensorProduct(
            *args,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            elora_freeze_base=self.lora_freeze_base,
            **kwargs,
        )

    def make_symmetric_contraction(self, *args, **kwargs):
        return make_cueq_symmetric_contraction(*args, **kwargs)


class OeqLoRABackend(_ConfiguredLoRABackendMixin, OeqBackend):
    name = "oeq+lora"

    def make_linear(self, *args, **kwargs):
        return OeqLoRALinear(
            *args,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            elora_freeze_base=self.lora_freeze_base,
            **kwargs,
        )

    def make_tensor_product(self, *args, **kwargs):
        if kwargs.get("internal_weights", False):
            return OeqLoRATensorProduct(
                *args,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                elora_freeze_base=self.lora_freeze_base,
                **kwargs,
            )
        return OeqTensorProduct(*args, **kwargs)

    def make_fully_connected_tensor_product(self, *args, **kwargs):
        return OeqLoRAFullyConnectedTensorProduct(
            *args,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            elora_freeze_base=self.lora_freeze_base,
            **kwargs,
        )

    def make_symmetric_contraction(self, *args, **kwargs):
        return OeqSymmetricContraction(
            *args,
            freeze_base=self.lora_freeze_base,
            **kwargs,
        )


def get_backend():
    config = get_wrapper_config()
    use_lora = config.adapter == "lora"
    if config.backend == "oeq":
        ensure_oeq_runtime_available()
    if config.backend == "cueq" and not IS_CUET_AVAILABLE:
        raise ImportError("backend=cueq requires cuequivariance_torch to be available.")
    if config.backend == "oeq" and use_lora:
        return OeqLoRABackend(
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            freeze_base=config.lora_freeze_base,
        )
    if config.backend == "cueq" and use_lora:
        return CueqLoRABackend(
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            freeze_base=config.lora_freeze_base,
        )
    if config.backend == "e3nn" and use_lora:
        return LoRABackend(
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            freeze_base=config.lora_freeze_base,
        )
    if config.backend == "cueq":
        return CueqBackend()
    if config.backend == "oeq":
        return OeqBackend()
    return E3NNBackend()


def get_backend_name() -> str:
    return get_backend().name


__all__ = [
    "E3NNBackend",
    "CueqBackend",
    "OeqBackend",
    "LoRABackend",
    "CueqLoRABackend",
    "OeqLoRABackend",
    "get_backend",
    "get_backend_name",
]
