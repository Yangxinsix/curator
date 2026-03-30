from __future__ import annotations

import torch
from torch import nn

from .cueq import CuetSymmetricContraction, cuet
from .utils import SingleWeightLoRAModuleMixin, freeze_module_parameters


class _BaseCueqELoRAWrapper(SingleWeightLoRAModuleMixin, nn.Module):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = super().__getattr__("base")
            return getattr(base, name)

    def merge_elora(self) -> None:
        self._merge_single_weight_elora(self.base.weight)

    def _init_base_wrapper(
        self,
        base: nn.Module,
        *,
        rank: int,
        alpha: float,
        freeze_base: bool,
    ) -> None:
        self.base = base
        shape = None
        if self.base.internal_weights and self.base.weight_numel > 0:
            shape = tuple(int(v) for v in self.base.weight.shape)
        self._init_single_weight_elora(shape, rank=rank, alpha=alpha)
        if freeze_base:
            freeze_module_parameters(self.base)


class CueqLoRALinear(_BaseCueqELoRAWrapper):
    def __init__(
        self,
        *args,
        rank: int = 16,
        alpha: float = 16.0,
        elora_freeze_base: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self._init_base_wrapper(
            cuet.Linear(*args, **kwargs),
            rank=rank,
            alpha=alpha,
            freeze_base=elora_freeze_base,
        )

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor | None = None,
        weight_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._has_unmerged_elora() and self.base.internal_weights:
            if weight is not None:
                raise ValueError("Internal weights are used, weight should be None")
            effective_weight = self._apply_single_weight_elora(self.base.weight)
            input_indices = {}
            if self.base.weight_classes > 1:
                if weight_indices is None:
                    raise ValueError("weight_indices should be provided if weight_classes > 1")
                input_indices[0] = weight_indices
            output = self.base.f(
                [effective_weight, self.base.transpose_in(x)],
                input_indices=input_indices,
            )
            return self.base.transpose_out(output[0])
        if weight is None and self._has_unmerged_elora():
            weight = self._apply_single_weight_elora(self.base.weight)
        return self.base(x, weight=weight, weight_indices=weight_indices)


class CueqLoRATensorProduct(_BaseCueqELoRAWrapper):
    base_ctor = None

    def __init__(
        self,
        *args,
        rank: int = 16,
        alpha: float = 16.0,
        elora_freeze_base: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if self.base_ctor is None:
            raise TypeError(f"{self.__class__.__name__} must define base_ctor")
        self._init_base_wrapper(
            self.base_ctor(*args, **kwargs),
            rank=rank,
            alpha=alpha,
            freeze_base=elora_freeze_base,
        )

    def _forward_with_internal_weight(
        self,
        *inputs,
        effective_weight: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        x1 = self.base.transpose_in1(inputs[0])
        x2 = self.base.transpose_in2(inputs[1])
        if "indices_1" in kwargs or "indices_2" in kwargs or "indices_out" in kwargs:
            indices_in = {}
            indices_1 = kwargs.get("indices_1")
            indices_2 = kwargs.get("indices_2")
            indices_out = kwargs.get("indices_out")
            size_out = kwargs.get("size_out")
            if indices_1 is not None:
                indices_in[1] = indices_1
            if indices_2 is not None:
                indices_in[2] = indices_2
            output_indices = None
            output_shapes = {}
            if indices_out is not None:
                if size_out is None:
                    raise ValueError("size_out should be provided if indices_out is provided")
                output_indices = {0: indices_out}
                output_shapes = {0: torch.empty(size_out, 1, device=x1.device)}
            output = self.base.f(
                [effective_weight, x1, x2],
                input_indices=indices_in,
                output_shapes=output_shapes,
                output_indices=output_indices,
            )
            return self.base.transpose_out(output[0])
        output = self.base.f([effective_weight, x1, x2])
        return self.base.transpose_out(output[0])

    def forward(self, *inputs, weight: torch.Tensor | None = None, **kwargs):
        if self._has_unmerged_elora() and getattr(self.base, "weight", None) is not None:
            if weight is not None:
                raise ValueError("Internal weights are used, weight should be None")
            effective_weight = self._apply_single_weight_elora(self.base.weight)
            return self._forward_with_internal_weight(
                *inputs,
                effective_weight=effective_weight,
                **kwargs,
            )
        if weight is None and self._has_unmerged_elora():
            weight = self._apply_single_weight_elora(self.base.weight)
        return self.base(*inputs, weight=weight, **kwargs)


class CueqLoRAFullyConnectedTensorProduct(CueqLoRATensorProduct):
    base_ctor = cuet.FullyConnectedTensorProduct


class CueqLoRAChannelWiseTensorProduct(CueqLoRATensorProduct):
    base_ctor = cuet.ChannelWiseTensorProduct


class CueqLoRASymmetricContraction(SingleWeightLoRAModuleMixin, CuetSymmetricContraction):
    def __init__(
        self,
        *args,
        rank: int = 16,
        alpha: float = 16.0,
        elora_freeze_base: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._init_single_weight_elora(
            tuple(int(v) for v in self.sc.weight.shape),
            rank=rank,
            alpha=alpha,
        )
        if elora_freeze_base:
            freeze_module_parameters(self.sc)

    def forward(self, x: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        weight = self._apply_single_weight_elora(self.sc.weight)
        return self.forward_with_weight(x, attrs, weight)

    def merge_elora(self) -> None:
        self._merge_single_weight_elora(self.sc.weight)
