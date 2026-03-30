from __future__ import annotations

import math
from typing import Optional

import torch
from e3nn import o3

from .utils import freeze_module_parameters, make_lora_matrix_pair


class _ELoRAModuleMixin:
    lora_rank: int
    lora_alpha: float
    lora_scale: float

    def _init_elora_state(self, rank: int, alpha: float) -> None:
        self.lora_rank = int(rank)
        self.lora_alpha = float(alpha)
        self.lora_scale = self.lora_alpha / float(max(1, self.lora_rank))
        self._merged = False
        self.lora_A = torch.nn.ParameterList()
        self.lora_B = torch.nn.ParameterList()

    def _append_lora_matrix(
        self,
        left_dim: int,
        right_dim: int,
        *,
        shape_prefix,
        init_scale: float,
    ) -> None:
        lora_a, lora_b = make_lora_matrix_pair(
            left_dim,
            right_dim,
            rank=self.lora_rank,
            shape_prefix=shape_prefix,
            init_scale=init_scale,
        )
        self.lora_A.append(lora_a)
        self.lora_B.append(lora_b)

    def _apply_elora_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if self._merged or len(self.lora_A) == 0:
            return weight
        delta_chunks = []
        for lora_a, lora_b in zip(self.lora_A, self.lora_B):
            delta_chunks.append(torch.matmul(lora_a, lora_b).flatten(start_dim=-2))
        delta = torch.cat(delta_chunks, dim=-1)
        return weight + self.lora_scale * delta

    def merge_elora(self) -> None:
        if self._merged or len(self.lora_A) == 0:
            return
        self.weight.data = self._apply_elora_weight(self.weight.data)
        self._merged = True


class ELoRALinear(_ELoRAModuleMixin, o3.Linear):
    def __init__(
        self,
        *args,
        elora_rank: int = 16,
        elora_alpha: float = 16.0,
        elora_freeze_base: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._init_elora_state(elora_rank, elora_alpha)
        if self.internal_weights and self.weight_numel > 0 and self.lora_rank > 0:
            prefix = tuple(self.weight.shape[:-1])
            for instruction in self.instructions:
                if instruction.i_in == -1:
                    continue
                self._append_lora_matrix(
                    int(instruction.path_shape[0]),
                    int(instruction.path_shape[1]),
                    shape_prefix=prefix,
                    init_scale=1.0 / math.sqrt(max(1, int(instruction.path_shape[0]))),
                )
        if elora_freeze_base:
            freeze_module_parameters(self, recurse=False)

    def forward(
        self,
        features,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ):
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError("Weights must be provided when internal_weights = False")
            weight = self._apply_elora_weight(self.weight)
        if bias is None:
            if self.bias_numel > 0 and not self.internal_weights:
                raise RuntimeError("Biases must be provided when internal_weights = False")
            bias = self.bias
        return self._compiled_main(features, weight, bias)


class ELoRATensorProduct(_ELoRAModuleMixin, o3.TensorProduct):
    def __init__(
        self,
        *args,
        elora_rank: int = 16,
        elora_alpha: float = 16.0,
        elora_freeze_base: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._init_elora_state(elora_rank, elora_alpha)
        if self.internal_weights and self.weight_numel > 0 and self.lora_rank > 0:
            prefix = tuple(self.weight.shape[:-1])
            for instruction in self.instructions:
                if not instruction.has_weight:
                    continue
                left_dim = int(math.prod(instruction.path_shape[:-1]))
                right_dim = int(instruction.path_shape[-1])
                self._append_lora_matrix(
                    left_dim,
                    right_dim,
                    shape_prefix=prefix,
                    init_scale=1.0 / math.sqrt(max(1, left_dim)),
                )
        if elora_freeze_base:
            freeze_module_parameters(self, recurse=False)

    def forward(self, x, y, weight: Optional[torch.Tensor] = None):
        assert x.shape[-1] == self._in1_dim, "Incorrect last dimension for x"
        assert y.shape[-1] == self._in2_dim, "Incorrect last dimension for y"
        real_weight = self._get_weights(weight)
        if weight is None and self.internal_weights:
            real_weight = self._apply_elora_weight(real_weight)
        return self._compiled_main_left_right(x, y, real_weight)


class ELoRAFullyConnectedTensorProduct(ELoRATensorProduct, o3.FullyConnectedTensorProduct):
    pass


__all__ = [
    "ELoRALinear",
    "ELoRATensorProduct",
    "ELoRAFullyConnectedTensorProduct",
]
