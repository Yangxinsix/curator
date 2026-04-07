from __future__ import annotations

from typing import Optional

import torch

from .oeq import OeqFullyConnectedTensorProduct, OeqLinear, OeqTensorProduct
from .utils import SingleWeightLoRAModuleMixin, freeze_module_parameters


class OeqLoRALinear(SingleWeightLoRAModuleMixin, OeqLinear):
    def __init__(
        self,
        *args,
        rank: int = 16,
        alpha: float = 16.0,
        elora_freeze_base: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        shape = tuple(int(v) for v in self.weight.shape) if self.internal_weights and self.weight_numel > 0 else None
        self._init_single_weight_elora(shape, rank=rank, alpha=alpha)
        if elora_freeze_base:
            freeze_module_parameters(self, recurse=False)
            if isinstance(self.bias, torch.nn.Parameter):
                self.bias.requires_grad_(False)

    def forward(
        self,
        features: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if weight is None and self.internal_weights:
            weight = self._apply_single_weight_elora(self.weight)
        return super().forward(features, weight=weight, bias=bias)

    def merge_elora(self) -> None:
        self._merge_single_weight_elora(self.weight)


class OeqLoRATensorProduct(SingleWeightLoRAModuleMixin, OeqTensorProduct):
    def __init__(
        self,
        *args,
        rank: int = 16,
        alpha: float = 16.0,
        elora_freeze_base: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        shape = tuple(int(v) for v in self.weight.shape) if self.internal_weights and self.weight_numel > 0 else None
        self._init_single_weight_elora(shape, rank=rank, alpha=alpha)
        if elora_freeze_base:
            freeze_module_parameters(self, recurse=False)

    def forward(self, x, y, weight: Optional[torch.Tensor] = None):
        if weight is None and self.internal_weights:
            weight = self._apply_single_weight_elora(self.weight)
        return super().forward(x, y, weight=weight)

    def merge_elora(self) -> None:
        self._merge_single_weight_elora(self.weight)


class OeqLoRAFullyConnectedTensorProduct(OeqLoRATensorProduct, OeqFullyConnectedTensorProduct):
    pass


__all__ = [
    "OeqLoRALinear",
    "OeqLoRATensorProduct",
    "OeqLoRAFullyConnectedTensorProduct",
]
