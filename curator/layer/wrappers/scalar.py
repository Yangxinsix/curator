from __future__ import annotations

import math
from typing import Callable, Iterable, Optional, Sequence

import torch
from torch import nn

from .utils import SingleWeightLoRAModuleMixin, freeze_module_parameters


def _calculate_fan_in(in_features: int) -> int:
    return max(1, int(in_features))


class _BaseScalarLinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        activation: Optional[Callable] = None,
        weight_init: str,
        weight_multiplier: float = 1.0,
        weight_divisor: float = 1.0,
        output_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.activation = activation
        self.weight_init = str(weight_init)
        self.weight_multiplier = float(weight_multiplier)
        self.weight_divisor = float(weight_divisor)
        self.output_multiplier = float(output_multiplier)
        self.weight = nn.Parameter(torch.empty((self.in_features, self.out_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight_init == "randn":
            self.weight.data.normal_()
            if self.bias is not None:
                self.bias.data.zero_()
            return
        if self.weight_init == "uniform_sqrt3":
            nn.init.uniform_(self.weight, -math.sqrt(3.0), math.sqrt(3.0))
            if self.bias is not None:
                nn.init.uniform_(self.bias, -math.sqrt(3.0), math.sqrt(3.0))
            return
        if self.weight_init == "torch_linear":
            nn.init.kaiming_uniform_(self.weight.t(), a=math.sqrt(5.0))
            if self.bias is not None:
                fan_in = _calculate_fan_in(self.in_features)
                bound = 1.0 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            return
        raise ValueError(f"Unsupported scalar weight init mode: {self.weight_init!r}")

    def _effective_weight(self) -> torch.Tensor:
        return self.weight

    def _scaled_weight(self) -> torch.Tensor:
        weight = self._effective_weight()
        if self.weight_multiplier != 1.0:
            weight = weight * self.weight_multiplier
        if self.weight_divisor != 1.0:
            weight = weight / self.weight_divisor
        return weight

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs @ self._scaled_weight()
        if self.bias is not None:
            outputs = outputs + self.bias
        if self.activation is not None:
            outputs = self.activation(outputs)
        if self.output_multiplier != 1.0:
            outputs = outputs * self.output_multiplier
        return outputs


class ScalarLinearLayer(_BaseScalarLinearLayer):
    pass


class ELoRAScalarLinearLayer(SingleWeightLoRAModuleMixin, _BaseScalarLinearLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        lora_init_scale: float = 1.0,
        freeze_base: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(in_features, out_features, **kwargs)
        self._init_single_weight_elora(
            (self.in_features, self.out_features),
            rank=rank,
            alpha=alpha,
            init_scale=lora_init_scale,
        )
        if freeze_base:
            freeze_module_parameters(self, recurse=False)

    def _effective_weight(self) -> torch.Tensor:
        return self._apply_single_weight_elora(self.weight)

    def merge_elora(self) -> None:
        self._merge_single_weight_elora(self.weight)


class ScalarMLP(nn.Sequential):
    def __init__(
        self,
        dims: Sequence[int],
        layers: Iterable[nn.Module],
        *,
        activation: Optional[Callable] = None,
        preset: str = "generic",
    ) -> None:
        super().__init__()
        self.hs = [int(dim) for dim in dims]
        self.num_layers = max(0, len(self.hs) - 1)
        self.preset = str(preset)
        for index, layer in enumerate(layers):
            self.add_module(f"layer{index}", layer)
        object.__setattr__(self, "activation", activation)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(preset={self.preset!r}, hs={self.hs!r})"


def build_scalar_linear(
    in_features: int,
    out_features: int,
    *,
    use_elora: bool = False,
    rank: int = 16,
    alpha: float = 16.0,
    freeze_base: bool = True,
    bias: bool = True,
    activation: Optional[Callable] = None,
    weight_init: str = "torch_linear",
    weight_multiplier: float = 1.0,
    weight_divisor: float = 1.0,
    output_multiplier: float = 1.0,
    lora_init_scale: float = 1.0,
) -> nn.Module:
    layer_cls = ELoRAScalarLinearLayer if use_elora else ScalarLinearLayer
    kwargs = {
        "bias": bool(bias),
        "activation": activation,
        "weight_init": weight_init,
        "weight_multiplier": float(weight_multiplier),
        "weight_divisor": float(weight_divisor),
        "output_multiplier": float(output_multiplier),
    }
    if use_elora:
        kwargs.update(
            {
                "rank": int(rank),
                "alpha": float(alpha),
                "freeze_base": bool(freeze_base),
                "lora_init_scale": float(lora_init_scale),
            }
        )
    return layer_cls(
        int(in_features),
        int(out_features),
        **kwargs,
    )


def build_scalar_mlp(
    dims: Sequence[int],
    *,
    layer_builder: Callable[[int, int, bool], nn.Module],
    activation: Optional[Callable] = None,
    preset: str = "generic",
) -> ScalarMLP:
    dims = [int(dim) for dim in dims]
    layers = [
        layer_builder(h_in, h_out, index == len(dims) - 2)
        for index, (h_in, h_out) in enumerate(zip(dims, dims[1:]))
    ]
    return ScalarMLP(
        dims,
        layers,
        activation=activation,
        preset=preset,
    )


__all__ = [
    "ScalarLinearLayer",
    "ELoRAScalarLinearLayer",
    "ScalarMLP",
    "build_scalar_linear",
    "build_scalar_mlp",
]
