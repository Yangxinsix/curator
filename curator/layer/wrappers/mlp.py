from __future__ import annotations

import math
from typing import Callable

import torch

from .scalar import build_scalar_linear, build_scalar_mlp
from .utils import ensure_torch_serialization_compat

ensure_torch_serialization_compat()

from e3nn.math import normalize2mom

from ..nonlinearities import ShiftedSoftPlus


def _resolve_elora_hyperparameters(
    *,
    rank: int,
    alpha: float,
    elora_rank: int | None,
    elora_alpha: float | None,
) -> tuple[int, float]:
    if elora_rank is not None:
        rank = int(elora_rank)
    if elora_alpha is not None:
        alpha = float(elora_alpha)
    return int(rank), float(alpha)


def build_fully_connected_net(
    hs,
    act=None,
    variance_in=1,
    variance_out=1,
    out_act=False,
    *,
    use_elora: bool = False,
    rank: int = 16,
    alpha: float = 16.0,
    elora_rank: int | None = None,
    elora_alpha: float | None = None,
    elora_freeze_base: bool = True,
):
    rank, alpha = _resolve_elora_hyperparameters(
        rank=rank,
        alpha=alpha,
        elora_rank=elora_rank,
        elora_alpha=elora_alpha,
    )
    dims = [int(dim) for dim in hs]
    normalized_act = normalize2mom(act) if act is not None else None

    var_in = float(variance_in)

    def _build_layer(h_in: int, h_out: int, is_last: bool):
        nonlocal var_in
        var_out = float(variance_out) if is_last else 1.0
        layer_act = normalized_act if (not is_last or out_act) else None
        weight_divisor = (
            math.sqrt(float(h_in) * var_in)
            if layer_act is not None
            else math.sqrt(float(h_in) * var_in / var_out)
        )
        output_multiplier = math.sqrt(var_out) if layer_act is not None else 1.0
        layer = build_scalar_linear(
            h_in,
            h_out,
            use_elora=use_elora,
            rank=rank,
            alpha=alpha,
            freeze_base=elora_freeze_base,
            bias=False,
            activation=layer_act,
            weight_init="randn",
            weight_divisor=weight_divisor,
            output_multiplier=output_multiplier,
        )
        var_in = var_out
        return layer

    return build_scalar_mlp(
        dims,
        layer_builder=_build_layer,
        activation=normalized_act,
        preset="e3nn_fc",
    )


def build_convnet_radial_mlp(
    *,
    input_dim: int,
    output_dim: int,
    hidden_layers_depth: int,
    hidden_layers_width: int,
    nonlinearity: str,
    use_elora: bool = False,
    rank: int = 16,
    alpha: float = 16.0,
    elora_rank: int | None = None,
    elora_alpha: float | None = None,
    elora_freeze_base: bool = True,
):
    rank, alpha = _resolve_elora_hyperparameters(
        rank=rank,
        alpha=alpha,
        elora_rank=elora_rank,
        elora_alpha=elora_alpha,
    )
    dims = [input_dim] + hidden_layers_depth * [hidden_layers_width] + [output_dim]
    act: Callable = {
        "ssp": ShiftedSoftPlus,
        "silu": torch.nn.functional.silu,
    }[nonlinearity]
    layer_index = 0

    def _build_layer(h_in: int, h_out: int, is_last: bool):
        nonlocal layer_index
        gain = 1.0 if layer_index == 0 else math.sqrt(2.0)
        base_alpha = gain / math.sqrt(max(1, h_in))
        layer = build_scalar_linear(
            h_in,
            h_out,
            use_elora=use_elora,
            rank=rank,
            alpha=alpha,
            freeze_base=elora_freeze_base,
            bias=False,
            activation=None if is_last else act,
            weight_init="uniform_sqrt3",
            weight_multiplier=base_alpha,
            lora_init_scale=1.0 / math.sqrt(max(1, h_in)),
        )
        layer_index += 1
        return layer

    return build_scalar_mlp(
        dims,
        layer_builder=_build_layer,
        activation=act,
        preset="convnet_radial",
    )


__all__ = [
    "build_fully_connected_net",
    "build_convnet_radial_mlp",
]
