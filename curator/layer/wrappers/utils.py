from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.serialization as torch_serialization
from torch import nn


def mark_lora_parameter(parameter: nn.Parameter | None) -> None:
    if parameter is not None:
        setattr(parameter, "_is_elora_parameter", True)


def is_elora_parameter(name: str, parameter: nn.Parameter) -> bool:
    return "lora_" in name or bool(getattr(parameter, "_is_elora_parameter", False))


def freeze_parameters(parameters: Iterable[nn.Parameter | None]) -> None:
    for parameter in parameters:
        if parameter is not None:
            parameter.requires_grad_(False)


def freeze_module_parameters(module: nn.Module, *, recurse: bool = True) -> None:
    freeze_parameters(module.parameters(recurse=recurse))


def ensure_torch_serialization_compat() -> None:
    try:
        torch_serialization.add_safe_globals([slice])
    except Exception:
        pass


def clamp_rank(rows: int, cols: int, rank: int) -> int:
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    rank = int(rank)
    return max(1, min(rows, cols, rank))


def _matrix_shape_from_tensor(shape: Sequence[int]) -> Tuple[int, int]:
    if len(shape) == 0:
        return (1, 1)
    if len(shape) == 1:
        return (int(shape[0]), 1)
    rows = int(math.prod(shape[:-1]))
    cols = int(shape[-1])
    return (rows, cols)


def make_lora_matrix_pair(
    rows: int,
    cols: int,
    *,
    rank: int,
    init_scale: float = 1.0,
    shape_prefix: Sequence[int] = (),
) -> Tuple[nn.Parameter, nn.Parameter]:
    prefix = tuple(int(v) for v in shape_prefix)
    lora_a = nn.Parameter(
        torch.randn(*prefix, int(rows), int(rank)) * (float(init_scale) / max(1, int(rows)))
    )
    lora_b = nn.Parameter(torch.zeros(*prefix, int(rank), int(cols)))
    mark_lora_parameter(lora_a)
    mark_lora_parameter(lora_b)
    return lora_a, lora_b


class LowRankTensorAdapter(nn.Module):
    def __init__(
        self,
        shape: Sequence[int],
        *,
        rank: int,
        alpha: float,
        init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.shape = tuple(int(v) for v in shape)
        rows, cols = _matrix_shape_from_tensor(self.shape)
        self.rank = clamp_rank(rows, cols, rank)
        self.alpha = float(alpha)
        self.scale = float(self.alpha / self.rank)
        self.lora_A, self.lora_B = make_lora_matrix_pair(
            rows,
            cols,
            rank=self.rank,
            init_scale=init_scale,
        )

    def delta(self) -> torch.Tensor:
        return (self.lora_A @ self.lora_B).reshape(self.shape) * self.scale


class SingleWeightLoRAModuleMixin:
    adapter: LowRankTensorAdapter | None
    _merged: bool

    def _init_single_weight_elora(
        self,
        shape: Sequence[int] | None,
        *,
        rank: int,
        alpha: float,
        init_scale: float = 1.0,
    ) -> None:
        self._merged = False
        self.adapter = None
        if shape is None:
            return
        self.adapter = LowRankTensorAdapter(
            tuple(int(v) for v in shape),
            rank=rank,
            alpha=alpha,
            init_scale=init_scale,
        )

    def _has_unmerged_elora(self) -> bool:
        return self.adapter is not None and not self._merged

    def _apply_single_weight_elora(self, weight: torch.Tensor) -> torch.Tensor:
        if not self._has_unmerged_elora():
            return weight
        return weight + self.adapter.delta()

    def _merge_single_weight_elora(self, target: nn.Parameter) -> bool:
        if not self._has_unmerged_elora():
            return False
        target.data = target.data + self.adapter.delta().detach()
        self._merged = True
        return True


class PathLowRankAdapter(nn.Module):
    def __init__(
        self,
        path_shapes: Iterable[Sequence[int]],
        *,
        rank: int,
        alpha: float,
        flatten_dims: int,
    ) -> None:
        super().__init__()
        self.flatten_dims = int(flatten_dims)
        self.adapters = nn.ModuleList()
        for shape in path_shapes:
            dims = tuple(int(v) for v in shape)
            if len(dims) != self.flatten_dims + 1:
                raise ValueError(
                    f"Expected path shape of length {self.flatten_dims + 1}, got {dims}"
                )
            rows = int(math.prod(dims[: self.flatten_dims]))
            cols = int(dims[self.flatten_dims])
            self.adapters.append(
                LowRankTensorAdapter(
                    (rows, cols),
                    rank=rank,
                    alpha=alpha,
                )
            )

    def delta(self) -> torch.Tensor:
        if len(self.adapters) == 0:
            return torch.zeros(0)
        return torch.cat([adapter.delta().reshape(-1) for adapter in self.adapters], dim=-1)

def collect_named_parameters(
    module: nn.Module,
    *,
    include=None,
    exclude=None,
    seen: set[int] | None = None,
) -> List[nn.Parameter]:
    if seen is None:
        seen = set()
    params: List[nn.Parameter] = []
    for name, param in module.named_parameters():
        if not isinstance(param, nn.Parameter):
            continue
        if include is not None and not include(name, param):
            continue
        if exclude is not None and exclude(name, param):
            continue
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)
        params.append(param)
    return params
