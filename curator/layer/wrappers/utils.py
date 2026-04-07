from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Iterable, Iterator, Sequence, Tuple

import torch
import torch.serialization as torch_serialization
from torch import nn


def mark_lora_parameter(parameter: nn.Parameter | None) -> None:
    if parameter is not None:
        setattr(parameter, "_is_elora_parameter", True)


def is_elora_parameter(name: str, parameter: nn.Parameter) -> bool:
    return "lora_" in name or bool(getattr(parameter, "_is_elora_parameter", False))


def freeze_module_parameters(module: nn.Module, *, recurse: bool = True) -> None:
    for parameter in module.parameters(recurse=recurse):
        if parameter is not None:
            parameter.requires_grad_(False)


def ensure_torch_serialization_compat() -> None:
    try:
        torch_serialization.add_safe_globals([slice])
    except Exception:
        pass


@contextmanager
def temporary_default_dtype(dtype: torch.dtype | None) -> Iterator[None]:
    if dtype is None:
        yield
        return
    previous_dtype = torch.get_default_dtype()
    if dtype == previous_dtype:
        yield
        return
    try:
        torch.set_default_dtype(dtype)
    except TypeError:
        yield
        return
    try:
        yield
    finally:
        torch.set_default_dtype(previous_dtype)


def infer_module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype | None]:
    device = torch.device("cpu")
    dtype = None
    for parameter in module.parameters():
        device = parameter.device
        if parameter.is_floating_point() and dtype is None:
            dtype = parameter.dtype
        if dtype is not None:
            break
    if dtype is None:
        for buffer in module.buffers():
            device = buffer.device
            if buffer.is_floating_point():
                dtype = buffer.dtype
                break
    return device, dtype


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
