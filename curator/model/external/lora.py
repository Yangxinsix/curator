from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from curator.layer.wrappers.utils import mark_lora_parameter


class ExternalLoRALinear(nn.Module):
    """LoRA wrapper for plain torch Linear layers used by external backbones."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"ExternalLoRALinear expects nn.Linear, got {type(base)!r}.")
        self.base = base
        self.rank = max(1, min(int(rank), int(base.out_features), int(base.in_features)))
        self.alpha = float(alpha)
        self.scale = self.alpha / float(self.rank)
        self._merged = False

        weight = base.weight
        self.lora_A = nn.Parameter(
            torch.randn(
                int(base.out_features),
                self.rank,
                device=weight.device,
                dtype=weight.dtype,
            )
            / max(1, int(base.out_features))
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                self.rank,
                int(base.in_features),
                device=weight.device,
                dtype=weight.dtype,
            )
        )
        mark_lora_parameter(self.lora_A)
        mark_lora_parameter(self.lora_B)

        if freeze_base:
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        if self._merged:
            return self.base.weight
        return self.base.weight + (self.lora_A @ self.lora_B) * self.scale

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight, self.bias)

    def merge_lora(self) -> None:
        if self._merged:
            return
        self.base.weight.data = self.weight.detach()
        self._merged = True


def patch_linear_lora(
    modules: nn.Module | Iterable[nn.Module],
    *,
    rank: int,
    alpha: float,
    freeze_base: bool = True,
) -> int:
    """Replace plain nn.Linear children under modules with ExternalLoRALinear."""

    if isinstance(modules, nn.Module):
        roots = [modules]
    else:
        roots = list(modules)

    patched = 0
    for root in roots:
        patched += _patch_linear_lora_inplace(
            root,
            rank=rank,
            alpha=alpha,
            freeze_base=freeze_base,
        )
    return patched


def _patch_linear_lora_inplace(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    freeze_base: bool,
) -> int:
    patched = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, ExternalLoRALinear):
            continue
        if isinstance(child, nn.Linear):
            setattr(
                module,
                child_name,
                ExternalLoRALinear(
                    child,
                    rank=rank,
                    alpha=alpha,
                    freeze_base=freeze_base,
                ),
            )
            patched += 1
            continue
        patched += _patch_linear_lora_inplace(
            child,
            rank=rank,
            alpha=alpha,
            freeze_base=freeze_base,
        )
    return patched


__all__ = ["ExternalLoRALinear", "patch_linear_lora"]
