from typing import Optional, Union

import torch
from torch import nn


class VarianceScale(nn.Module):
    """Fixed scale fitted from hidden-feature variance."""

    _has_reference: Optional[bool]
    _sum: Optional[torch.Tensor]
    _sum_sq: Optional[torch.Tensor]
    _ref_sum: Optional[torch.Tensor]
    _ref_sum_sq: Optional[torch.Tensor]

    def __init__(self, scale: Optional[float] = None):
        super().__init__()
        self.register_buffer("scale", torch.tensor(1.0 if scale is None else float(scale)))
        self.register_buffer("fitted", torch.tensor(scale is not None, dtype=torch.bool))
        self._observing = False
        self._has_reference = None
        self._count = 0
        self._sum = None
        self._sum_sq = None
        self._ref_count = 0
        self._ref_sum = None
        self._ref_sum_sq = None

    def start_fitting(self) -> None:
        self.reset()
        self._observing = True

    def reset(self) -> None:
        self.scale.fill_(1.0)
        self.fitted.fill_(False)
        self._observing = False
        self._has_reference = None
        self._count = 0
        self._sum = None
        self._sum_sq = None
        self._ref_count = 0
        self._ref_sum = None
        self._ref_sum_sq = None

    def stop_fitting(self) -> None:
        self._observing = False

    def set_scale(self, scale: Union[float, torch.Tensor]) -> None:
        value = torch.as_tensor(scale, device=self.scale.device, dtype=self.scale.dtype)
        if value.numel() != 1 or not torch.isfinite(value) or value <= 0:
            raise ValueError("VarianceScale scale must be a finite positive scalar.")
        self.scale.copy_(value)
        self.fitted.fill_(True)

    @staticmethod
    def _samples(x: torch.Tensor) -> torch.Tensor:
        x = x.detach().to(dtype=torch.float64)
        return x.reshape(-1, 1) if x.ndim == 1 else x.reshape(-1, x.shape[-1])

    @staticmethod
    def _variance(
        count: int,
        total: Optional[torch.Tensor],
        total_sq: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if count == 0 or total is None or total_sq is None:
            raise RuntimeError("VarianceScale did not observe any samples.")
        return (total_sq / count - (total / count).square()).mean()

    @torch.no_grad()
    def _observe(self, x: torch.Tensor, ref: Optional[torch.Tensor]) -> None:
        has_reference = ref is not None
        if self._has_reference is not None and self._has_reference != has_reference:
            raise ValueError("Pass ref either for every observed batch or for none of them.")
        self._has_reference = has_reference

        samples = self._samples(x)
        self._count += samples.shape[0]
        batch_sum = samples.sum(dim=0)
        batch_sum_sq = samples.square().sum(dim=0)
        self._sum = batch_sum if self._sum is None else self._sum + batch_sum
        self._sum_sq = batch_sum_sq if self._sum_sq is None else self._sum_sq + batch_sum_sq

        if ref is not None:
            samples = self._samples(ref)
            self._ref_count += samples.shape[0]
            batch_sum = samples.sum(dim=0)
            batch_sum_sq = samples.square().sum(dim=0)
            self._ref_sum = batch_sum if self._ref_sum is None else self._ref_sum + batch_sum
            self._ref_sum_sq = (
                batch_sum_sq if self._ref_sum_sq is None else self._ref_sum_sq + batch_sum_sq
            )

    def fit(self) -> float:
        variance = self._variance(self._count, self._sum, self._sum_sq)
        target = (
            self._variance(self._ref_count, self._ref_sum, self._ref_sum_sq)
            if self._has_reference
            else variance.new_tensor(1.0)
        )
        if not torch.isfinite(variance) or variance <= 0:
            raise ValueError("Observed hidden features must have positive finite variance.")
        if not torch.isfinite(target) or target <= 0:
            raise ValueError("Reference features must have positive finite variance.")
        self.set_scale(torch.sqrt(target / variance))
        self.stop_fitting()
        return self.scale.item()

    def forward(
        self,
        x: torch.Tensor,
        ref: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not torch.jit.is_scripting() and self._observing:
            self._observe(x, ref)
        return x * self.scale if bool(self.fitted.item()) else x
