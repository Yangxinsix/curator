"""Optional CUDA cubic basis for the two trained Small+ACVR proposal splines."""

from functools import cache
from pathlib import Path

import torch
from torch.nn import functional as F
from operator_kan_architectures import OperatorSplineLayer


@cache
def extension():
    from torch.utils.cpp_extension import load

    return load(
        name="acvr_cubic_spline_cuda_v1",
        sources=[str(Path(__file__).with_name("cubic_spline.cu"))],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
    )


class CubicBasis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, order):
        ctx.save_for_backward(x)
        ctx.order = order
        return extension().evaluate(x.contiguous(), order)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        if ctx.order == 3:
            return torch.zeros_like(x), None
        deriv = CubicBasis.apply(x, ctx.order + 1)
        return (grad_output * deriv).sum(-1), None


class NativeSplineLayer(OperatorSplineLayer):
    def forward(self, values):
        if not values.is_cuda or values.dtype not in (torch.float32, torch.float64):
            return super().forward(values)
        base_input = F.silu(values) if self.activation == "silu" else values
        bounded = torch.tanh((values - self.center) / self.scale)
        basis = CubicBasis.apply(bounded, 0)
        return self.base(base_input) + F.linear(
            basis.flatten(-2), self.spline_weight.flatten(1)
        )


def enable_native_splines(proposal):
    layers = (proposal.operator_head.first, proposal.operator_head.last)
    for layer in layers:
        expected = torch.arange(
            -4, 5, device=layer.knots.device, dtype=layer.knots.dtype
        )
        if not isinstance(layer, OperatorSplineLayer) or not torch.equal(
            layer.knots, expected
        ):
            raise ValueError(
                "Native proposal splines require the fixed cubic grid [-4,...,4]"
            )
    for layer in layers:
        layer.__class__ = NativeSplineLayer
