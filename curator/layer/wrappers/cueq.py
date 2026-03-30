from __future__ import annotations

import itertools
import warnings
from typing import Iterator, Optional

import numpy as np
import torch

from .utils import ensure_torch_serialization_compat

ensure_torch_serialization_compat()

from e3nn import o3

from .._symmetric_contraction import SymmetricContraction

try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet

    IS_CUET_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    warnings.warn(
        f"cuequivariance could not be loaded ({exc!r}); cueq acceleration disabled.",
        RuntimeWarning,
    )
    cue = None
    cuet = None
    IS_CUET_AVAILABLE = False


if IS_CUET_AVAILABLE:

    class O3_e3nn(cue.O3):
        def __mul__(  # pylint: disable=no-self-argument
            rep1: "O3_e3nn", rep2: "O3_e3nn"
        ) -> Iterator["O3_e3nn"]:
            return [O3_e3nn(l=ir.l, p=ir.p) for ir in cue.O3.__mul__(rep1, rep2)]

        @classmethod
        def clebsch_gordan(
            cls, rep1: "O3_e3nn", rep2: "O3_e3nn", rep3: "O3_e3nn"
        ) -> np.ndarray:
            rep1, rep2, rep3 = cls._from(rep1), cls._from(rep2), cls._from(rep3)

            if rep1.p * rep2.p == rep3.p:
                return o3.wigner_3j(rep1.l, rep2.l, rep3.l).numpy()[None] * np.sqrt(
                    rep3.dim
                )
            return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

        def __lt__(  # pylint: disable=no-self-argument
            rep1: "O3_e3nn", rep2: "O3_e3nn"
        ) -> bool:
            rep2 = rep1._from(rep2)
            return (rep1.l, rep1.p) < (rep2.l, rep2.p)

        @classmethod
        def iterator(cls) -> Iterator["O3_e3nn"]:
            for l in itertools.count(0):
                yield O3_e3nn(l=l, p=1 * (-1) ** l)
                yield O3_e3nn(l=l, p=-1 * (-1) ** l)

    CUEQ_LAYOUT = cue.mul_ir
    CUEQ_GROUP = O3_e3nn
else:
    CUEQ_LAYOUT = None
    CUEQ_GROUP = None


class CuetSymmetricContraction(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
        irreps_out,
        correlation,
        num_elements=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        if not IS_CUET_AVAILABLE:
            raise RuntimeError("cuequivariance is not available.")
        self.sc = cuet.SymmetricContraction(
            cue.Irreps(CUEQ_GROUP, irreps_in),
            cue.Irreps(CUEQ_GROUP, irreps_out),
            layout=CUEQ_LAYOUT,
            layout_in=cue.ir_mul,
            layout_out=CUEQ_LAYOUT,
            contraction_degree=correlation,
            num_elements=num_elements,
            original_mace=True,
            dtype=torch.get_default_dtype(),
            math_dtype=torch.get_default_dtype(),
            *args,
            **kwargs,
        )
        self.layout = CUEQ_LAYOUT
        self.contraction_degree = correlation
        self.num_elements = num_elements

    def _prepare_inputs(
        self, x: torch.Tensor, attrs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layout == cue.mul_ir:
            x = x.transpose(1, 2)
        index_attrs = torch.nonzero(attrs)[:, 1].int()
        return x.flatten(1), index_attrs

    def _project_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if self.sc.projection is not None:
            weight = torch.einsum("zau,ab->zbu", weight, self.sc.projection)
        return weight.flatten(1)

    def forward_with_weight(
        self,
        x: torch.Tensor,
        attrs: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        x_flat, index_attrs = self._prepare_inputs(x, attrs)
        weight_flat = self._project_weight(weight)
        output = self.sc.f(
            [weight_flat, self.sc.transpose_in(x_flat)],
            input_indices={0: index_attrs},
        )
        return self.sc.transpose_out(output[0])

    def forward(self, x: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_weight(x, attrs, self.sc.weight)


def _consume_irreps_args(args, kwargs, *names):
    args = list(args)
    values = []
    for name in names:
        if args:
            values.append(args.pop(0))
            continue
        if name not in kwargs:
            raise TypeError(f"Missing required irrep argument: {name}")
        values.append(kwargs.pop(name))
    return values, tuple(args), kwargs


def make_linear(*args, **kwargs):
    if not IS_CUET_AVAILABLE:
        raise RuntimeError("cuequivariance is not available.")
    (irreps_in, irreps_out), rest, kwargs = _consume_irreps_args(
        args,
        kwargs,
        "irreps_in",
        "irreps_out",
    )
    return cuet.Linear(
        cue.Irreps(CUEQ_GROUP, irreps_in),
        cue.Irreps(CUEQ_GROUP, irreps_out),
        *rest,
        layout=CUEQ_LAYOUT,
        **kwargs,
    )


def make_tensor_product(*args, **kwargs):
    if not IS_CUET_AVAILABLE:
        raise RuntimeError("cuequivariance is not available.")
    (irreps_in1, irreps_in2, irreps_out), rest, kwargs = _consume_irreps_args(
        args,
        kwargs,
        "irreps_in1",
        "irreps_in2",
        "irreps_out",
    )
    kwargs.pop("instructions", None)
    return cuet.ChannelWiseTensorProduct(
        cue.Irreps(CUEQ_GROUP, irreps_in1),
        cue.Irreps(CUEQ_GROUP, irreps_in2),
        cue.Irreps(CUEQ_GROUP, irreps_out),
        *rest,
        layout=CUEQ_LAYOUT,
        **kwargs,
    )


def make_fully_connected_tensor_product(*args, **kwargs):
    if not IS_CUET_AVAILABLE:
        raise RuntimeError("cuequivariance is not available.")
    (irreps_in1, irreps_in2, irreps_out), rest, kwargs = _consume_irreps_args(
        args,
        kwargs,
        "irreps_in1",
        "irreps_in2",
        "irreps_out",
    )
    return cuet.FullyConnectedTensorProduct(
        cue.Irreps(CUEQ_GROUP, irreps_in1),
        cue.Irreps(CUEQ_GROUP, irreps_in2),
        cue.Irreps(CUEQ_GROUP, irreps_out),
        *rest,
        layout=CUEQ_LAYOUT,
        **kwargs,
    )


def make_symmetric_contraction(*args, **kwargs):
    return CuetSymmetricContraction(*args, **kwargs)


def make_e3nn_symmetric_contraction(*args, **kwargs):
    return SymmetricContraction(*args, **kwargs)
