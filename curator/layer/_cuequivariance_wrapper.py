# The code is borrowed from https://github.com/ACEsuit/mace/blob/main/mace/modules/wrapper_ops.py with modifications

from e3nn import o3
import torch
import numpy as np
import warnings
import types
from typing import Iterator, List, Optional
import itertools
from ._symmetric_contraction import SymmetricContraction

try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet

    IS_CUET_AVAILABLE = True


except ImportError:
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
    warnings.warn("cuequivariance is not available. Cuequivariance acceleration will be disabled.")

# set up a global flag to enable or disable cuequivariance
USE_CUEQ_GLOBAL = False
def set_use_cueq(value: bool):
    """Set global flag for using cueq."""
    global USE_CUEQ_GLOBAL
    USE_CUEQ_GLOBAL = value

class Linear:
    """Returns an e3nn linear layer or cueq linear layer"""
    def __new__(
        cls,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        use_cueq: bool = True,
        *args,
        **kwargs,
    ):
        if IS_CUET_AVAILABLE and use_cueq and USE_CUEQ_GLOBAL:
            return cuet.Linear(
                cue.Irreps(CUEQ_GROUP, irreps_in), 
                cue.Irreps(CUEQ_GROUP, irreps_out),
                layout=CUEQ_LAYOUT,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                *args, 
                **kwargs,
            )

        return o3.Linear(
            irreps_in,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            *args,
            **kwargs,
        )

class TensorProduct:
    """Wrapper around o3.TensorProduct/cuet.ChannelwiseTensorProduct"""
    def __new__(
        cls,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: Optional[List] = None,
        shared_weights: bool = False,
        internal_weights: bool = False,
        use_cueq: bool = True,
        *args,
        **kwargs,
    ):
        if IS_CUET_AVAILABLE and use_cueq and USE_CUEQ_GLOBAL:
            return cuet.ChannelWiseTensorProduct(
                cue.Irreps(CUEQ_GROUP, irreps_in1),
                cue.Irreps(CUEQ_GROUP, irreps_in2),
                cue.Irreps(CUEQ_GROUP, irreps_out),
                layout=CUEQ_LAYOUT,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
            )

        return o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions=instructions,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            *args,
            **kwargs,
        )
    
class FullyConnectedTensorProduct:
    """Wrapper around o3.FullyConnectedTensorProduct/cuet.FullyConnectedTensorProduct"""

    def __new__(
        cls,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        use_cueq: bool = True,
        *args,
        **kwargs,
    ):
        if IS_CUET_AVAILABLE and use_cueq and USE_CUEQ_GLOBAL:
            return cuet.FullyConnectedTensorProduct(
                cue.Irreps(CUEQ_GROUP, irreps_in1),
                cue.Irreps(CUEQ_GROUP, irreps_in2),
                cue.Irreps(CUEQ_GROUP, irreps_out),
                layout=CUEQ_LAYOUT,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                *args,
                **kwargs,
            )

        return o3.FullyConnectedTensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            *args,
            **kwargs,
        )

class CuetSymmetricContractionWrapper(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
        irreps_out,
        correlation,
        num_elements=None,
        *args,
        **kwargs
    ):
        super().__init__()
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

    def forward(self, x: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        if self.layout == cue.mul_ir:
            x = x.transpose(1, 2)
        index_attrs = torch.nonzero(attrs)[:, 1].int()

        # Because self.sc is the original instance, we call self.sc.forward.
        return self.sc.forward(x.flatten(1), index_attrs)

class SymmetricContractionWrapper:
    """Wrapper around SymmetricContraction/cuet.SymmetricContraction"""
    def __new__(
        cls,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: int,
        num_elements: Optional[int] = None,
        use_cueq: bool = True,
        *args,
        **kwargs,
    ):
        if IS_CUET_AVAILABLE and use_cueq and USE_CUEQ_GLOBAL:
            return CuetSymmetricContractionWrapper(
                irreps_in, irreps_out, correlation, num_elements, *args, **kwargs
            )
        
        return SymmetricContraction(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            correlation=correlation,
            num_elements=num_elements,
            *args,
            **kwargs,
        )