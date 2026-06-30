from .wrappers import cueq as _cueq
from .wrappers.cueq import (
    CUEQ_GROUP,
    CUEQ_LAYOUT,
    IS_CUET_AVAILABLE,
    CuetSymmetricContraction,
    make_e3nn_symmetric_contraction,
    make_fully_connected_tensor_product,
    make_linear,
    make_symmetric_contraction,
    make_tensor_product,
)

O3_e3nn = getattr(_cueq, "O3_e3nn", None)
Linear = make_linear
TensorProduct = make_tensor_product
FullyConnectedTensorProduct = make_fully_connected_tensor_product
SymmetricContraction = CuetSymmetricContraction
CuetSymmetricContractionWrapper = CuetSymmetricContraction

__all__ = [
    "CUEQ_GROUP",
    "CUEQ_LAYOUT",
    "IS_CUET_AVAILABLE",
    "CuetSymmetricContraction",
    "CuetSymmetricContractionWrapper",
    "FullyConnectedTensorProduct",
    "Linear",
    "O3_e3nn",
    "SymmetricContraction",
    "TensorProduct",
    "make_e3nn_symmetric_contraction",
    "make_fully_connected_tensor_product",
    "make_linear",
    "make_symmetric_contraction",
    "make_tensor_product",
]
