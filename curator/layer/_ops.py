from __future__ import annotations

from .wrappers.backends import get_backend, get_backend_name
from .wrappers.cueq import IS_CUET_AVAILABLE


def ScalarLinear(*args, **kwargs):
    return get_backend().make_scalar_linear(*args, **kwargs)


def Linear(*args, **kwargs):
    return get_backend().make_linear(*args, **kwargs)


def TensorProduct(*args, **kwargs):
    return get_backend().make_tensor_product(*args, **kwargs)


def FullyConnectedTensorProduct(*args, **kwargs):
    return get_backend().make_fully_connected_tensor_product(*args, **kwargs)


def SymmetricContraction(*args, **kwargs):
    return get_backend().make_symmetric_contraction(*args, **kwargs)


def FullyConnectedNet(*args, **kwargs):
    return get_backend().make_fully_connected_net(*args, **kwargs)


def build_convnet_radial_mlp(*args, **kwargs):
    return get_backend().make_convnet_radial_mlp(*args, **kwargs)


__all__ = [
    "IS_CUET_AVAILABLE",
    "ScalarLinear",
    "Linear",
    "TensorProduct",
    "FullyConnectedTensorProduct",
    "SymmetricContraction",
    "FullyConnectedNet",
    "build_convnet_radial_mlp",
    "get_backend_name",
]
