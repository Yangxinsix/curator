from __future__ import annotations

from .wrappers.backends import get_backend, get_backend_name
from .wrappers.cueq import IS_CUET_AVAILABLE


def ScalarLinear(*args, use_cueq: bool = True, use_elora: bool | None = None, **kwargs):
    return get_backend(use_cueq=use_cueq, use_elora=use_elora).make_scalar_linear(
        *args, **kwargs
    )


def Linear(*args, use_cueq: bool = True, use_elora: bool | None = None, **kwargs):
    return get_backend(use_cueq=use_cueq, use_elora=use_elora).make_linear(
        *args, **kwargs
    )


def TensorProduct(
    *args,
    use_cueq: bool = True,
    use_elora: bool | None = None,
    **kwargs,
):
    return get_backend(use_cueq=use_cueq, use_elora=use_elora).make_tensor_product(
        *args, **kwargs
    )


def FullyConnectedTensorProduct(
    *args,
    use_cueq: bool = True,
    use_elora: bool | None = None,
    **kwargs,
):
    return get_backend(
        use_cueq=use_cueq,
        use_elora=use_elora,
    ).make_fully_connected_tensor_product(*args, **kwargs)


def SymmetricContraction(
    *args,
    use_cueq: bool = True,
    use_elora: bool | None = None,
    **kwargs,
):
    return get_backend(
        use_cueq=use_cueq,
        use_elora=use_elora,
    ).make_symmetric_contraction(*args, **kwargs)


def FullyConnectedNet(
    *args,
    use_cueq: bool = True,
    use_elora: bool | None = None,
    **kwargs,
):
    return get_backend(
        use_cueq=use_cueq,
        use_elora=use_elora,
    ).make_fully_connected_net(*args, **kwargs)


def build_convnet_radial_mlp(
    *args,
    use_cueq: bool = True,
    use_elora: bool | None = None,
    **kwargs,
):
    return get_backend(
        use_cueq=use_cueq,
        use_elora=use_elora,
    ).make_convnet_radial_mlp(*args, **kwargs)


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
