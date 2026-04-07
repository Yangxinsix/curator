from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Iterable, Optional, Sequence

import torch

from .utils import ensure_torch_serialization_compat, freeze_module_parameters

ensure_torch_serialization_compat()

from e3nn import o3


_MIN_TORCH_VERSION = (2, 10)
_HAS_OEQ_MODULE = find_spec("openequivariance") is not None


def _parse_torch_version() -> tuple[int, int]:
    version = str(torch.__version__).split("+", 1)[0]
    parts = version.split(".")
    major = int(parts[0]) if parts else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    return (major, minor)


def oeq_runtime_error() -> Optional[str]:
    if not _HAS_OEQ_MODULE:
        return "backend=oeq requires the `openequivariance` package to be installed."
    if not torch.cuda.is_available():
        return "backend=oeq requires CUDA; no CUDA device is currently available."
    if _parse_torch_version() < _MIN_TORCH_VERSION:
        return (
            f"backend=oeq requires PyTorch >= {_MIN_TORCH_VERSION[0]}.{_MIN_TORCH_VERSION[1]}; "
            f"found torch=={torch.__version__}."
        )
    return None


def is_oeq_runtime_usable() -> bool:
    return oeq_runtime_error() is None


def ensure_oeq_runtime_available() -> None:
    error = oeq_runtime_error()
    if error is not None:
        raise RuntimeError(error)


def _import_oeq():
    ensure_oeq_runtime_available()
    import openequivariance as oeq

    return oeq


def _import_oeq_symmetric_contraction():
    ensure_oeq_runtime_available()
    from openequivariance._torch.symmetric_contraction.symmetric_contraction import (
        SymmetricContraction as OeqNativeSymmetricContraction,
    )

    return OeqNativeSymmetricContraction


def _normalize_shared_internal_weights(
    *,
    internal_weights: Optional[bool],
    shared_weights: Optional[bool],
) -> tuple[bool, bool]:
    if shared_weights is False and internal_weights is None:
        internal_weights = False
    if shared_weights is None:
        shared_weights = True
    if internal_weights is None:
        internal_weights = True
    if internal_weights and not shared_weights:
        raise ValueError("internal_weights=True requires shared_weights=True.")
    return bool(internal_weights), bool(shared_weights)


def _normalize_weight_dtype(dtype: torch.dtype) -> type[Any]:
    if dtype == torch.float32:
        import numpy as np

        return np.float32
    if dtype == torch.float64:
        import numpy as np

        return np.float64
    raise TypeError(
        f"OpenEquivariance wrappers only support float32/float64, got dtype={dtype}."
    )


def _promote_instruction(ins: Any) -> tuple[int, int, int, str, bool, float]:
    if hasattr(ins, "i_in1") and hasattr(ins, "i_in2") and hasattr(ins, "i_out"):
        return (
            int(ins.i_in1),
            int(ins.i_in2),
            int(ins.i_out),
            str(ins.connection_mode),
            bool(ins.has_weight),
            float(ins.path_weight),
        )
    raw = tuple(ins)
    if len(raw) == 5:
        i_in1, i_in2, i_out, connection_mode, has_weight = raw
        path_weight = 1.0
    elif len(raw) == 6:
        i_in1, i_in2, i_out, connection_mode, has_weight, path_weight = raw
    else:
        raise ValueError(f"Unsupported tensor product instruction: {ins!r}")
    return (
        int(i_in1),
        int(i_in2),
        int(i_out),
        str(connection_mode),
        bool(has_weight),
        float(path_weight),
    )


def _validate_oeq_instructions(
    instructions: Sequence[tuple[int, int, int, str, bool, float]],
) -> None:
    weighted = [ins for ins in instructions if ins[4]]
    if len(weighted) != len(instructions):
        raise NotImplementedError(
            "backend=oeq only supports fully weighted tensor products; found instruction(s) with has_weight=False."
        )
    modes = {ins[3] for ins in instructions}
    if not modes:
        return
    if not modes.issubset({"uvu", "uvw"}):
        raise NotImplementedError(
            f"backend=oeq only supports 'uvu'/'uvw' tensor product modes, got {sorted(modes)}."
        )
    if len(modes) > 1:
        raise NotImplementedError(
            f"backend=oeq does not support mixed tensor product modes in one operator, got {sorted(modes)}."
        )


def _bias_instruction_slices(
    *,
    irreps_out: o3.Irreps,
    instructions: Iterable[Any],
) -> list[tuple[slice, int, int]]:
    output_slices = irreps_out.slices()
    bias_slices: list[tuple[slice, int, int]] = []
    offset = 0
    for ins in instructions:
        if int(ins.i_in) != -1:
            continue
        path_shape = tuple(int(v) for v in ins.path_shape)
        width = int(path_shape[0]) if path_shape else 0
        out_slice = output_slices[int(ins.i_out)]
        bias_slices.append((out_slice, offset, offset + width))
        offset += width
    return bias_slices


@dataclass(frozen=True)
class _RuntimeSpec:
    irreps_in1: str
    irreps_in2: str
    irreps_out: str
    instructions: tuple[tuple[int, int, int, str, bool, float], ...]
    irrep_normalization: str
    path_normalization: str
    shared_weights: bool


class _OeqTensorProductSupport(torch.nn.Module):
    weight_numel: int
    internal_weights: bool
    shared_weights: bool

    def _init_runtime_spec(self, spec: _RuntimeSpec) -> None:
        self._runtime_spec = spec
        object.__setattr__(self, "_oeq_tp", None)
        object.__setattr__(self, "_oeq_dtype", None)

    def _build_runtime_tp(self, dtype: torch.dtype):
        oeq = _import_oeq()
        weight_dtype = _normalize_weight_dtype(dtype)
        problem = oeq.TPProblem(
            oeq.Irreps(self._runtime_spec.irreps_in1),
            oeq.Irreps(self._runtime_spec.irreps_in2),
            oeq.Irreps(self._runtime_spec.irreps_out),
            list(self._runtime_spec.instructions),
            irrep_normalization=self._runtime_spec.irrep_normalization,
            path_normalization=self._runtime_spec.path_normalization,
            internal_weights=False,
            shared_weights=self._runtime_spec.shared_weights,
            irrep_dtype=weight_dtype,
            weight_dtype=weight_dtype,
            layout="mul_ir",
        )
        tp = oeq.TensorProduct(problem)
        object.__setattr__(self, "_oeq_tp", tp)
        object.__setattr__(self, "_oeq_dtype", dtype)
        return tp

    def _runtime_tp(self, dtype: torch.dtype):
        tp = getattr(self, "_oeq_tp", None)
        current_dtype = getattr(self, "_oeq_dtype", None)
        if tp is None or current_dtype != dtype:
            tp = self._build_runtime_tp(dtype)
        return tp

    def _reorder_to_oeq(
        self,
        weight: torch.Tensor,
        *,
        dtype: torch.dtype,
        shared_weights: bool,
    ) -> torch.Tensor:
        tp = self._runtime_tp(dtype)
        return tp.reorder_weights_from_e3nn(
            weight,
            has_batch_dim=not shared_weights,
        )

    @staticmethod
    def _flatten_input(tensor: torch.Tensor, expected_dim: int) -> tuple[torch.Tensor, torch.Size]:
        if tensor.shape[-1] != expected_dim:
            raise ValueError(
                f"Incorrect trailing dimension: expected {expected_dim}, got {tensor.shape[-1]}."
            )
        batch_shape = tensor.shape[:-1]
        flat = tensor.reshape(-1, expected_dim)
        return flat, batch_shape

    @staticmethod
    def _reshape_output(tensor: torch.Tensor, batch_shape: torch.Size) -> torch.Tensor:
        return tensor.reshape(*batch_shape, tensor.shape[-1])

    def _prepare_weight(
        self,
        weight: Optional[torch.Tensor],
        *,
        batch_shape: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError("Weights must be provided when internal_weights = False")
            weight = self.weight
        if self.weight_numel == 0:
            if self.shared_weights:
                return weight.reshape(self.weight_numel)
            return weight.reshape(*batch_shape, self.weight_numel).reshape(-1, self.weight_numel)
        if self.shared_weights:
            if weight.shape[-1] != self.weight_numel:
                raise ValueError(
                    f"Expected shared weights with trailing size {self.weight_numel}, got {tuple(weight.shape)}."
                )
            if weight.ndim > 1:
                weight = weight.reshape(-1, self.weight_numel)
                if weight.shape[0] != 1:
                    raise ValueError(
                        "Shared-weight tensor product expects a single weight vector when shared_weights=True."
                    )
                weight = weight[0]
            return self._reorder_to_oeq(weight, dtype=dtype, shared_weights=True)
        if weight.shape[-1] != self.weight_numel:
            raise ValueError(
                f"Expected per-sample weights with trailing size {self.weight_numel}, got {tuple(weight.shape)}."
            )
        flat_weight = weight.reshape(-1, self.weight_numel)
        expected_batch = 1
        for dim in batch_shape:
            expected_batch *= int(dim)
        if flat_weight.shape[0] != expected_batch:
            raise ValueError(
                f"Weight batch size {flat_weight.shape[0]} does not match input batch size {expected_batch}."
            )
        return self._reorder_to_oeq(flat_weight, dtype=dtype, shared_weights=False)

    def weight_view_for_instruction(
        self,
        instruction: int,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if weight is None:
            if not self.internal_weights:
                raise RuntimeError("Weights must be provided when internal_weights = False")
            weight = self.weight
        batch_shape = weight.shape[:-1]
        offset = 0
        for ins_i, ins in enumerate(self.instructions):
            shape = tuple(int(v) for v in ins.path_shape)
            width = 1
            for dim in shape:
                width *= dim
            if ins_i == instruction:
                return weight.narrow(-1, offset, width).view(batch_shape + shape)
            offset += width
        raise IndexError(f"Instruction index out of range: {instruction}")

    def weight_views(
        self,
        weight: Optional[torch.Tensor] = None,
        yield_instruction: bool = False,
    ):
        if weight is None:
            if not self.internal_weights:
                raise RuntimeError("Weights must be provided when internal_weights = False")
            weight = self.weight
        offset = 0
        batch_shape = weight.shape[:-1]
        for ins_i, ins in enumerate(self.instructions):
            shape = tuple(int(v) for v in ins.path_shape)
            width = 1
            for dim in shape:
                width *= dim
            current = weight.narrow(-1, offset, width).view(batch_shape + shape)
            offset += width
            if yield_instruction:
                yield ins_i, ins, current
            else:
                yield current


class OeqTensorProduct(_OeqTensorProductSupport):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        in1_var=None,
        in2_var=None,
        out_var=None,
        irrep_normalization: str = None,
        path_normalization: str = None,
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        compile_left_right: bool = True,
        compile_right: bool = False,
        normalization=None,
        _specialized_code: Optional[bool] = None,
        _optimize_einsums: Optional[bool] = None,
    ) -> None:
        super().__init__()
        del compile_left_right, compile_right, normalization, _specialized_code, _optimize_einsums
        internal_weights, shared_weights = _normalize_shared_internal_weights(
            internal_weights=internal_weights,
            shared_weights=shared_weights,
        )
        reference = o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions,
            in1_var=in1_var,
            in2_var=in2_var,
            out_var=out_var,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            internal_weights=False,
            shared_weights=shared_weights,
        )
        runtime_instructions = tuple(_promote_instruction(ins) for ins in reference.instructions)
        _validate_oeq_instructions(runtime_instructions)

        self.irreps_in1 = reference.irreps_in1
        self.irreps_in2 = reference.irreps_in2
        self.irreps_out = reference.irreps_out
        self.instructions = reference.instructions
        self.in1_var = list(in1_var) if in1_var is not None else None
        self.in2_var = list(in2_var) if in2_var is not None else None
        self.out_var = list(out_var) if out_var is not None else None
        self.irrep_normalization = str(irrep_normalization or "component")
        self.path_normalization = str(path_normalization or "element")
        self.internal_weights = internal_weights
        self.shared_weights = shared_weights
        self.weight_numel = int(reference.weight_numel)
        self._in1_dim = int(reference._in1_dim)
        self._in2_dim = int(reference._in2_dim)

        self._init_runtime_spec(
            _RuntimeSpec(
                irreps_in1=str(self.irreps_in1),
                irreps_in2=str(self.irreps_in2),
                irreps_out=str(self.irreps_out),
                instructions=runtime_instructions,
                irrep_normalization=self.irrep_normalization,
                path_normalization=self.path_normalization,
                shared_weights=self.shared_weights,
            )
        )

        if self.internal_weights and self.weight_numel > 0:
            self.weight = torch.nn.Parameter(torch.randn(self.weight_numel))
        else:
            self.register_buffer("weight", torch.empty(0))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.irreps_in1} x {self.irreps_in2} "
            f"-> {self.irreps_out} | {self.weight_numel} weights)"
        )

    def forward(self, x, y, weight: Optional[torch.Tensor] = None):
        flat_x, batch_shape = self._flatten_input(x, self._in1_dim)
        flat_y, other_shape = self._flatten_input(y, self._in2_dim)
        if batch_shape != other_shape:
            raise ValueError(
                f"Mismatched batch shape for tensor product inputs: {tuple(batch_shape)} vs {tuple(other_shape)}."
            )
        runtime_weight = self._prepare_weight(
            weight,
            batch_shape=batch_shape,
            dtype=flat_x.dtype,
        )
        tp = self._runtime_tp(flat_x.dtype)
        out = tp(flat_x, flat_y, runtime_weight)
        return self._reshape_output(out, batch_shape)


class OeqFullyConnectedTensorProduct(OeqTensorProduct):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        irrep_normalization: str = None,
        path_normalization: str = None,
        **kwargs,
    ) -> None:
        shared_weights = kwargs.pop("shared_weights", None)
        internal_weights = kwargs.pop("internal_weights", None)
        reference = o3.FullyConnectedTensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            internal_weights=False,
            shared_weights=True if shared_weights is None else shared_weights,
            **kwargs,
        )
        super().__init__(
            reference.irreps_in1,
            reference.irreps_in2,
            reference.irreps_out,
            [_promote_instruction(ins) for ins in reference.instructions],
            in1_var=None,
            in2_var=None,
            out_var=None,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
        )


class OeqLinear(_OeqTensorProductSupport):
    def __init__(
        self,
        irreps_in,
        irreps_out,
        *,
        f_in: Optional[int] = None,
        f_out: Optional[int] = None,
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        instructions: Optional[list[tuple[int, int]]] = None,
        biases: bool | list[bool] = False,
        path_normalization: str = "element",
        _optimize_einsums: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if f_in is not None or f_out is not None:
            raise NotImplementedError("backend=oeq does not support Linear(f_in=..., f_out=...).")
        del _optimize_einsums
        internal_weights, shared_weights = _normalize_shared_internal_weights(
            internal_weights=internal_weights,
            shared_weights=shared_weights,
        )
        reference = o3.Linear(
            irreps_in,
            irreps_out,
            internal_weights=False,
            shared_weights=shared_weights,
            instructions=instructions,
            biases=biases,
            path_normalization=path_normalization,
        )
        runtime_instructions = tuple(
            (
                int(ins.i_in),
                0,
                int(ins.i_out),
                "uvw",
                True,
                float(ins.path_weight),
            )
            for ins in reference.instructions
            if int(ins.i_in) != -1
        )
        _validate_oeq_instructions(runtime_instructions)

        self.irreps_in = reference.irreps_in
        self.irreps_out = reference.irreps_out
        self.instructions = reference.instructions
        self.internal_weights = internal_weights
        self.shared_weights = shared_weights
        self.weight_numel = int(reference.weight_numel)
        self.bias_numel = int(reference.bias_numel)
        self.path_normalization = str(path_normalization)
        self.register_buffer("output_mask", reference.output_mask.clone())
        self._in_dim = int(self.irreps_in.dim)
        self._out_dim = int(self.irreps_out.dim)
        self._bias_slices = _bias_instruction_slices(
            irreps_out=self.irreps_out,
            instructions=self.instructions,
        )

        self._init_runtime_spec(
            _RuntimeSpec(
                irreps_in1=str(self.irreps_in),
                irreps_in2="1x0e",
                irreps_out=str(self.irreps_out),
                instructions=runtime_instructions,
                irrep_normalization="component",
                path_normalization=self.path_normalization,
                shared_weights=self.shared_weights,
            )
        )

        if self.internal_weights and self.weight_numel > 0:
            self.weight = torch.nn.Parameter(torch.randn(self.weight_numel))
        else:
            self.register_buffer("weight", torch.empty(0))
        if self.internal_weights and self.bias_numel > 0:
            self.bias = torch.nn.Parameter(torch.zeros(self.bias_numel))
        else:
            self.register_buffer("bias", torch.empty(0))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.irreps_in} -> {self.irreps_out} | {self.weight_numel} weights)"

    def _prepare_bias(
        self,
        bias: Optional[torch.Tensor],
        *,
        batch_shape: torch.Size,
    ) -> Optional[torch.Tensor]:
        if self.bias_numel == 0:
            return None
        if bias is None:
            if not self.internal_weights:
                raise RuntimeError("Biases must be provided when internal_weights = False")
            bias = self.bias
        if bias.shape[-1] != self.bias_numel:
            raise ValueError(
                f"Expected bias tensor with trailing size {self.bias_numel}, got {tuple(bias.shape)}."
            )
        if self.shared_weights:
            if bias.ndim > 1:
                bias = bias.reshape(-1, self.bias_numel)
                if bias.shape[0] != 1:
                    raise ValueError("Shared-weight linear expects a single bias vector.")
                bias = bias[0]
            return bias
        flat = bias.reshape(-1, self.bias_numel)
        expected_batch = 1
        for dim in batch_shape:
            expected_batch *= int(dim)
        if flat.shape[0] != expected_batch:
            raise ValueError(
                f"Bias batch size {flat.shape[0]} does not match input batch size {expected_batch}."
            )
        return flat

    def _apply_bias(
        self,
        output: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if bias is None:
            return output
        out = output.clone()
        if bias.ndim == 1:
            for out_slice, start, stop in self._bias_slices:
                out[..., out_slice] = out[..., out_slice] + bias[start:stop]
            return out
        for out_slice, start, stop in self._bias_slices:
            out[..., out_slice] = out[..., out_slice] + bias[..., start:stop]
        return out

    def forward(
        self,
        features: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        flat_x, batch_shape = self._flatten_input(features, self._in_dim)
        ones = torch.ones(
            flat_x.shape[0],
            1,
            dtype=flat_x.dtype,
            device=flat_x.device,
        )
        runtime_weight = self._prepare_weight(
            weight,
            batch_shape=batch_shape,
            dtype=flat_x.dtype,
        )
        runtime_bias = self._prepare_bias(bias, batch_shape=batch_shape)
        tp = self._runtime_tp(flat_x.dtype)
        out = tp(flat_x, ones, runtime_weight)
        out = self._reshape_output(out, batch_shape)
        return self._apply_bias(out, runtime_bias)


def OeqSymmetricContraction(*args, freeze_base: bool = False, **kwargs):
    native_cls = _import_oeq_symmetric_contraction()
    module = native_cls(*args, **kwargs)
    if not hasattr(module, "contraction_degree"):
        try:
            module.contraction_degree = int(kwargs.get("correlation"))
        except Exception:
            pass
    if freeze_base:
        freeze_module_parameters(module)
    return module


def make_linear(*args, **kwargs):
    return OeqLinear(*args, **kwargs)


def make_tensor_product(*args, **kwargs):
    return OeqTensorProduct(*args, **kwargs)


def make_fully_connected_tensor_product(*args, **kwargs):
    return OeqFullyConnectedTensorProduct(*args, **kwargs)


def make_symmetric_contraction(*args, **kwargs):
    return OeqSymmetricContraction(*args, **kwargs)


__all__ = [
    "OeqLinear",
    "OeqTensorProduct",
    "OeqFullyConnectedTensorProduct",
    "OeqSymmetricContraction",
    "ensure_oeq_runtime_available",
    "is_oeq_runtime_usable",
    "make_linear",
    "make_tensor_product",
    "make_fully_connected_tensor_product",
    "make_symmetric_contraction",
]
