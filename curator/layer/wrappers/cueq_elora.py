from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from cuequivariance.group_theory.descriptors.irreps_tp import (
    channelwise_tensor_product as cue_channelwise_tensor_product,
    fully_connected_tensor_product as cue_fully_connected_tensor_product,
    linear as cue_linear_descriptor,
)
from torch import nn

from e3nn import o3

from .cueq import (
    CuetSymmetricContraction,
    _apply_dtype_defaults,
    cue,
    cuet,
    make_tensor_product as make_cueq_tensor_product,
)
from .._symmetric_contraction import SymmetricContraction as E3NNSymmetricContraction
from .utils import clamp_rank, freeze_module_parameters, make_lora_matrix_pair


@dataclass(frozen=True)
class _CueqPathSpec:
    key: tuple[int, ...]
    path_shape: tuple[int, ...]
    weight_slice: slice
    left_dim: int
    right_dim: int
    init_scale: float


def _path_dims(path_shape: Iterable[int]) -> tuple[int, int]:
    dims = tuple(int(v) for v in path_shape)
    if len(dims) == 0:
        return (1, 1)
    if len(dims) == 1:
        return (int(dims[0]), 1)
    return (int(math.prod(dims[:-1])), int(dims[-1]))


def _make_path_spec(key: tuple[int, ...], path_shape: Iterable[int], weight_slice: slice) -> _CueqPathSpec:
    shape = tuple(int(v) for v in path_shape)
    left_dim, right_dim = _path_dims(shape)
    return _CueqPathSpec(
        key=tuple(int(v) for v in key),
        path_shape=shape,
        weight_slice=weight_slice,
        left_dim=left_dim,
        right_dim=right_dim,
        init_scale=1.0 / math.sqrt(max(1, left_dim)),
    )


def _make_linear_reference_specs(irreps_in, irreps_out) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    reference = o3.Linear(
        str(irreps_in),
        str(irreps_out),
        internal_weights=True,
        shared_weights=True,
    )
    return [
        ((int(ins.i_in), int(ins.i_out)), tuple(int(v) for v in ins.path_shape))
        for ins in reference.instructions
        if int(ins.i_in) != -1
    ]


def _make_fctp_reference_specs(irreps_in1, irreps_in2, irreps_out) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    reference = o3.FullyConnectedTensorProduct(
        str(irreps_in1),
        str(irreps_in2),
        str(irreps_out),
        internal_weights=True,
        shared_weights=True,
    )
    return [
        (
            (int(ins.i_in1), int(ins.i_in2), int(ins.i_out)),
            tuple(int(v) for v in ins.path_shape),
        )
        for ins in reference.instructions
        if bool(ins.has_weight)
    ]


def _make_channelwise_reference_specs(
    reference_instructions,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    if reference_instructions is None:
        return []
    return [
        (
            (int(ins.i_in1), int(ins.i_in2), int(ins.i_out)),
            tuple(int(v) for v in ins.path_shape),
        )
        for ins in reference_instructions
        if bool(getattr(ins, "has_weight", True))
    ]


def _descriptor_specs_from_linear(irreps_in, irreps_out) -> list[_CueqPathSpec]:
    descriptor = cue_linear_descriptor(irreps_in, irreps_out).polynomial.operations[0][1]
    specs = []
    for path_i, path in enumerate(descriptor.paths):
        specs.append(
            _make_path_spec(
                (int(path.indices[1]), int(path.indices[2])),
                descriptor.get_segment_shape(0, path_i),
                descriptor.segment_slice(0, path_i),
            )
        )
    return specs


def _descriptor_specs_from_fctp(irreps_in1, irreps_in2, irreps_out) -> list[_CueqPathSpec]:
    descriptor = cue_fully_connected_tensor_product(
        irreps_in1,
        irreps_in2,
        irreps_out,
    ).polynomial.operations[0][1]
    specs = []
    for path_i, path in enumerate(descriptor.paths):
        specs.append(
            _make_path_spec(
                (int(path.indices[1]), int(path.indices[2]), int(path.indices[3])),
                descriptor.get_segment_shape(0, path_i),
                descriptor.segment_slice(0, path_i),
            )
        )
    return specs


def _descriptor_specs_from_channelwise(irreps_in1, irreps_in2, filter_irreps_out) -> list[_CueqPathSpec]:
    descriptor = cue_channelwise_tensor_product(
        irreps_in1,
        irreps_in2,
        filter_irreps_out,
    ).polynomial.operations[0][1]
    specs = []
    for path_i, path in enumerate(descriptor.paths):
        specs.append(
            _make_path_spec(
                (int(path.indices[1]), int(path.indices[2]), int(path.indices[3])),
                descriptor.get_segment_shape(0, path_i),
                descriptor.segment_slice(0, path_i),
            )
        )
    return specs


def _reorder_specs(
    descriptor_specs: list[_CueqPathSpec],
    reference_specs: list[tuple[tuple[int, ...], tuple[int, ...]]],
) -> list[_CueqPathSpec]:
    if not reference_specs:
        return descriptor_specs
    descriptor_by_key = {spec.key: spec for spec in descriptor_specs}
    ordered = []
    for key, path_shape in reference_specs:
        spec = descriptor_by_key.get(tuple(int(v) for v in key))
        if spec is None:
            raise KeyError(f"Could not find cueq descriptor path for instruction {key!r}.")
        if tuple(spec.path_shape) != tuple(int(v) for v in path_shape):
            raise ValueError(
                f"Path shape mismatch for instruction {key!r}: "
                f"cueq={spec.path_shape!r} vs reference={tuple(path_shape)!r}."
            )
        ordered.append(spec)
    return ordered


def _make_symmetric_contraction_reference_specs(
    irreps_in,
    irreps_out,
    correlation: int,
    num_elements: int | None,
) -> list[_CueqPathSpec]:
    reference = E3NNSymmetricContraction(
        str(irreps_in),
        str(irreps_out),
        correlation=int(correlation),
        internal_weights=True,
        shared_weights=True,
        num_elements=num_elements,
    )
    specs: list[_CueqPathSpec] = []
    offset = 0
    for out_i, contraction in enumerate(reference.contractions):
        weights = [contraction.weights_max, *list(contraction.weights)]
        for degree, weight in zip(range(contraction.correlation, 0, -1), weights):
            path_shape = tuple(int(v) for v in weight.shape[1:])
            width = int(path_shape[0]) if path_shape else 0
            specs.append(
                _make_path_spec(
                    (int(out_i), int(degree)),
                    path_shape,
                    slice(offset, offset + width),
                )
            )
            offset += width
    return specs


class _CueqPathwiseELoRAModuleMixin:
    lora_rank: int
    lora_alpha: float
    lora_scale: float
    _merged: bool
    _path_specs: list[_CueqPathSpec]
    _path_scales: list[float]

    def _init_pathwise_elora(
        self,
        weight_shape: tuple[int, ...] | None,
        path_specs: list[_CueqPathSpec],
        *,
        rank: int,
        alpha: float,
    ) -> None:
        self.lora_rank = int(rank)
        self.lora_alpha = float(alpha)
        self.lora_scale = self.lora_alpha / float(max(1, self.lora_rank))
        self._merged = False
        self._path_specs = list(path_specs)
        self._path_scales = []
        self.lora_A = nn.ParameterList()
        self.lora_B = nn.ParameterList()
        if weight_shape is None or self.lora_rank <= 0:
            return
        prefix = tuple(int(v) for v in weight_shape[:-1])
        for spec in self._path_specs:
            path_rank = clamp_rank(spec.left_dim, spec.right_dim, self.lora_rank)
            lora_a, lora_b = make_lora_matrix_pair(
                spec.left_dim,
                spec.right_dim,
                rank=path_rank,
                shape_prefix=prefix,
                init_scale=spec.init_scale,
            )
            self.lora_A.append(lora_a)
            self.lora_B.append(lora_b)
            self._path_scales.append(self.lora_alpha / float(path_rank))

    def _has_unmerged_elora(self) -> bool:
        return not self._merged and len(self.lora_A) > 0

    def _apply_pathwise_elora(self, weight: torch.Tensor) -> torch.Tensor:
        if not self._has_unmerged_elora():
            return weight
        delta = torch.zeros_like(weight)
        prefix = tuple(int(v) for v in weight.shape[:-1])
        for spec, path_scale, lora_a, lora_b in zip(
            self._path_specs,
            self._path_scales,
            self.lora_A,
            self.lora_B,
        ):
            update = torch.matmul(lora_a, lora_b).reshape(*prefix, *spec.path_shape)
            delta[..., spec.weight_slice] = update.flatten(start_dim=-len(spec.path_shape)) * path_scale
        return weight + delta

    def _merge_pathwise_elora(self, target: nn.Parameter) -> None:
        if not self._has_unmerged_elora():
            return
        target.data = self._apply_pathwise_elora(target.data)
        self._merged = True


class _BaseCueqELoRAWrapper(_CueqPathwiseELoRAModuleMixin, nn.Module):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = super().__getattr__("base")
            return getattr(base, name)

    def merge_elora(self) -> None:
        self._merge_pathwise_elora(self.base.weight)

    def _init_base_wrapper(
        self,
        base: nn.Module,
        path_specs: list[_CueqPathSpec],
        *,
        rank: int,
        alpha: float,
        freeze_base: bool,
    ) -> None:
        self.base = base
        shape = None
        if self.base.internal_weights and self.base.weight_numel > 0:
            shape = tuple(int(v) for v in self.base.weight.shape)
        self._init_pathwise_elora(shape, path_specs, rank=rank, alpha=alpha)
        if freeze_base:
            freeze_module_parameters(self.base)


class CueqLoRALinear(_BaseCueqELoRAWrapper):
    def __init__(
        self,
        *args,
        rank: int = 16,
        alpha: float = 16.0,
        elora_freeze_base: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        kwargs = _apply_dtype_defaults(kwargs)
        base = cuet.Linear(*args, **kwargs)
        path_specs = _reorder_specs(
            _descriptor_specs_from_linear(base.irreps_in, base.irreps_out),
            _make_linear_reference_specs(base.irreps_in, base.irreps_out),
        )
        self._init_base_wrapper(
            base,
            path_specs,
            rank=rank,
            alpha=alpha,
            freeze_base=elora_freeze_base,
        )

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor | None = None,
        weight_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._has_unmerged_elora() and self.base.internal_weights:
            if weight is not None:
                raise ValueError("Internal weights are used, weight should be None")
            effective_weight = self._apply_pathwise_elora(self.base.weight)
            input_indices = {}
            if self.base.weight_classes > 1:
                if weight_indices is None:
                    raise ValueError("weight_indices should be provided if weight_classes > 1")
                input_indices[0] = weight_indices
            output = self.base.f(
                [effective_weight, self.base.transpose_in(x)],
                input_indices=input_indices,
            )
            return self.base.transpose_out(output[0])
        return self.base(x, weight=weight, weight_indices=weight_indices)


class CueqLoRATensorProduct(_BaseCueqELoRAWrapper):
    def _forward_with_internal_weight(
        self,
        *inputs,
        effective_weight: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        x1 = self.base.transpose_in1(inputs[0])
        x2 = self.base.transpose_in2(inputs[1])
        if "indices_1" in kwargs or "indices_2" in kwargs or "indices_out" in kwargs:
            indices_in = {}
            indices_1 = kwargs.get("indices_1")
            indices_2 = kwargs.get("indices_2")
            indices_out = kwargs.get("indices_out")
            size_out = kwargs.get("size_out")
            if indices_1 is not None:
                indices_in[1] = indices_1
            if indices_2 is not None:
                indices_in[2] = indices_2
            output_indices = None
            output_shapes = {}
            if indices_out is not None:
                if size_out is None:
                    raise ValueError("size_out should be provided if indices_out is provided")
                output_indices = {0: indices_out}
                output_shapes = {0: torch.empty(size_out, 1, device=x1.device)}
            output = self.base.f(
                [effective_weight, x1, x2],
                input_indices=indices_in,
                output_shapes=output_shapes,
                output_indices=output_indices,
            )
            return self.base.transpose_out(output[0])
        output = self.base.f([effective_weight, x1, x2])
        return self.base.transpose_out(output[0])

    def forward(self, *inputs, weight: torch.Tensor | None = None, **kwargs):
        if self._has_unmerged_elora() and getattr(self.base, "weight", None) is not None:
            if weight is not None:
                raise ValueError("Internal weights are used, weight should be None")
            effective_weight = self._apply_pathwise_elora(self.base.weight)
            return self._forward_with_internal_weight(
                *inputs,
                effective_weight=effective_weight,
                **kwargs,
            )
        return self.base(*inputs, weight=weight, **kwargs)


class CueqLoRAFullyConnectedTensorProduct(CueqLoRATensorProduct):
    def __init__(
        self,
        *args,
        rank: int = 16,
        alpha: float = 16.0,
        elora_freeze_base: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        kwargs = _apply_dtype_defaults(kwargs)
        base = cuet.FullyConnectedTensorProduct(*args, **kwargs)
        path_specs = _reorder_specs(
            _descriptor_specs_from_fctp(base.irreps_in1, base.irreps_in2, base.irreps_out),
            _make_fctp_reference_specs(base.irreps_in1, base.irreps_in2, base.irreps_out),
        )
        self._init_base_wrapper(
            base,
            path_specs,
            rank=rank,
            alpha=alpha,
            freeze_base=elora_freeze_base,
        )


class CueqLoRAChannelWiseTensorProduct(CueqLoRATensorProduct):
    def __init__(
        self,
        *args,
        rank: int = 16,
        alpha: float = 16.0,
        elora_freeze_base: bool = True,
        reference_instructions=None,
        **kwargs,
    ) -> None:
        super().__init__()
        base = make_cueq_tensor_product(*args, **kwargs)
        filter_irreps_out = [mul_ir.ir for mul_ir in base.irreps_out]
        path_specs = _reorder_specs(
            _descriptor_specs_from_channelwise(base.irreps_in1, base.irreps_in2, filter_irreps_out),
            _make_channelwise_reference_specs(reference_instructions),
        )
        self._init_base_wrapper(
            base,
            path_specs,
            rank=rank,
            alpha=alpha,
            freeze_base=elora_freeze_base,
        )


class CueqLoRASymmetricContraction(_CueqPathwiseELoRAModuleMixin, CuetSymmetricContraction):
    def __init__(
        self,
        *args,
        rank: int = 16,
        alpha: float = 16.0,
        elora_freeze_base: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        path_specs = _make_symmetric_contraction_reference_specs(
            self.sc.irreps_in,
            self.sc.irreps_out,
            self.contraction_degree,
            self.num_elements,
        )
        total_width = sum(spec.left_dim for spec in path_specs)
        if total_width != int(self.sc.weight.shape[1]):
            raise ValueError(
                "Symmetric-contraction path specs do not match visible weight width: "
                f"specs={total_width}, weight={int(self.sc.weight.shape[1])}."
            )
        self.lora_rank = int(rank)
        self.lora_alpha = float(alpha)
        self.lora_scale = self.lora_alpha / float(max(1, self.lora_rank))
        self._merged = False
        self._path_specs = path_specs
        self._path_scales = []
        self.lora_A = nn.ParameterList()
        self.lora_B = nn.ParameterList()
        num_elements = int(self.sc.weight.shape[0])
        if self.lora_rank > 0:
            for spec in self._path_specs:
                path_rank = clamp_rank(spec.left_dim, spec.right_dim, self.lora_rank)
                lora_a, lora_b = make_lora_matrix_pair(
                    spec.left_dim,
                    spec.right_dim,
                    rank=path_rank,
                    shape_prefix=(num_elements,),
                    init_scale=spec.init_scale,
                )
                self.lora_A.append(lora_a)
                self.lora_B.append(lora_b)
                self._path_scales.append(self.lora_alpha / float(path_rank))
        if elora_freeze_base:
            freeze_module_parameters(self.sc)

    def _apply_visible_pathwise_elora(self, weight: torch.Tensor) -> torch.Tensor:
        if not self._has_unmerged_elora():
            return weight
        delta = torch.zeros_like(weight)
        num_elements = int(weight.shape[0])
        for spec, path_scale, lora_a, lora_b in zip(
            self._path_specs,
            self._path_scales,
            self.lora_A,
            self.lora_B,
        ):
            update = torch.matmul(lora_a, lora_b).reshape(num_elements, *spec.path_shape)
            delta[:, spec.weight_slice, :] = update * path_scale
        return weight + delta

    def forward(self, x: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_weight(x, attrs, self._apply_visible_pathwise_elora(self.sc.weight))

    def merge_elora(self) -> None:
        if not self._has_unmerged_elora():
            return
        self.sc.weight.data = self._apply_visible_pathwise_elora(self.sc.weight.data)
        self._merged = True
