"""Exact parameterization of the selected atomwise readout derivatives.

The returned factors use input-major outer-product coordinates.  For a torch
Linear this is a fixed permutation of ``weight.flatten(), bias``; Euclidean
distances and an iid Gaussian projection are unchanged by that permutation.
No empirical layer normalization is performed here.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn


def readout_parameter_layout(module: nn.Module) -> Dict[str, Any]:
    """Describe supported native parameters without rejecting legacy features.

General equivariant layers need their own parameter Jacobian contraction.  An
unsupported description is intentionally harmless until a parameter-gradient
mapping requests it.
    """
    layout: Dict[str, Any] = {
        "supported": False,
        "kind": "unsupported",
        "module_type": type(module).__name__,
        "reason": "Only torch Linear and single-path scalar e3nn Linear readouts are supported.",
    }
    if isinstance(module, nn.Linear):
        layout.update(
            supported=True,
            kind="torch_linear",
            input_width=int(module.in_features),
            output_width=int(module.out_features),
            has_bias=module.bias is not None,
            weight_scale=1.0,
            parameter_count=int(module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)),
            coordinate_order="input-major weight outer product, then bias coordinates",
        )
        layout.pop("reason")
        return layout
    try:
        from e3nn.o3 import Linear
    except ImportError:
        return layout
    if not isinstance(module, Linear):
        return layout
    instructions = module.instructions
    scalar_input = len(module.irreps_in) == 1 and module.irreps_in[0].ir.l == 0
    scalar_output = len(module.irreps_out) == 1 and module.irreps_out[0].ir.l == 0
    if not scalar_input or not scalar_output or len(instructions) != 1:
        layout["reason"] = "e3nn readout requires exactly one dense scalar weight path."
        return layout
    width_in, width_out = int(module.irreps_in.dim), int(module.irreps_out.dim)
    instruction = instructions[0]
    bias = getattr(module, "bias", None)
    if (instruction.i_in != 0 or instruction.i_out != 0
            or tuple(instruction.path_shape) != (width_in, width_out)
            or module.weight.numel() != width_in * width_out
            or (bias is not None and bias.numel() != 0)
            or not getattr(module, "internal_weights", False)
            or not getattr(module, "shared_weights", False)):
        layout["reason"] = "e3nn readout requires internal shared scalar weights with no bias."
        return layout
    layout.update(
        supported=True,
        kind="e3nn_scalar_linear",
        input_width=width_in,
        output_width=width_out,
        has_bias=False,
        weight_scale=float(instruction.path_weight),
        parameter_count=int(module.weight.numel()),
        coordinate_order="input-major weight outer product",
    )
    layout.pop("reason")
    return layout


def parameter_gradient_factors(
    feats: List[torch.Tensor],
    grads: List[torch.Tensor],
    layouts: Optional[List[Dict[str, Any]]],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Convert captured ``[input, 1]`` and energy adjoints to true derivatives.

Removing nonexistent biases and applying e3nn's fixed path coefficient are
chain-rule corrections, not feature rescaling or layer normalization.  The
output must still be projected jointly before any nonlinear transform.
    """
    if layouts is None:
        raise ValueError("Joint readout-gradient mapping requires readout_layouts from FeatureExtractor.")
    if not feats or len(feats) != len(grads) or len(feats) != len(layouts):
        raise ValueError("Readout feature, gradient, and layout lists must have the same nonzero length.")
    pairs = []
    atom_count, device, dtype = None, None, None
    for index, (feat, grad, layout) in enumerate(zip(feats, grads, layouts)):
        if not layout.get("supported", False):
            raise ValueError(f"Unsupported readout parameter layout at layer {index}: {layout.get('reason', 'missing supported layout')}")
        if feat.ndim != 2 or grad.ndim != 2:
            raise ValueError(f"Readout layer {index} requires atom-by-channel feature and gradient matrices.")
        input_width = int(layout["input_width"])
        output_width = int(layout["output_width"])
        if feat.shape[1] != input_width + 1 or grad.shape[1] != output_width:
            raise ValueError(f"Readout layer {index} widths disagree with its parameter layout.")
        if feat.shape[0] != grad.shape[0] or (atom_count is not None and feat.shape[0] != atom_count):
            raise ValueError(f"Readout layer {index} has a different atom population.")
        if (not feat.is_floating_point() or not grad.is_floating_point()
                or feat.device != grad.device or feat.dtype != grad.dtype
                or (device is not None and (feat.device != device or feat.dtype != dtype))):
            raise ValueError("All readout factors must share a floating-point dtype and device.")
        scale = float(layout["weight_scale"])
        if not math.isfinite(scale):
            raise ValueError(f"Readout layer {index} has a nonfinite weight scale.")
        has_bias = bool(layout["has_bias"])
        expected_size = (input_width + int(has_bias)) * output_width
        if int(layout["parameter_count"]) != expected_size:
            raise ValueError(f"Readout layer {index} parameter count disagrees with its factors.")
        if has_bias and scale != 1.0:
            raise ValueError("Scaled readout weights with bias require an explicit bias parameterization.")
        if not torch.equal(feat[:, -1], torch.ones_like(feat[:, -1])):
            raise ValueError(f"Readout layer {index} is missing the captured constant bias coordinate.")
        corrected = feat if has_bias else feat[:, :-1]
        if scale != 1.0:
            corrected = corrected * scale
        pairs.append((corrected, grad))
        atom_count, device, dtype = feat.shape[0], feat.device, feat.dtype
    return pairs
