from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Optional, Union

import torch

from curator.data import properties
from curator.data.properties import HEAD_PRESETS, HeadConfig
from curator.layer import AtomwiseNN, GlobalRescaleShift

from .conversion import convert_single_to_multi_domain


def _upgrade_legacy_atomwise_module(module: AtomwiseNN) -> None:
    if getattr(module, "heads", None):
        if not hasattr(module, "separate_heads"):
            module.separate_heads = False
        return

    heads = [
        HeadConfig(
            key=properties.energy,
            is_atomwise=True,
            reduction="sum",
            atomwise_key=properties.atomic_energy,
        )
    ]
    module.separate_heads = False
    module.heads = heads
    module.model_outputs = [head.key for head in heads]
    module.per_atom_flags = [bool(head.write_atomwise) for head in heads]
    module.aggregation_modes = [
        (head.reduction if head.reduction is not None else "sum")
        for head in heads
    ]
    module.per_atom_keys = [(head.atomwise_key or (head.key + "_pa")) for head in heads]
    module.split_size = [int(head.dim) for head in heads]


def _upgrade_legacy_rescale_transforms(module: GlobalRescaleShift) -> None:
    heads = list(getattr(module, "heads", []) or [])
    transform_groups = [
        getattr(module, "scales", []),
        getattr(module, "shifts", []),
        getattr(module, "atomic_scales", []),
        getattr(module, "atomic_shifts", []),
    ]
    for transforms in transform_groups:
        for idx, transform in enumerate(transforms):
            if hasattr(transform, "data_key"):
                continue
            if idx < len(heads):
                head = heads[idx]
                data_key = head.atomwise_key if head.is_atomwise and head.atomwise_key else head.key
                transform.data_key = data_key
            elif hasattr(transform, "key"):
                transform.data_key = transform.key
            if not hasattr(transform, "atomwise_data_key"):
                transform.atomwise_data_key = getattr(transform, "data_key", None) != getattr(transform, "key", None)


def _upgrade_legacy_rescale_module(module: GlobalRescaleShift) -> None:
    if all(hasattr(module, name) for name in ("heads", "scales", "shifts", "atomic_scales", "atomic_shifts")):
        _upgrade_legacy_rescale_transforms(module)
        return

    raw_state = vars(module)

    def find_reference_tensor(value: Any) -> Optional[torch.Tensor]:
        if torch.is_tensor(value):
            return value
        if isinstance(value, torch.nn.Parameter):
            return value.data
        if isinstance(value, (list, tuple)):
            for item in value:
                ref = find_reference_tensor(item)
                if ref is not None:
                    return ref
        return None

    def as_scalar(value: Any, default: float) -> float:
        if value is None:
            return default
        if torch.is_tensor(value):
            flat = value.detach().cpu().reshape(-1)
            return float(flat[0].item()) if flat.numel() > 0 else default
        if isinstance(value, list):
            return float(value[0]) if value else default
        return float(value)

    reference_tensor = None
    for legacy_name in ("scale_by", "shift_by", "per_species_shifts"):
        reference_tensor = find_reference_tensor(raw_state.get(legacy_name))
        if reference_tensor is not None:
            break

    output_keys = list(raw_state.get("output_keys", []))
    scale_keys = list(raw_state.get("scale_keys", []))
    shift_keys = list(raw_state.get("shift_keys", []))
    if not output_keys:
        output_keys = list(dict.fromkeys(scale_keys + shift_keys)) or [properties.energy]

    atomwise_shift = bool(raw_state.get("atomwise_shift", False))
    atomwise_normalization = bool(raw_state.get("atomwise_normalization", False))
    per_species_shift = None
    per_species_all = raw_state.get("per_species_shifts")
    if isinstance(per_species_all, dict):
        per_species_shift = per_species_all.get(properties.energy) or per_species_all.get("energy")

    scale_by = as_scalar(raw_state.get("scale_by"), 1.0)
    shift_by = as_scalar(raw_state.get("shift_by"), 0.0)

    heads: List[HeadConfig] = []
    for key in output_keys:
        if key in HEAD_PRESETS:
            base = HEAD_PRESETS[key]
            heads.append(
                HeadConfig(
                    key=base.key,
                    dim=base.dim,
                    is_atomwise=base.is_atomwise,
                    reduction=base.reduction,
                    atomwise_key=base.atomwise_key,
                    write_atomwise=base.write_atomwise,
                    scale_by=scale_by if key in scale_keys else None,
                    shift_by=shift_by if key in shift_keys else None,
                    atomwise_shift=atomwise_shift if key in shift_keys else False,
                    atomwise_normalization=atomwise_normalization if key in shift_keys else False,
                    per_species_shift=(
                        per_species_shift
                        if key in {properties.energy, properties.atomic_energy, "energy", "atomic_energy"}
                        else None
                    ),
                )
            )
        else:
            heads.append(
                HeadConfig(
                    key=key,
                    dim=1,
                    is_atomwise=False,
                    reduction=None,
                    scale_by=scale_by if key in scale_keys else None,
                    shift_by=shift_by if key in shift_keys else None,
                )
            )

    module.heads = heads
    module._initialize_transforms(scale_trainable=False, shift_trainable=False)
    for legacy_name in (
        "output_keys",
        "scale_keys",
        "shift_keys",
        "atomwise_shift",
        "atomwise_normalization",
        "scale_by",
        "shift_by",
        "per_species_shifts",
    ):
        raw_state.pop(legacy_name, None)
    if reference_tensor is not None:
        to_kwargs = {"device": reference_tensor.device}
        if torch.is_floating_point(reference_tensor) or torch.is_complex(reference_tensor):
            to_kwargs["dtype"] = reference_tensor.dtype
        module.to(**to_kwargs)
    _upgrade_legacy_rescale_transforms(module)


def _upgrade_legacy_checkpoint_model(model: torch.nn.Module) -> torch.nn.Module:
    for module in model.modules():
        if module.__class__.__name__ == "Painn":
            if not hasattr(module, "cutoff_fn"):
                module.cutoff_fn = None
            if not hasattr(module, "radial_basis"):
                module.radial_basis = None
        if module.__class__.__name__ == "PainnUpdate":
            # Legacy PaiNN checkpoints used affine maps on Cartesian vector
            # channels.  Their learned fixed-vector biases break rotations and
            # must not feed an architecture-independent direct-force head.
            for name in ("update_U", "update_V"):
                linear = getattr(module, name, None)
                if linear is not None and getattr(linear, "bias", None) is not None:
                    linear.register_parameter("bias", None)
        if module.__class__.__name__ == "GradientOutput":
            module.produces_forces = (
                properties.forces in getattr(module, "model_outputs", [])
                and not getattr(module, "compute_edge_forces_only", False)
            )
        if isinstance(module, AtomwiseNN):
            _upgrade_legacy_atomwise_module(module)
        if isinstance(module, GlobalRescaleShift):
            _upgrade_legacy_rescale_module(module)
    return model


def _register_legacy_outputspec() -> None:
    try:
        import curator.layer._atomwise_nn as atomwise
    except Exception:
        return
    if hasattr(atomwise, "OutputSpec"):
        return

    class OutputSpec:
        def __init__(
            self,
            key: str,
            dim: int = 1,
            is_atomwise: bool = False,
            reduction: Optional[str] = "sum",
            atomwise_key: Optional[str] = None,
            write_atomwise: bool = False,
        ) -> None:
            self.key = key
            self.dim = dim
            self.is_atomwise = is_atomwise
            self.reduction = reduction
            self.atomwise_key = atomwise_key
            self.write_atomwise = write_atomwise

    atomwise.OutputSpec = OutputSpec


def upgrade_checkpoint(
    ckpt_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Path:
    import curator.model.compat

    ckpt_path = Path(ckpt_path)
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if output_path is None:
        output_path = ckpt_path.with_name(f"{ckpt_path.stem}_converted{ckpt_path.suffix}")
    output_path = Path(output_path)

    _register_legacy_outputspec()
    try:
        obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(ckpt_path, map_location=device)

    if isinstance(obj, torch.nn.Module):
        upgraded_model = convert_single_to_multi_domain(_upgrade_legacy_checkpoint_model(obj))
        torch.save(upgraded_model, output_path)
        return output_path

    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    if "model" not in obj:
        raise KeyError("Checkpoint is missing 'model' entry to upgrade.")

    upgraded_model = convert_single_to_multi_domain(_upgrade_legacy_checkpoint_model(obj["model"]))
    obj["model"] = upgraded_model
    if "state_dict" in obj:
        state_dict = upgraded_model.state_dict()
        obj["state_dict"] = OrderedDict((f"model.{key}", value) for key, value in state_dict.items())
    torch.save(obj, output_path)
    return output_path


__all__ = ["upgrade_checkpoint"]
