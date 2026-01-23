#!/usr/bin/env python3
"""Transfer element-specific weights from a small-element MACE model to a larger one.

This script copies all matching-shape parameters and, for tensors whose only
shape difference is the element dimension, copies rows/cols for elements
shared between source and target (matched by atomic number).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch

from curator.utils import load_model

try:
    from ase.data import atomic_numbers
except Exception:  # pragma: no cover - ase is an optional dependency
    atomic_numbers = None


def _get_index_to_z(model: torch.nn.Module) -> List[int]:
    rep = model.representation if hasattr(model, "representation") else model
    onehot = None
    if hasattr(rep, "embeddings") and "onehot_embedding" in rep.embeddings:
        onehot = rep.embeddings["onehot_embedding"]
    if onehot is not None:
        tm = getattr(onehot, "type_mapper", None)
        if tm is not None and hasattr(tm, "index_to_Z"):
            return tm.index_to_Z.detach().cpu().tolist()
        if getattr(onehot, "species", None) is not None and atomic_numbers is not None:
            return [atomic_numbers[s] for s in onehot.species]
        if getattr(onehot, "num_elements", None) is not None:
            return list(range(1, int(onehot.num_elements) + 1))
    if getattr(rep, "atomic_numbers", None) is not None:
        return rep.atomic_numbers.detach().cpu().tolist()
    if getattr(rep, "species", None) is not None and atomic_numbers is not None:
        return [atomic_numbers[s] for s in rep.species]
    raise ValueError("Unable to determine element ordering for model")


def _build_index_map(src_z: List[int], tgt_z: List[int]) -> Tuple[List[int], List[int], List[int]]:
    src_idx = {z: i for i, z in enumerate(src_z)}
    tgt_idx = {z: i for i, z in enumerate(tgt_z)}
    common = sorted(set(src_idx) & set(tgt_idx))
    src_indices = [src_idx[z] for z in common]
    tgt_indices = [tgt_idx[z] for z in common]
    return common, src_indices, tgt_indices


def _copy_with_element_axis(
    src: torch.Tensor,
    tgt: torch.Tensor,
    axis: int,
    src_indices: torch.Tensor,
    tgt_indices: torch.Tensor,
) -> torch.Tensor:
    src_sel = torch.index_select(src, axis, src_indices)
    new_tgt = tgt.clone()
    new_tgt.index_copy_(axis, tgt_indices, src_sel)
    return new_tgt


def transfer_weights(
    src_model: torch.nn.Module,
    tgt_model: torch.nn.Module,
) -> Tuple[List[int], List[str], List[str], List[str]]:
    src_state = src_model.representation.state_dict()
    tgt_state = tgt_model.representation.state_dict()

    src_z = _get_index_to_z(src_model)
    tgt_z = _get_index_to_z(tgt_model)
    common, src_idx_list, tgt_idx_list = _build_index_map(src_z, tgt_z)

    if not common:
        raise ValueError("No overlapping elements between source and target")

    src_indices = torch.tensor(src_idx_list, dtype=torch.long)
    tgt_indices = torch.tensor(tgt_idx_list, dtype=torch.long)

    src_num = len(src_z)
    tgt_num = len(tgt_z)

    copied_full: List[str] = []
    copied_partial: List[str] = []
    skipped: List[str] = []

    for key, tgt_val in tgt_state.items():
        src_val = src_state.get(key)
        if src_val is None:
            skipped.append(key)
            continue

        if src_val.shape == tgt_val.shape:
            tgt_state[key] = src_val.detach().clone()
            copied_full.append(key)
            continue

        if src_val.dim() != tgt_val.dim():
            skipped.append(key)
            continue

        diff_axes = [i for i, (s, t) in enumerate(zip(src_val.shape, tgt_val.shape)) if s != t]
        if len(diff_axes) != 1:
            skipped.append(key)
            continue

        axis = diff_axes[0]
        if src_val.shape[axis] != src_num or tgt_val.shape[axis] != tgt_num:
            skipped.append(key)
            continue

        tgt_state[key] = _copy_with_element_axis(
            src_val, tgt_val, axis, src_indices, tgt_indices
        )
        copied_partial.append(key)

    tgt_model.representation.load_state_dict(tgt_state, strict=False)
    return common, copied_full, copied_partial, skipped


def _detect_checkpoint(path: Path) -> Tuple[bool, dict | None]:
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception:
        return False, None
    if isinstance(obj, dict) and "state_dict" in obj:
        return True, obj
    return False, None


def _save_with_checkpoint(target_path: Path, output_path: Path, model: torch.nn.Module) -> None:
    is_ckpt, ckpt = _detect_checkpoint(target_path)
    if not is_ckpt or ckpt is None:
        torch.save(model, output_path)
        return

    state_dict = model.state_dict()
    ckpt_state = ckpt.get("state_dict", {})
    needs_prefix = any(k.startswith("model.") for k in ckpt_state.keys())
    if needs_prefix:
        ckpt["state_dict"] = {f"model.{k}": v for k, v in state_dict.items()}
    else:
        ckpt["state_dict"] = state_dict
    torch.save(ckpt, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Path to 6-element MACE model")
    parser.add_argument("--target", required=True, help="Path to 89-element MACE model")
    parser.add_argument("--output", required=True, help="Where to save the updated model")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be copied")
    args = parser.parse_args()

    device = torch.device(args.device)
    src = load_model(args.source, device=device, load_compiled=False, load_weights_only=False)
    tgt = load_model(args.target, device=device, load_compiled=False, load_weights_only=False)

    common, copied_full, copied_partial, skipped = transfer_weights(src, tgt)

    print(f"Common elements: {len(common)}")
    if len(common) <= 30:
        print(f"Common Z: {common}")
    print(f"Copied full params: {len(copied_full)}")
    print(f"Copied element-mapped params: {len(copied_partial)}")
    print(f"Skipped params: {len(skipped)}")

    if args.dry_run:
        return

    _save_with_checkpoint(Path(args.target), Path(args.output), tgt)
    print(f"Saved updated model to: {args.output}")


if __name__ == "__main__":
    main()
