from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from . import properties
from .atoms_data import AtomsData, atoms_data_from_dict, is_atomwise_target


def cat_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    if tensors[0].shape:
        return torch.cat(tensors)
    return torch.stack(tensors)


def _missing_atoms_value(key: str, atoms: Dict[str, torch.Tensor]) -> torch.Tensor:
    if key == properties.pbc:
        return torch.zeros((1, 3), dtype=torch.bool)
    if key == properties.cell:
        dtype = atoms[properties.positions].dtype
        device = atoms[properties.positions].device
        return torch.zeros((3, 3), dtype=dtype, device=device)
    return torch.tensor([])


def _collate_tensor_property(key: str, tensors: List[torch.Tensor]) -> torch.Tensor:
    if key.endswith(properties.energy_hessian_projected) or key.endswith(
        properties.energy_hessian_probe_vectors
    ):
        num_probes = tensors[0].shape[0]
        if any(value.dim() != 3 or value.shape[0] != num_probes for value in tensors):
            raise ValueError(
                f"Cannot collate '{key}': expected [num_probes, n_atoms, 3] "
                "with the same num_probes for every structure."
            )
        return torch.cat(tensors, dim=1)

    if key.endswith(properties.energy_hessian):
        total_atoms = sum(value.shape[0] for value in tensors)
        collated = tensors[0].new_zeros((total_atoms, 3, total_atoms, 3))
        offset = 0
        for value in tensors:
            num_atoms = value.shape[0]
            expected = (num_atoms, 3, num_atoms, 3)
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"Cannot collate '{key}': expected per-structure Hessian "
                    f"shape {expected}, got {tuple(value.shape)}."
                )
            collated[offset : offset + num_atoms, :, offset : offset + num_atoms, :] = value
            offset += num_atoms
        return collated

    return cat_tensors(tensors)


def _collate_atoms_dicts(atoms_list: List[Dict[str, torch.Tensor]], pin_memory: bool) -> Dict[str, torch.Tensor]:
    keys = []
    for atoms in atoms_list:
        for key in atoms:
            if key not in keys:
                keys.append(key)
    dict_of_lists = {
        key: [atoms[key] if key in atoms else _missing_atoms_value(key, atoms) for atoms in atoms_list]
        for key in keys
    }
    pin = (lambda x: x.pin_memory()) if pin_memory else (lambda x: x)

    collated = {k: pin(_collate_tensor_property(k, v)) for k, v in dict_of_lists.items()}

    image_idx = torch.repeat_interleave(
        torch.arange(len(atoms_list)), collated[properties.n_atoms], dim=0
    )
    collated[properties.image_idx] = image_idx

    if properties.domain in collated:
        domain_graph = collated[properties.domain].to(torch.long)
        if domain_graph.dim() > 1:
            domain_graph = domain_graph.view(-1)
        collated[properties.domain] = domain_graph

    # add offset to edge_idx
    if properties.edge_idx in collated:
        edge_offset = torch.zeros_like(collated[properties.n_atoms])
        edge_offset[1:] = collated[properties.n_atoms][:-1]
        edge_offset = torch.cumsum(edge_offset, dim=0)
        edge_offset = torch.repeat_interleave(edge_offset, collated[properties.n_pairs])
        edge_idx = collated[properties.edge_idx] + edge_offset.unsqueeze(-1)
        collated[properties.edge_idx] = edge_idx

    return collated


def collate_atoms_data(samples: List[Any], pin_memory: bool = False) -> Dict[str, Any]:
    atoms_data: List[AtomsData] = []
    for sample in samples:
        if isinstance(sample, AtomsData):
            atoms_data.append(sample)
        elif isinstance(sample, dict):
            atoms_data.append(
                atoms_data_from_dict(
                    sample,
                    task=sample.get("task", "default"),
                    weight=float(sample.get("weight", 1.0)),
                    meta=sample.get("meta"),
                )
            )
        else:
            raise TypeError(f"Unsupported sample type: {type(sample)}")

    atoms_list = [s.atoms for s in atoms_data]
    targets_list = [s.normalized_targets() for s in atoms_data]
    tasks = [s.task for s in atoms_data]
    weights = torch.tensor([float(s.weight) for s in atoms_data], dtype=torch.float)
    metas = [s.meta for s in atoms_data]

    collated = _collate_atoms_dicts(atoms_list, pin_memory=pin_memory)

    target_keys = set()
    for t in targets_list:
        target_keys.update(t.keys())

    targets_batch: Dict[str, torch.Tensor] = {}
    masks: Dict[str, torch.Tensor] = {}

    for key in sorted(target_keys):
        ref = next((t.get(key) for t in targets_list if t.get(key) is not None), None)
        if ref is None:
            continue

        is_atomwise = is_atomwise_target(key)
        values: List[torch.Tensor] = []
        mask_vals: List[bool] = []
        for i, t in enumerate(targets_list):
            val = t.get(key)
            if val is None:
                if is_atomwise:
                    n_atoms = int(atoms_list[i][properties.n_atoms].item())
                    shape = (n_atoms,) + tuple(ref.shape[1:])
                else:
                    shape = tuple(ref.shape)
                val = torch.zeros(shape, dtype=ref.dtype)
                mask_vals.append(False)
            else:
                mask_vals.append(True)
            values.append(val)

        targets_batch[key] = _collate_tensor_property(key, values)
        masks[key] = torch.tensor(mask_vals, dtype=torch.bool)
        collated[key] = targets_batch[key]

    collated["targets"] = targets_batch
    collated["masks"] = masks
    collated["task"] = tasks
    collated["weight"] = weights
    collated["meta"] = metas
    return collated
