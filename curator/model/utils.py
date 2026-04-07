from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
from ase import Atoms
from ase.data import atomic_numbers as ase_atomic_numbers
from torch import nn

from curator.data import properties
from curator.layer.utils import find_layer_by_name_recursive


@dataclass
class BatchStructure:
    numbers: torch.Tensor
    positions: torch.Tensor
    cell: Optional[torch.Tensor]
    pbc: torch.Tensor
    edge_index: Optional[torch.Tensor] = None
    edge_diff: Optional[torch.Tensor] = None
    edge_dist: Optional[torch.Tensor] = None
    cell_displacements: Optional[torch.Tensor] = None

    @property
    def num_atoms(self) -> int:
        return int(self.numbers.shape[0])


def require_batch_fields(data: properties.Type, fields: Iterable[str]) -> None:
    missing = [field for field in fields if field not in data]
    if missing:
        raise KeyError(f"Batch is missing required fields: {missing}")


def normalize_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.dim() != 2:
        raise ValueError(f"Unsupported edge index shape: {tuple(edge_index.shape)}")
    if edge_index.shape[-1] == 2:
        return edge_index.to(torch.long)
    if edge_index.shape[0] == 2:
        return edge_index.transpose(0, 1).contiguous().to(torch.long)
    raise ValueError(f"Unsupported edge index shape: {tuple(edge_index.shape)}")


def extract_cells(data: properties.Type, batch_size: int) -> Optional[torch.Tensor]:
    cell = data.get(properties.cell)
    if cell is None:
        return None
    if not torch.is_tensor(cell) or cell.numel() == 0:
        return None
    if cell.dim() == 3 and cell.shape[0] == batch_size and cell.shape[1:] == (3, 3):
        return cell
    if cell.dim() == 2 and cell.shape == (3, 3) and batch_size == 1:
        return cell.unsqueeze(0)
    if cell.dim() == 2 and cell.shape[1] == 3 and cell.shape[0] == 3 * batch_size:
        return cell.view(batch_size, 3, 3)
    if cell.dim() == 1 and cell.numel() == 9 * batch_size:
        return cell.view(batch_size, 3, 3)
    raise ValueError(f"Unsupported cell shape: {tuple(cell.shape)}")


def extract_pbc(data: properties.Type, batch_size: int) -> torch.Tensor:
    pbc = data.get(properties.pbc)
    if pbc is None:
        return torch.zeros((batch_size, 3), dtype=torch.bool)
    if isinstance(pbc, bool):
        return torch.full((batch_size, 3), bool(pbc), dtype=torch.bool)
    if not torch.is_tensor(pbc):
        pbc = torch.as_tensor(pbc, dtype=torch.bool)
    pbc = pbc.to(torch.bool)
    if pbc.dim() == 1 and pbc.numel() == 3:
        return pbc.view(1, 3).expand(batch_size, 3).clone()
    if pbc.dim() == 2 and pbc.shape == (batch_size, 3):
        return pbc
    raise ValueError(f"Unsupported pbc shape: {tuple(pbc.shape)}")


def build_image_index(n_atoms: torch.Tensor, device: torch.device) -> torch.Tensor:
    counts = n_atoms.view(-1).to(torch.long)
    return torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.long),
        counts,
    )


def build_ptr(n_atoms: torch.Tensor, device: torch.device) -> torch.Tensor:
    counts = n_atoms.view(-1).to(torch.long)
    ptr = torch.zeros((counts.shape[0] + 1,), dtype=torch.long, device=device)
    ptr[1:] = torch.cumsum(counts, dim=0)
    return ptr


def split_batch_structures(data: properties.Type) -> list[BatchStructure]:
    require_batch_fields(
        data,
        (properties.n_atoms, properties.atomic_numbers, properties.positions),
    )
    n_atoms = data[properties.n_atoms].view(-1).to(torch.long)
    positions = data[properties.positions].view(-1, 3)
    numbers = data[properties.atomic_numbers].view(-1).to(torch.long)
    cells = extract_cells(data, len(n_atoms))
    pbcs = extract_pbc(data, len(n_atoms)).to(device=positions.device)

    edge_index = data.get(properties.edge_idx)
    edge_diff = data.get(properties.edge_diff)
    edge_dist = data.get(properties.edge_dist)
    cell_displacements = data.get(properties.cell_displacements)
    if edge_index is not None:
        edge_index = normalize_edge_index(edge_index)

    structures: list[BatchStructure] = []
    atom_offset = 0
    for i, count_tensor in enumerate(n_atoms):
        count = int(count_tensor.item())
        atom_slice = slice(atom_offset, atom_offset + count)
        local_edge_index = None
        local_edge_diff = None
        local_edge_dist = None
        local_cell_displacements = None
        if edge_index is not None:
            mask = (edge_index[:, 0] >= atom_offset) & (edge_index[:, 0] < atom_offset + count)
            local_edge_index = edge_index[mask] - atom_offset
            if edge_diff is not None:
                local_edge_diff = edge_diff[mask]
            if edge_dist is not None:
                local_edge_dist = edge_dist[mask]
            if cell_displacements is not None:
                local_cell_displacements = cell_displacements[mask]
        structures.append(
            BatchStructure(
                numbers=numbers[atom_slice],
                positions=positions[atom_slice],
                cell=None if cells is None else cells[i],
                pbc=pbcs[i],
                edge_index=local_edge_index,
                edge_diff=local_edge_diff,
                edge_dist=local_edge_dist,
                cell_displacements=local_cell_displacements,
            )
        )
        atom_offset += count
    return structures


def structure_to_atoms(struct: BatchStructure) -> Atoms:
    kwargs = {
        "numbers": struct.numbers.detach().cpu().numpy(),
        "positions": struct.positions.detach().cpu().numpy(),
        "pbc": tuple(bool(v) for v in struct.pbc.detach().cpu().tolist()),
    }
    if struct.cell is not None:
        kwargs["cell"] = struct.cell.detach().cpu().numpy()
    return Atoms(**kwargs)


def batch_to_atoms(data: properties.Type) -> list[Atoms]:
    atoms_list: list[Atoms] = []
    for struct in split_batch_structures(data):
        atoms_list.append(structure_to_atoms(struct))
    return atoms_list


def infer_cell_offsets(
    edge_index: torch.Tensor,
    positions: torch.Tensor,
    cell: Optional[torch.Tensor],
    edge_diff: Optional[torch.Tensor] = None,
    cell_displacements: Optional[torch.Tensor] = None,
    atol: float = 1e-4,
) -> torch.Tensor:
    num_edges = int(edge_index.shape[0])
    if cell is None or cell.numel() == 0:
        return positions.new_zeros((num_edges, 3))
    if num_edges == 0:
        return cell.new_zeros((0, 3))
    if cell_displacements is not None:
        shift_cart = cell_displacements
    elif edge_diff is not None:
        center = positions[edge_index[:, 0]]
        neighbor = positions[edge_index[:, 1]]
        shift_cart = edge_diff - (neighbor - center)
    else:
        raise ValueError("Need edge_diff or cell_displacements to infer cell offsets.")
    frac = shift_cart @ torch.linalg.pinv(cell.to(shift_cart.dtype))
    rounded = torch.round(frac)
    if torch.max(torch.abs(frac - rounded)).item() > atol:
        raise ValueError("Failed to infer integer cell offsets from CURATOR edge data.")
    return rounded.to(shift_cart.dtype)


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _normalize_symbol(raw: str) -> int:
    if raw.isdigit():
        return int(raw)
    return int(ase_atomic_numbers[raw])


def build_atomic_number_to_type_map(
    type_names: Iterable[str],
    type_map: Optional[str] = None,
) -> Dict[int, int]:
    resolved_names = list(type_names)
    if type_map:
        name_to_index = {name: idx for idx, name in enumerate(resolved_names)}
        mapping: Dict[int, int] = {}
        for item in type_map.split(","):
            if not item.strip():
                continue
            src_raw, dst_raw = item.split(":", 1)
            src_z = _normalize_symbol(src_raw.strip())
            dst_key = dst_raw.strip()
            if dst_key.isdigit():
                mapping[src_z] = int(dst_key)
            else:
                if dst_key not in name_to_index:
                    raise ValueError(
                        f"Unknown target atom type '{dst_key}'. Known types: {resolved_names}"
                    )
                mapping[src_z] = name_to_index[dst_key]
        return mapping

    mapping = {}
    for idx, name in enumerate(resolved_names):
        mapping[_normalize_symbol(name)] = idx
    return mapping


def map_atomic_numbers_to_types(numbers: torch.Tensor, z_to_type: Dict[int, int]) -> torch.Tensor:
    result = torch.empty_like(numbers, dtype=torch.long)
    for idx, value in enumerate(numbers.view(-1).tolist()):
        if value not in z_to_type:
            known = sorted(z_to_type)
            raise ValueError(
                f"Atomic number {value} is not supported by the external model. Known atomic numbers: {known}"
            )
        result.view(-1)[idx] = z_to_type[value]
    return result


def extract_state_dict(payload) -> OrderedDict[str, torch.Tensor]:
    if isinstance(payload, nn.Module):
        return OrderedDict(payload.state_dict())
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                return OrderedDict(payload[key])
        if "model" in payload and isinstance(payload["model"], nn.Module):
            return OrderedDict(payload["model"].state_dict())
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            return OrderedDict(payload)
    raise TypeError("Unsupported checkpoint format for external backbone loading.")


def strip_state_dict_prefix(
    state_dict: OrderedDict[str, torch.Tensor],
    prefixes: Iterable[str],
) -> OrderedDict[str, torch.Tensor]:
    prefixes = tuple(prefixes)
    for prefix in prefixes:
        matching = [key for key in state_dict if key.startswith(prefix)]
        if matching:
            return OrderedDict((key[len(prefix):], value) for key, value in state_dict.items() if key.startswith(prefix))
    return state_dict


def resolve_target_layer(
    model: nn.Module,
    target_layer: str,
    fallbacks: Iterable[str] = (),
) -> nn.Module:
    layer = find_layer_by_name_recursive(model, target_layer)
    if layer is not None:
        return layer
    for fallback in fallbacks:
        layer = find_layer_by_name_recursive(model, fallback)
        if layer is not None:
            return layer
    known = [target_layer, *fallbacks]
    raise ValueError(f"Cannot find target layer among: {known}")


def bind_target_layer_aliases(adapter: nn.Module, target_layer: str, module: nn.Module) -> None:
    adapter.readout_mlp = module
    adapter.final_layer = module
    if target_layer not in {"readout_mlp", "final_layer"}:
        setattr(adapter, target_layer, module)


def infer_allegro_cutoff(model: nn.Module) -> float:
    edge_norm = getattr(model, "edge_norm", None)
    if edge_norm is None:
        edge_norm = find_layer_by_name_recursive(model, "edge_norm")
    if edge_norm is not None and getattr(edge_norm, "r_max", None) is not None:
        return float(edge_norm.r_max)
    if getattr(model, "r_max", None) is not None:
        return float(model.r_max)
    raise ValueError("Could not infer Allegro cutoff. Pass `cutoff=` explicitly.")


def infer_allegro_type_names(
    model: nn.Module,
    species: Optional[Iterable[str]] = None,
    *,
    error_hint: Optional[str] = None,
) -> list[str]:
    if species:
        return list(species)
    type_names = getattr(model, "type_names", None)
    if type_names:
        return list(type_names)
    metadata = getattr(model, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("type_names"):
        return str(metadata["type_names"]).split()
    message = "Could not infer Allegro atom type names."
    if error_hint:
        message = f"{message} {error_hint}"
    raise ValueError(message)
