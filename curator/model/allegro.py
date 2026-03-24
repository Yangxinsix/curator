from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from curator.data import properties
from curator.layer import AtomwiseNN
from curator.model.external import ExternalBackboneRepresentation
from curator.model.external_utils import (
    build_atomic_number_to_type_map,
    build_image_index,
    build_ptr,
    extract_cells,
    extract_pbc,
    map_atomic_numbers_to_types,
    normalize_edge_index,
    require_batch_fields,
)


def _infer_allegro_cutoff(backbone: nn.Module) -> float:
    edge_norm = getattr(backbone, "edge_norm", None)
    if edge_norm is not None and getattr(edge_norm, "r_max", None) is not None:
        return float(edge_norm.r_max)
    if getattr(backbone, "r_max", None) is not None:
        return float(backbone.r_max)
    raise ValueError("Could not infer Allegro cutoff. Pass `cutoff=` explicitly.")


def _infer_type_names(backbone: nn.Module, species: Optional[list[str]]) -> list[str]:
    if species:
        return list(species)
    type_names = getattr(backbone, "type_names", None)
    if type_names:
        return list(type_names)
    metadata = getattr(backbone, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("type_names"):
        return str(metadata["type_names"]).split()
    raise ValueError(
        "Could not infer Allegro atom types. Pass `species=[...]` to the representation."
    )


class AllegroRepresentation(ExternalBackboneRepresentation):
    feature_keys = ("node_features", "edge_features", "node_feats", "edge_feats")
    node_feature_keys = ("node_features", "node_feats")
    edge_feature_keys = ("edge_features", "edge_feats")

    def __init__(
        self,
        backbone: nn.Module,
        cutoff: Optional[float] = None,
        species: Optional[list[str]] = None,
        type_map: Optional[str] = None,
        pretrained_path: Optional[str] = None,
        strict_load: bool = True,
        feature_dim: int = 128,
        feature_layer: Optional[str] = None,
        feature_hook: str = "output",
        readout=AtomwiseNN,
        heads: Optional[list] = None,
    ) -> None:
        self.type_names = _infer_type_names(backbone, species)
        self.z_to_type = build_atomic_number_to_type_map(self.type_names, type_map=type_map)
        super().__init__(
            backbone=backbone,
            cutoff=cutoff,
            pretrained_path=pretrained_path,
            strict_load=strict_load,
            feature_dim=feature_dim,
            feature_layer=feature_layer,
            feature_hook=feature_hook,
            readout=readout,
            heads=heads,
        )

    def _infer_cutoff(self) -> float:
        return _infer_allegro_cutoff(self.backbone)

    def _resolve_feature_tensor_from_container(self, container) -> Optional[torch.Tensor]:
        for key in self.node_feature_keys:
            value = container.get(key) if isinstance(container, dict) else getattr(container, key, None)
            if torch.is_tensor(value):
                return value
        for key in self.edge_feature_keys:
            value = container.get(key) if isinstance(container, dict) else getattr(container, key, None)
            if torch.is_tensor(value):
                return value
        return super()._resolve_feature_tensor_from_container(container)

    def _build_native_batch(self, data: properties.Type):
        require_batch_fields(
            data,
            (
                properties.atomic_numbers,
                properties.positions,
                properties.n_atoms,
                properties.edge_idx,
                properties.edge_diff,
            ),
        )
        numbers = data[properties.atomic_numbers].view(-1).to(torch.long)
        positions = data[properties.positions].view(-1, 3)
        edge_index_nx2 = normalize_edge_index(data[properties.edge_idx])
        image_idx = data.get(properties.image_idx)
        if image_idx is None:
            image_idx = build_image_index(data[properties.n_atoms], positions.device)
        cells = extract_cells(data, len(data[properties.n_atoms].view(-1)))
        pbc = extract_pbc(data, len(data[properties.n_atoms].view(-1))).to(device=positions.device)
        edge_diff = data[properties.edge_diff].view(-1, 3)
        edge_dist = data.get(properties.edge_dist)
        if edge_dist is None:
            edge_dist = torch.linalg.norm(edge_diff, dim=-1)

        batch = {
            "pos": positions,
            "edge_index": edge_index_nx2.transpose(0, 1).contiguous(),
            "edge_vectors": edge_diff,
            "edge_lengths": edge_dist.view(-1, 1),
            "atomic_numbers": numbers,
            "atom_types": map_atomic_numbers_to_types(numbers, self.z_to_type),
            "batch": image_idx.view(-1).to(torch.long),
            "ptr": build_ptr(data[properties.n_atoms], positions.device),
            "num_nodes": torch.tensor([positions.shape[0]], dtype=torch.long, device=positions.device),
        }
        if cells is not None:
            batch["cell"] = cells.to(device=positions.device, dtype=positions.dtype)
        batch["pbc"] = pbc
        return batch
