from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import torch
from torch import nn

from curator.data import properties
from curator.layer import AtomwiseNN
from curator.model.utils import (
    build_atomic_number_to_type_map,
    build_image_index,
    build_ptr,
    extract_cells,
    extract_pbc,
    infer_allegro_cutoff,
    infer_allegro_type_names,
    map_atomic_numbers_to_types,
    normalize_edge_index,
    require_batch_fields,
)

from .backbone import ExternalBackboneRepresentation


class AllegroRepresentation(ExternalBackboneRepresentation):
    feature_keys = ("node_features", "edge_features", "node_feats", "edge_feats")
    node_feature_keys = ("node_features", "node_feats")
    edge_feature_keys = ("edge_features", "edge_feats")

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        backbone_builder: Optional[Callable[..., nn.Module] | str] = None,
        backbone_kwargs: Optional[Mapping[str, Any]] = None,
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
        self.species = list(species) if species is not None else None
        self.type_map = type_map
        super().__init__(
            backbone=backbone,
            backbone_builder=backbone_builder,
            backbone_kwargs=backbone_kwargs,
            cutoff=cutoff,
            pretrained_path=pretrained_path,
            strict_load=strict_load,
            feature_dim=feature_dim,
            feature_layer=feature_layer,
            feature_hook=feature_hook,
            readout=readout,
            heads=heads,
        )
        self.type_names = infer_allegro_type_names(
            self.backbone,
            self.species,
            error_hint="Pass `species=[...]` to the representation.",
        )
        self.z_to_type = build_atomic_number_to_type_map(self.type_names, type_map=self.type_map)

    def _infer_cutoff(self) -> float:
        return infer_allegro_cutoff(self.backbone)

    def export_init_kwargs(self) -> dict[str, Any]:
        rep_config: dict[str, Any] = super().export_init_kwargs()
        rep_config["species"] = list(self.type_names)
        rep_config["type_map"] = self.type_map
        return rep_config

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
