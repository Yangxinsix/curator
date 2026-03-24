from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from curator.data import properties
from curator.layer import AtomwiseNN
from curator.model.external import ExternalBackboneRepresentation
from curator.model.external_utils import (
    build_image_index,
    build_ptr,
    extract_cells,
    extract_pbc,
    infer_cell_offsets,
    normalize_edge_index,
    require_batch_fields,
    split_batch_structures,
)


def _infer_esen_cutoff(backbone: nn.Module) -> float:
    candidates = (
        getattr(backbone, "cutoff", None),
        getattr(getattr(backbone, "backbone", None), "cutoff", None),
        getattr(getattr(getattr(backbone, "module", None), "backbone", None), "cutoff", None),
    )
    for value in candidates:
        if value is not None:
            return float(value)
    raise ValueError("Could not infer eSEN cutoff. Pass `cutoff=` explicitly.")


class ESENRepresentation(ExternalBackboneRepresentation):
    feature_keys = ("node_features", "node_feats", "edge_features", "edge_feats")
    node_feature_keys = ("node_features", "node_feats")
    edge_feature_keys = ("edge_features", "edge_feats")

    def __init__(
        self,
        backbone: nn.Module,
        cutoff: Optional[float] = None,
        pretrained_path: Optional[str] = None,
        strict_load: bool = True,
        feature_dim: int = 128,
        feature_layer: Optional[str] = "energy_block",
        feature_hook: str = "input",
        readout=AtomwiseNN,
        heads: Optional[list] = None,
        dataset_name: str = "train",
        max_neighbors: int = 300,
        use_fairchem_batch: Optional[bool] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.max_neighbors = int(max_neighbors)
        self.use_fairchem_batch = use_fairchem_batch
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
        return _infer_esen_cutoff(self.backbone)

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

    def _build_plain_batch(self, data: properties.Type):
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
        positions = data[properties.positions].view(-1, 3)
        n_atoms = data[properties.n_atoms].view(-1).to(torch.long)
        edge_index_nx2 = normalize_edge_index(data[properties.edge_idx])
        cells = extract_cells(data, len(n_atoms))
        pbc = extract_pbc(data, len(n_atoms)).to(device=positions.device)
        image_idx = data.get(properties.image_idx)
        if image_idx is None:
            image_idx = build_image_index(n_atoms, positions.device)
        edge_diff = data[properties.edge_diff].view(-1, 3)
        edge_dist = data.get(properties.edge_dist)
        if edge_dist is None:
            edge_dist = torch.linalg.norm(edge_diff, dim=-1)
        structures = split_batch_structures(data)
        if cells is None:
            cell_offsets = positions.new_zeros((edge_index_nx2.shape[0], 3))
            cells = positions.new_zeros((len(n_atoms), 3, 3))
            n_edges_per_structure = torch.tensor(
                [
                    0 if struct.edge_index is None else int(struct.edge_index.shape[0])
                    for struct in structures
                ],
                dtype=torch.long,
                device=positions.device,
            )
        else:
            per_edge_offsets = []
            edge_counts = []
            for struct in structures:
                if struct.edge_index is None:
                    edge_counts.append(0)
                    continue
                edge_counts.append(int(struct.edge_index.shape[0]))
                per_edge_offsets.append(
                    infer_cell_offsets(
                        struct.edge_index,
                        struct.positions,
                        struct.cell,
                        edge_diff=struct.edge_diff,
                        cell_displacements=struct.cell_displacements,
                    )
                )
            cell_offsets = (
                torch.cat(per_edge_offsets, dim=0)
                if per_edge_offsets
                else positions.new_zeros((edge_index_nx2.shape[0], 3))
            )
            n_edges_per_structure = torch.tensor(edge_counts, dtype=torch.long, device=positions.device)

        return {
            "pos": positions,
            "atomic_numbers": data[properties.atomic_numbers].view(-1).to(torch.long),
            "edge_index": edge_index_nx2.transpose(0, 1).contiguous(),
            "edge_vectors": edge_diff,
            "edge_lengths": edge_dist.view(-1, 1),
            "cell": cells.to(device=positions.device, dtype=positions.dtype),
            "pbc": pbc,
            "cell_offsets": cell_offsets,
            "natoms": n_atoms,
            "nedges": n_edges_per_structure,
            "batch": image_idx.view(-1).to(torch.long),
            "ptr": build_ptr(n_atoms, positions.device),
            "num_nodes": torch.tensor([positions.shape[0]], dtype=torch.long, device=positions.device),
            "charge": torch.zeros((len(n_atoms),), dtype=torch.long, device=positions.device),
            "spin": torch.zeros((len(n_atoms),), dtype=torch.long, device=positions.device),
            "fixed": torch.zeros((positions.shape[0],), dtype=torch.long, device=positions.device),
            "tags": torch.zeros((positions.shape[0],), dtype=torch.long, device=positions.device),
            "dataset": self.dataset_name,
        }

    def _build_fairchem_batch(self, data: properties.Type):
        from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

        frames = []
        for struct in split_batch_structures(data):
            if struct.edge_index is None or struct.edge_diff is None:
                raise ValueError("eSEN training requires CURATOR edge_index and edge_diff for direct conversion.")
            edge_index = torch.stack(
                (struct.edge_index[:, 1], struct.edge_index[:, 0]),
                dim=0,
            ).contiguous()
            if struct.cell is None:
                cell = struct.positions.new_zeros((1, 3, 3))
                cell_offsets = struct.positions.new_zeros((edge_index.shape[1], 3))
                pbc = torch.zeros((1, 3), dtype=torch.bool, device=struct.positions.device)
            else:
                cell = struct.cell.view(1, 3, 3)
                cell_offsets = infer_cell_offsets(
                    struct.edge_index,
                    struct.positions,
                    struct.cell,
                    edge_diff=struct.edge_diff,
                    cell_displacements=struct.cell_displacements,
                )
                pbc = struct.pbc.view(1, 3)
            edge_dist = struct.edge_dist
            if edge_dist is None:
                edge_dist = torch.linalg.norm(struct.edge_diff, dim=-1)
            frame = AtomicData.from_dict(
                {
                    "pos": struct.positions,
                    "atomic_numbers": struct.numbers,
                    "cell": cell,
                    "pbc": pbc,
                    "natoms": torch.tensor([struct.num_atoms], dtype=torch.long, device=struct.positions.device),
                    "edge_index": edge_index,
                    "edge_vectors": struct.edge_diff,
                    "edge_lengths": edge_dist.view(-1, 1),
                    "cell_offsets": cell_offsets,
                    "nedges": torch.tensor([edge_index.shape[1]], dtype=torch.long, device=struct.positions.device),
                    "charge": torch.zeros((1,), dtype=torch.long, device=struct.positions.device),
                    "spin": torch.zeros((1,), dtype=torch.long, device=struct.positions.device),
                    "fixed": torch.zeros((struct.num_atoms,), dtype=torch.long, device=struct.positions.device),
                    "tags": torch.zeros((struct.num_atoms,), dtype=torch.long, device=struct.positions.device),
                    "dataset": self.dataset_name,
                }
            )
            frames.append(frame)
        return atomicdata_list_to_batch(frames)

    def _build_native_batch(self, data: properties.Type):
        if self.use_fairchem_batch is False:
            return self._build_plain_batch(data)
        if self.use_fairchem_batch is True:
            return self._build_fairchem_batch(data)
        try:
            return self._build_fairchem_batch(data)
        except ModuleNotFoundError:
            return self._build_plain_batch(data)
