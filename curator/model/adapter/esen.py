from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn

from curator.model.utils import (
    BatchStructure,
    bind_target_layer_aliases,
    infer_cell_offsets,
    resolve_target_layer,
    split_batch_structures,
    structure_to_atoms,
)

from .utils import (
    ExternalModelSpec,
    build_representation,
    register_adapter_loader,
)


def _resolve_task_name(predict_unit, requested: Optional[str]) -> str:
    valid = list(predict_unit.dataset_to_tasks.keys())
    if requested is not None:
        if requested not in valid:
            raise ValueError(f"Unknown eSEN task '{requested}'. Valid tasks: {valid}")
        return requested
    if len(valid) == 1:
        return valid[0]
    raise ValueError(f"eSEN task is ambiguous. Pass '?task=...'. Valid tasks: {valid}")


class ESENAdapter(nn.Module):
    def __init__(
        self,
        predict_unit,
        task_name: str,
        target_layer: str = "energy_block",
        max_neighbors: int = 300,
    ) -> None:
        super().__init__()
        self.predict_unit = predict_unit
        self.model = predict_unit.model
        self.task_name = task_name
        self.target_layer = target_layer
        self.max_neighbors = int(max_neighbors)
        backbone = self.predict_unit.model.module.backbone
        self.cutoff = float(backbone.cutoff)
        self.representation = build_representation(self.cutoff)
        target_module = resolve_target_layer(self.predict_unit.model, target_layer, ("energy_block",))
        bind_target_layer_aliases(self, target_layer, target_module)

    def _to_fairchem_frame_direct(self, struct: BatchStructure):
        from fairchem.core.datasets.atomic_data import AtomicData

        if struct.edge_index is None:
            raise ValueError("Missing CURATOR edge_index for direct eSEN conversion.")
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

        frame = {
            "pos": struct.positions,
            "atomic_numbers": struct.numbers,
            "cell": cell,
            "pbc": pbc,
            "natoms": torch.tensor([struct.num_atoms], dtype=torch.long, device=struct.positions.device),
            "edge_index": edge_index,
            "cell_offsets": cell_offsets,
            "nedges": torch.tensor([edge_index.shape[1]], dtype=torch.long, device=struct.positions.device),
            "charge": torch.zeros((1,), dtype=torch.long, device=struct.positions.device),
            "spin": torch.zeros((1,), dtype=torch.long, device=struct.positions.device),
            "fixed": torch.zeros((struct.num_atoms,), dtype=torch.long, device=struct.positions.device),
            "tags": torch.zeros((struct.num_atoms,), dtype=torch.long, device=struct.positions.device),
            "dataset": self.task_name,
        }
        return AtomicData.from_dict(frame)

    def _to_fairchem_frame_fallback(self, struct: BatchStructure):
        from fairchem.core.datasets.atomic_data import AtomicData

        return AtomicData.from_ase(
            structure_to_atoms(struct),
            task_name=self.task_name,
            r_edges=True,
            radius=self.cutoff,
            max_neigh=self.max_neighbors,
            r_data_keys=["spin", "charge"],
            target_dtype=struct.positions.dtype,
        )

    def _build_batch(self, data):
        from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

        frames = []
        for struct in split_batch_structures(data):
            if struct.edge_index is not None and (
                struct.cell is None or struct.edge_diff is not None or struct.cell_displacements is not None
            ):
                frames.append(self._to_fairchem_frame_direct(struct))
            else:
                frames.append(self._to_fairchem_frame_fallback(struct))
        return atomicdata_list_to_batch(frames)

    def forward(self, data):
        batch = self._build_batch(data)
        if not self.predict_unit.lazy_model_intialized:
            self.predict_unit._lazy_init(batch)
            self.model = self.predict_unit.model
        device = self.predict_unit.device
        dtype = self.predict_unit.inference_settings.base_precision_dtype
        batch = batch.to(device).clone()
        for key, value in batch:
            if torch.is_tensor(value) and value.is_floating_point():
                batch[key] = value.to(dtype)
        self.predict_unit.model.module.on_predict_check(batch)
        self.predict_unit.model(batch)
        return data


def _load_esen(spec: ExternalModelSpec, device: Optional[torch.device]) -> nn.Module:
    try:
        from fairchem.core.calculate import pretrained_mlip
        from fairchem.core.units.mlip_unit import load_predict_unit
    except Exception as exc:
        raise ModuleNotFoundError(
            "eSEN support requires the FAIR Chemistry (fairchem) MLIP inference stack."
        ) from exc

    inference_settings = spec.params.get("inference_settings", "default")
    task_name = spec.params.get("task")
    target_layer = spec.params.get("target_layer", "energy_block")
    max_neighbors = int(spec.params.get("max_neighbors", "300"))
    resource_path = Path(spec.resource)
    if resource_path.is_file():
        predict_unit = load_predict_unit(
            str(resource_path),
            inference_settings=inference_settings,
            device=str(device) if device is not None else None,
        )
    else:
        predict_unit = pretrained_mlip.get_predict_unit(
            spec.resource,
            inference_settings=inference_settings,
            device=str(device) if device is not None else None,
        )
    resolved_task = _resolve_task_name(predict_unit, task_name)
    adapter = ESENAdapter(
        predict_unit=predict_unit,
        task_name=resolved_task,
        target_layer=target_layer,
        max_neighbors=max_neighbors,
    )
    if device is not None:
        adapter.to(device)
    adapter.eval()
    return adapter


register_adapter_loader("esen", _load_esen)

__all__ = ["ESENAdapter"]
