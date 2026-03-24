from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .utils import (
    ExternalModelSpec,
    BatchStructure,
    bind_target_layer_aliases,
    build_atomic_number_to_type_map,
    build_representation,
    ensure_local_nequip_source_on_path,
    map_atomic_numbers_to_types,
    register_adapter_loader,
    resolve_target_layer,
    split_batch_structures,
)


def _infer_allegro_cutoff(model: nn.Module) -> float:
    edge_norm = getattr(model, "edge_norm", None)
    if edge_norm is None:
        edge_norm = resolve_target_layer(model, "edge_norm")
    cutoff = getattr(edge_norm, "r_max", None)
    if cutoff is None:
        raise ValueError("Could not infer Allegro cutoff from model.edge_norm.r_max.")
    return float(cutoff)


def _infer_allegro_type_names(model: nn.Module) -> list[str]:
    type_names = getattr(model, "type_names", None)
    if type_names:
        return list(type_names)
    metadata = getattr(model, "metadata", None)
    if isinstance(metadata, dict) and "type_names" in metadata:
        return str(metadata["type_names"]).split()
    raise ValueError("Could not infer Allegro atom type names from the loaded model.")


class AllegroAdapter(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        target_layer: str = "edge_readout",
        type_map: Optional[str] = None,
        neighborlist_backend: str = "matscipy",
    ) -> None:
        super().__init__()
        self.model = model
        self.target_layer = target_layer
        self.neighborlist_backend = neighborlist_backend
        self.cutoff = _infer_allegro_cutoff(model)
        self.representation = build_representation(self.cutoff)
        self.type_names = _infer_allegro_type_names(model)
        self.z_to_type = build_atomic_number_to_type_map(self.type_names, type_map=type_map)
        target_module = resolve_target_layer(model, target_layer, ("edge_readout",))
        bind_target_layer_aliases(self, target_layer, target_module)

    def _to_nequip_frame(self, struct: BatchStructure):
        from nequip.data import AtomicDataDict
        from nequip.data import from_dict as nequip_from_dict
        from nequip.data._nl import compute_neighborlist_

        frame = {
            AtomicDataDict.POSITIONS_KEY: struct.positions,
            AtomicDataDict.ATOMIC_NUMBERS_KEY: struct.numbers,
            AtomicDataDict.ATOM_TYPE_KEY: map_atomic_numbers_to_types(struct.numbers, self.z_to_type),
            AtomicDataDict.PBC_KEY: struct.pbc.view(1, 3),
        }
        if struct.cell is not None:
            frame[AtomicDataDict.CELL_KEY] = struct.cell.view(1, 3, 3)
        if struct.edge_index is not None and struct.edge_diff is not None:
            frame[AtomicDataDict.EDGE_INDEX_KEY] = struct.edge_index.transpose(0, 1).contiguous()
            frame[AtomicDataDict.EDGE_VECTORS_KEY] = struct.edge_diff
            if struct.edge_dist is not None:
                frame[AtomicDataDict.EDGE_LENGTH_KEY] = struct.edge_dist.view(-1, 1)
        data = nequip_from_dict(frame)
        if AtomicDataDict.EDGE_INDEX_KEY not in data:
            data = compute_neighborlist_(data, self.cutoff, backend=self.neighborlist_backend)
        return data

    def _build_batch(self, data):
        from nequip.data import AtomicDataDict

        frames = [self._to_nequip_frame(struct) for struct in split_batch_structures(data)]
        return AtomicDataDict.batched_from_list(frames)

    def forward(self, data):
        batch = self._build_batch(data)
        device = next(self.parameters()).device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        self.model(batch)
        return data


def _load_allegro(spec: ExternalModelSpec, device: Optional[torch.device]) -> nn.Module:
    ensure_local_nequip_source_on_path()
    try:
        from nequip.model.saved_models import load_saved_model
    except Exception as exc:
        raise ModuleNotFoundError(
            "Allegro support requires the NequIP saved-model loader and the 'nequip-allegro' package."
        ) from exc

    compile_mode = spec.params.get("compile_mode", "eager")
    target_layer = spec.params.get("target_layer", "edge_readout")
    type_map = spec.params.get("type_map")
    neighborlist_backend = spec.params.get("neighborlist_backend", "matscipy")
    model = load_saved_model(spec.resource, compile_mode=compile_mode)
    adapter = AllegroAdapter(
        model=model,
        target_layer=target_layer,
        type_map=type_map,
        neighborlist_backend=neighborlist_backend,
    )
    if device is not None:
        adapter.to(device)
    adapter.eval()
    return adapter


register_adapter_loader("allegro", _load_allegro)
