from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from curator.model.utils import (
    BatchStructure,
    bind_target_layer_aliases,
    build_atomic_number_to_type_map,
    infer_allegro_cutoff,
    infer_allegro_type_names,
    map_atomic_numbers_to_types,
    resolve_target_layer,
    split_batch_structures,
)

from .utils import (
    ExternalModelSpec,
    build_representation,
    ensure_local_nequip_source_on_path,
    parse_bool,
    register_adapter_loader,
)
from .nequip_net import resolve_nequip_net_artifact


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
        self.cutoff = infer_allegro_cutoff(model)
        self.representation = build_representation(self.cutoff)
        self.type_names = infer_allegro_type_names(model)
        self.z_to_type = build_atomic_number_to_type_map(self.type_names, type_map=type_map)
        target_module = resolve_target_layer(model, target_layer, ("edge_readout",))
        bind_target_layer_aliases(self, target_layer, target_module)

    def _to_nequip_frame(self, struct: BatchStructure):
        from nequip.data import AtomicDataDict
        from nequip.data import from_dict as nequip_from_dict
        from nequip.data._nl import compute_neighborlist_

        # nequip.data.from_dict casts cell-like inputs to torch.get_default_dtype()
        # but leaves tensor positions unchanged, so provide all float fields in
        # the same dtype before handing the frame to NequIP.
        float_dtype = torch.get_default_dtype()
        frame = {
            AtomicDataDict.POSITIONS_KEY: struct.positions.to(dtype=float_dtype),
            AtomicDataDict.ATOMIC_NUMBERS_KEY: struct.numbers,
            AtomicDataDict.ATOM_TYPE_KEY: map_atomic_numbers_to_types(struct.numbers, self.z_to_type),
            AtomicDataDict.PBC_KEY: struct.pbc.view(1, 3),
        }
        if struct.cell is not None:
            frame[AtomicDataDict.CELL_KEY] = struct.cell.to(dtype=float_dtype).view(1, 3, 3)
        if struct.edge_index is not None and struct.edge_diff is not None:
            frame[AtomicDataDict.EDGE_INDEX_KEY] = struct.edge_index.transpose(0, 1).contiguous()
            frame[AtomicDataDict.EDGE_VECTORS_KEY] = struct.edge_diff.to(dtype=float_dtype)
            if struct.edge_dist is not None:
                frame[AtomicDataDict.EDGE_LENGTH_KEY] = struct.edge_dist.to(dtype=float_dtype).view(-1, 1)
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
    resource = resolve_nequip_net_artifact(
        spec.resource,
        version=spec.params.get("version", "0.1"),
        cache_dir=spec.params.get("cache_dir"),
        download=parse_bool(spec.params.get("download"), True),
        timeout_sec=int(spec.params.get("timeout_sec", "300")),
    )
    model = load_saved_model(resource, compile_mode=compile_mode)
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


def _load_allegro_net(spec: ExternalModelSpec, device: Optional[torch.device]) -> nn.Module:
    params = dict(spec.params)
    version = params.get("version", "0.1")
    resource = spec.resource
    if not resource.startswith("nequip.net:"):
        resource = f"nequip.net:{resource}:{version}"
    return _load_allegro(
        ExternalModelSpec(
            scheme="allegro",
            resource=resource,
            params=params,
        ),
        device=device,
    )


register_adapter_loader("allegro", _load_allegro)
register_adapter_loader("allegro_net", _load_allegro_net)

__all__ = ["AllegroAdapter"]
