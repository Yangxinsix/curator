from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict, Optional
from urllib.parse import parse_qs

import numpy as np
import torch
from ase import Atoms
from torch import nn

from curator.data import properties
from curator.layer.utils import find_layer_by_name_recursive


@dataclass
class ExternalModelSpec:
    scheme: str
    resource: str
    params: Dict[str, str]


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


def parse_external_model_spec(raw: str) -> Optional[ExternalModelSpec]:
    if not isinstance(raw, str) or ":" not in raw:
        return None
    scheme, rest = raw.split(":", 1)
    if not scheme:
        return None
    scheme = scheme.strip().lower()
    resource, sep, query = rest.partition("?")
    resource = resource.strip()
    if not resource:
        return None
    params: Dict[str, str] = {}
    if sep:
        parsed = parse_qs(query, keep_blank_values=True)
        params = {k: v[-1] for k, v in parsed.items() if v}
    return ExternalModelSpec(scheme=scheme, resource=resource, params=params)


_ADAPTER_LOADERS: Dict[str, Callable[[ExternalModelSpec, Optional[torch.device]], nn.Module]] = {}


def register_adapter_loader(scheme: str, loader: Callable[[ExternalModelSpec, Optional[torch.device]], nn.Module]) -> None:
    _ADAPTER_LOADERS[scheme.lower()] = loader


def is_external_model_spec(raw: str) -> bool:
    spec = parse_external_model_spec(raw)
    return spec is not None and spec.scheme in _ADAPTER_LOADERS


def load_external_model(raw: str, device: Optional[torch.device] = None) -> nn.Module:
    spec = parse_external_model_spec(raw)
    if spec is None:
        raise ValueError(f"Invalid external model spec: {raw}")
    loader = _ADAPTER_LOADERS.get(spec.scheme)
    if loader is None:
        known = ", ".join(sorted(_ADAPTER_LOADERS.keys()))
        raise ValueError(f"Unsupported external model scheme '{spec.scheme}'. Known schemes: {known}")
    return loader(spec, device=device)


class _MatGLAtoms2GraphFallback:
    """Fallback converter avoiding optional MatGL neighbor-list dependencies."""

    def __init__(self, element_types: tuple[str, ...], cutoff: float, backend: str) -> None:
        self.element_types = tuple(element_types)
        self.cutoff = float(cutoff)
        self.backend = backend.upper()

        if self.backend == "DGL":
            from matgl.graph._converters_dgl import GraphConverter
        else:
            from matgl.graph._converters_pyg import GraphConverter

        class _Converter(GraphConverter):
            def get_graph(self, structure):
                raise NotImplementedError

        self._converter = _Converter()

    def get_graph(self, atoms: Atoms):
        try:
            from matscipy.neighbours import neighbour_list
        except Exception as exc:
            raise ModuleNotFoundError(
                "matscipy is required for MatGL fallback neighbor list."
            ) from exc

        periodic = bool(atoms.pbc.all())
        work_atoms = atoms

        # matscipy expects an invertible cell even when pbc=False.
        if not periodic:
            work_atoms = atoms.copy()
            pos = np.asarray(work_atoms.get_positions())
            if pos.size == 0:
                lengths = np.array([1.0, 1.0, 1.0], dtype=np.float64)
            else:
                mins = pos.min(axis=0)
                maxs = pos.max(axis=0)
                span = np.maximum(maxs - mins, 1.0)
                pad = max(2.0 * self.cutoff, 1.0)
                lengths = np.maximum(span + 2.0 * pad, 1.0)
                center = 0.5 * (mins + maxs)
                shift = 0.5 * lengths - center
                work_atoms.positions = pos + shift
            work_atoms.set_cell(np.diag(lengths))
            work_atoms.set_pbc((False, False, False))

        src_id, dst_id, images, _ = neighbour_list("ijSD", work_atoms, self.cutoff)

        if periodic:
            frac_or_cart_coords = np.asarray(atoms.get_scaled_positions(False), dtype=np.float32)
            lattice_for_graph = [np.array(atoms.cell.array, dtype=np.float32)]
            images = np.asarray(images, dtype=np.int64)
        else:
            frac_or_cart_coords = np.asarray(atoms.get_positions(), dtype=np.float32)
            lattice_for_graph = np.expand_dims(np.identity(3, dtype=np.float32), axis=0)
            images = np.zeros((len(src_id), 3), dtype=np.int64)

        return self._converter.get_graph_from_processed_structure(
            atoms,
            src_id,
            dst_id,
            images,
            lattice_for_graph,
            self.element_types,
            frac_or_cart_coords,
            is_atoms=True,
        )


class MatGLAdapter(nn.Module):
    """Adapter that lets MatGL models run on CURATOR atoms batches."""

    def __init__(
        self,
        model: nn.Module,
        backend: str = "PYG",
        target_layer: str = "final_layer",
        use_potential: bool = True,
        calc_forces: bool = True,
        calc_stresses: bool = False,
        calc_hessian: bool = False,
    ) -> None:
        super().__init__()
        self.backend = backend.upper()
        self.target_layer = target_layer
        self.model = model

        from matgl.apps.pes import Potential

        if isinstance(model, Potential):
            self.potential = model
            self.core_model = model.model
        elif use_potential:
            self.potential = Potential(
                model=model,
                calc_forces=calc_forces,
                calc_stresses=calc_stresses,
                calc_hessian=calc_hessian,
            )
            self.core_model = self.potential.model
        else:
            self.potential = None
            self.core_model = model

        element_types = getattr(self.core_model, "element_types", None)
        cutoff = getattr(self.core_model, "cutoff", None)
        if element_types is None or cutoff is None:
            raise ValueError("MatGL model must define 'element_types' and 'cutoff'.")

        if self.backend not in {"DGL", "PYG"}:
            raise ValueError(f"Unsupported MatGL backend: {self.backend}")
        self.graph_converter = self._build_graph_converter(tuple(element_types), float(cutoff))

        # CURATOR select path expects models[i].representation.cutoff.
        self.representation = SimpleNamespace(cutoff=float(cutoff))

        # Keep CURATOR default target name usable without extra config.
        self.readout_mlp = self._resolve_target_layer(self.target_layer)
        if self.target_layer != "readout_mlp":
            setattr(self, self.target_layer, self.readout_mlp)
        self.final_layer = self.readout_mlp

    def _resolve_target_layer(self, target_layer: str) -> nn.Module:
        layer = find_layer_by_name_recursive(self.core_model, target_layer)
        if layer is not None:
            return layer
        for fallback in ("final_layer", "readout"):
            layer = find_layer_by_name_recursive(self.core_model, fallback)
            if layer is not None:
                return layer
        raise ValueError(
            f"Cannot find target layer '{target_layer}' (or fallbacks final_layer/readout) in MatGL model."
        )

    def _build_graph_converter(self, element_types: tuple[str, ...], cutoff: float):
        try:
            if self.backend == "DGL":
                from matgl.ext._ase_dgl import Atoms2Graph
            else:
                from matgl.ext._ase_pyg import Atoms2Graph
            return Atoms2Graph(element_types, cutoff)
        except Exception:
            return _MatGLAtoms2GraphFallback(element_types, cutoff, self.backend)

    def _extract_cells(self, data: properties.Type, batch_size: int) -> Optional[torch.Tensor]:
        cell = data.get(properties.cell)
        if cell is None:
            return None
        if not torch.is_tensor(cell) or cell.numel() == 0:
            return None
        cell = cell.detach().cpu()

        if cell.dim() == 3 and cell.shape[0] == batch_size and cell.shape[1:] == (3, 3):
            return cell
        if cell.dim() == 2 and cell.shape == (3, 3) and batch_size == 1:
            return cell.unsqueeze(0)
        if cell.dim() == 2 and cell.shape[1] == 3 and cell.shape[0] == 3 * batch_size:
            return cell.view(batch_size, 3, 3)
        if cell.dim() == 1 and cell.numel() == 9 * batch_size:
            return cell.view(batch_size, 3, 3)
        return None

    def _batch_to_atoms(self, data: properties.Type) -> list[Atoms]:
        if properties.n_atoms not in data or properties.atomic_numbers not in data or properties.positions not in data:
            raise KeyError(
                f"Batch must include {properties.n_atoms}, {properties.atomic_numbers}, and {properties.positions}."
            )
        n_atoms = data[properties.n_atoms].detach().view(-1).to("cpu", torch.long).tolist()
        z = data[properties.atomic_numbers].detach().to("cpu", torch.long).view(-1)
        pos = data[properties.positions].detach().to("cpu").view(-1, 3)
        cells = self._extract_cells(data, len(n_atoms))

        atoms_list: list[Atoms] = []
        offset = 0
        for i, n in enumerate(n_atoms):
            n_int = int(n)
            numbers = z[offset : offset + n_int].numpy()
            positions = pos[offset : offset + n_int].numpy()
            if cells is not None:
                atoms = Atoms(numbers=numbers, positions=positions, cell=cells[i].numpy(), pbc=True)
            else:
                atoms = Atoms(numbers=numbers, positions=positions, pbc=False)
            atoms_list.append(atoms)
            offset += n_int
        return atoms_list

    @staticmethod
    def _normalize_lattice(lat: torch.Tensor) -> torch.Tensor:
        lat_t = torch.as_tensor(lat).detach().cpu()
        if lat_t.dim() == 3 and lat_t.shape[0] == 1:
            return lat_t[0]
        if lat_t.dim() == 2:
            return lat_t
        if lat_t.dim() == 1 and lat_t.numel() == 9:
            return lat_t.view(3, 3)
        raise ValueError(f"Unsupported lattice shape from MatGL converter: {tuple(lat_t.shape)}")

    def _batch_graphs(self, atoms_list: list[Atoms]):
        graphs = []
        lattices = []
        state_attrs = []
        for atoms in atoms_list:
            g, lat, state_attr = self.graph_converter.get_graph(atoms)
            graphs.append(g)
            lattices.append(self._normalize_lattice(lat))
            state_attrs.append(np.asarray(state_attr, dtype=np.float32))

        if self.backend == "DGL":
            import dgl

            graph_batch = dgl.batch(graphs)
        else:
            from torch_geometric.data import Batch

            graph_batch = Batch.from_data_list(graphs)

        lat_batch = torch.stack(lattices, dim=0)
        state_attr_batch = torch.tensor(np.vstack(state_attrs), dtype=lat_batch.dtype)
        return graph_batch, lat_batch, state_attr_batch

    def forward(self, data: properties.Type) -> properties.Type:
        atoms_list = self._batch_to_atoms(data)
        graph_batch, lat_batch, state_attr_batch = self._batch_graphs(atoms_list)

        device = next(self.parameters()).device
        graph_batch = graph_batch.to(device)
        lat_batch = lat_batch.to(device)
        state_attr_batch = state_attr_batch.to(device)

        if self.potential is not None:
            self.potential(g=graph_batch, lat=lat_batch, state_attr=state_attr_batch)
        else:
            self.core_model(g=graph_batch, state_attr=state_attr_batch)
        return data


def _load_matgl(spec: ExternalModelSpec, device: Optional[torch.device]) -> nn.Module:
    import matgl

    backend = spec.params.get("backend", "PYG").upper()
    target_layer = spec.params.get("target_layer", "final_layer")
    use_potential = _parse_bool(spec.params.get("use_potential"), True)
    calc_forces = _parse_bool(spec.params.get("calc_forces"), True)
    calc_stresses = _parse_bool(spec.params.get("calc_stresses"), False)
    calc_hessian = _parse_bool(spec.params.get("calc_hessian"), False)

    matgl.set_backend(backend)
    model = matgl.load_model(spec.resource)
    adapter = MatGLAdapter(
        model=model,
        backend=backend,
        target_layer=target_layer,
        use_potential=use_potential,
        calc_forces=calc_forces,
        calc_stresses=calc_stresses,
        calc_hessian=calc_hessian,
    )
    if device is not None:
        adapter.to(device)
    adapter.eval()
    return adapter


register_adapter_loader("matgl", _load_matgl)
