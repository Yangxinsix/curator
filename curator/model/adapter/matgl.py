from __future__ import annotations

import numpy as np
import torch
from ase import Atoms
from torch import nn

from curator.data import properties

from .utils import (
    ExternalModelSpec,
    batch_to_atoms,
    bind_target_layer_aliases,
    build_representation,
    parse_bool,
    register_adapter_loader,
    resolve_target_layer,
)


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
        self.representation = build_representation(float(cutoff))
        target_module = resolve_target_layer(self.core_model, target_layer, ("final_layer", "readout"))
        bind_target_layer_aliases(self, target_layer, target_module)

    def _build_graph_converter(self, element_types: tuple[str, ...], cutoff: float):
        try:
            if self.backend == "DGL":
                from matgl.ext._ase_dgl import Atoms2Graph
            else:
                from matgl.ext._ase_pyg import Atoms2Graph
            return Atoms2Graph(element_types, cutoff)
        except Exception:
            return _MatGLAtoms2GraphFallback(element_types, cutoff, self.backend)

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
        atoms_list = batch_to_atoms(data)
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
    use_potential = parse_bool(spec.params.get("use_potential"), True)
    calc_forces = parse_bool(spec.params.get("calc_forces"), True)
    calc_stresses = parse_bool(spec.params.get("calc_stresses"), False)
    calc_hessian = parse_bool(spec.params.get("calc_hessian"), False)

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
