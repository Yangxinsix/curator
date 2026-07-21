from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Optional
import warnings

import numpy as np
import torch
from ase import Atoms
from ase.stress import full_3x3_to_voigt_6_stress
from torch import nn

from curator.data import properties
from curator.model.base import ParameterGroup, Representation, collect_unique_parameters
from curator.model.utils import batch_to_atoms
from .backbone import ExternalRepresentation
from .lora import patch_linear_lora


class MatGLRepresentation(ExternalRepresentation):
    """Curator representation wrapper around a MatGL PES Potential."""

    readout_module_names = (
        "final_layer",
        "readout",
        "readout_layer",
        "output_layer",
        "output_layers",
        "energy_readout",
        "energy_head",
    )

    def __init__(
        self,
        potential: nn.Module,
        *,
        model_outputs: Optional[Iterable[str]] = None,
        state_attr: Optional[torch.Tensor] = None,
        stress_unit: str = "GPa",
        heads: Optional[list] = None,
    ) -> None:
        Representation.__init__(self, heads=heads)
        self.potential = potential
        self.core_model = getattr(potential, "model", potential)
        self.state_attr = state_attr
        self.stress_unit = str(stress_unit)
        self.model_outputs = list(model_outputs or (properties.energy, properties.forces))

        element_types = getattr(self.core_model, "element_types", None)
        cutoff = getattr(self.core_model, "cutoff", None)
        if element_types is None or cutoff is None:
            raise ValueError("MatGL model must define 'element_types' and 'cutoff'.")
        self.element_types = tuple(str(item) for item in element_types)
        self.cutoff = float(cutoff)
        self.graph_converter = self._build_graph_converter(self.element_types, self.cutoff)
        self._matgl_lora_patch_count = self._apply_lora_if_requested()

    @staticmethod
    def _build_graph_converter(element_types: tuple[str, ...], cutoff: float):
        try:
            from matgl.ext.ase import Atoms2Graph

            return Atoms2Graph(element_types, cutoff)
        except Exception as exc:
            raise ModuleNotFoundError(
                "MatGLRepresentation requires matgl.ext.ase.Atoms2Graph from MatGL >=4."
            ) from exc

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

        from torch_geometric.data import Batch

        graph_batch = Batch.from_data_list(graphs)
        lat_batch = torch.stack(lattices, dim=0)
        state_attr_batch = torch.tensor(np.vstack(state_attrs), dtype=lat_batch.dtype)
        if self.state_attr is not None:
            state_attr_batch = self.state_attr
        return graph_batch, lat_batch, state_attr_batch

    @staticmethod
    def _stress_to_voigt(stresses: torch.Tensor, batch_size: int, stress_unit: str) -> torch.Tensor:
        if stresses.numel() <= 1:
            return stresses
        stress_tensor = stresses
        if stress_tensor.dim() == 2 and stress_tensor.shape == (3 * batch_size, 3):
            stress_tensor = stress_tensor.view(batch_size, 3, 3)
        elif stress_tensor.dim() == 2 and stress_tensor.shape == (3, 3) and batch_size == 1:
            stress_tensor = stress_tensor.unsqueeze(0)
        if stress_tensor.dim() == 3 and stress_tensor.shape[1:] == (3, 3):
            if stress_unit.lower() == "gpa":
                stress_tensor = stress_tensor / 160.21766208
            rows = []
            for item in stress_tensor:
                rows.append(
                    torch.as_tensor(
                        full_3x3_to_voigt_6_stress(item.detach().cpu().numpy()),
                        dtype=item.dtype,
                        device=item.device,
                    )
                )
            return torch.stack(rows, dim=0)
        return stresses

    def module_groups(self):
        return OrderedDict((("backbone", [self.potential]),))

    def _readout_modules(self) -> list[nn.Module]:
        modules: list[nn.Module] = []
        seen: set[int] = set()

        def append(module: nn.Module | None) -> None:
            if not isinstance(module, nn.Module):
                return
            module_id = id(module)
            if module_id in seen:
                return
            seen.add(module_id)
            modules.append(module)

        for name in self.readout_module_names:
            append(getattr(self.core_model, name, None))

        if modules:
            return modules

        for name, module in self.core_model.named_children():
            lowered = name.lower()
            if "readout" in lowered or lowered.startswith("output") or lowered.startswith("final"):
                append(module)
        return modules

    def _apply_lora_if_requested(self) -> int:
        from curator.layer.wrappers.config import get_wrapper_config

        wrapper_config = get_wrapper_config()
        if wrapper_config.adapter != "lora":
            return 0
        self.backend = wrapper_config.backend
        self.adapter = wrapper_config.adapter
        self.lora_rank = wrapper_config.lora_rank
        self.lora_alpha = wrapper_config.lora_alpha
        self.lora_freeze_base = wrapper_config.lora_freeze_base
        self.lora_target_groups = wrapper_config.lora_target_groups

        if wrapper_config.lora_target_groups is not None:
            warnings.warn(
                "MatGL native LoRA currently ignores wrapper.lora_target_groups and "
                "patches the native readout only.",
                stacklevel=2,
            )

        readout_modules = self._readout_modules()
        target_modules = readout_modules or [self.potential]
        if not readout_modules:
            warnings.warn(
                "Could not identify a MatGL readout module; patching all nn.Linear "
                "layers in the MatGL potential for LoRA.",
                stacklevel=2,
            )

        if wrapper_config.lora_freeze_base:
            for parameter in self.potential.parameters():
                parameter.requires_grad_(False)

        patched = patch_linear_lora(
            target_modules,
            rank=wrapper_config.lora_rank,
            alpha=wrapper_config.lora_alpha,
            freeze_base=wrapper_config.lora_freeze_base,
        )
        if patched == 0:
            warnings.warn(
                "MatGL native LoRA did not find any nn.Linear layer to patch.",
                stacklevel=2,
            )
        return patched

    def parameter_groups(self) -> list[ParameterGroup]:
        groups: list[ParameterGroup] = []
        readout_modules = self._readout_modules()

        seen: set[int] = set()
        readout_params = collect_unique_parameters(readout_modules, seen=seen)
        backbone_params = collect_unique_parameters([self.potential], seen=seen)

        if backbone_params:
            groups.append(ParameterGroup(name="backbone", params=backbone_params))
        if readout_params:
            groups.append(ParameterGroup(name="readout", params=readout_params))
        return groups

    def export_init_kwargs(self) -> dict:
        return {
            "potential": self.potential,
            "model_outputs": list(self.model_outputs),
            "state_attr": self.state_attr,
            "stress_unit": self.stress_unit,
            "heads": list(self.heads),
        }

    def _potential_device(self) -> torch.device:
        for tensor in self.potential.parameters():
            return tensor.device
        for tensor in self.potential.buffers():
            return tensor.device
        return torch.device("cpu")

    def forward(self, data: properties.Type) -> properties.Type:
        atoms_list = batch_to_atoms(data)
        graph_batch, lat_batch, state_attr_batch = self._batch_graphs(atoms_list)
        device = self._potential_device()
        graph_batch = graph_batch.to(device)
        lat_batch = lat_batch.to(device)
        state_attr_batch = state_attr_batch.to(device)

        energies, forces, stresses, *_ = self.potential(
            graph_batch,
            lat_batch,
            state_attr_batch,
        )
        energies = torch.atleast_1d(energies)
        out = data.copy()
        if properties.energy in self.model_outputs:
            out[properties.energy] = energies
        if properties.forces in self.model_outputs:
            out[properties.forces] = forces
        if properties.stress in self.model_outputs:
            out[properties.stress] = self._stress_to_voigt(stresses, len(atoms_list), self.stress_unit)
        return out


__all__ = ["MatGLRepresentation"]
