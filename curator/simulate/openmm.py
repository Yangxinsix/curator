from __future__ import annotations

"""Utilities to run CURATOR models inside OpenMM via openmm-ml/OpenMM-Torch."""

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import torch

from curator.data import properties
from curator.data._neighborlist import NeighborListTransform, TorchNeighborList
from curator.layer import find_layer_by_name_recursive
from curator.model import EnsembleModel
from curator.utils import load_models
from ase.data import chemical_symbols, atomic_numbers as symbol_to_z


_EV_TO_KJMOL = 96.48533212331002
_ANGSTROM_PER_NANOMETER = 10.0


class CuratorOpenMM(torch.nn.Module):
    """TorchScript-friendly wrapper that adapts CURATOR models to OpenMM inputs.

    The wrapper expects OpenMM-style inputs (positions in nm, box vectors in nm)
    and returns a scalar potential energy in kJ/mol from ``forward``.
    Forces are available through the exported ``forward_with_forces`` method.
    """

    def __init__(
        self,
        model: Union[torch.nn.Module, Sequence[Union[str, torch.nn.Module]], str],
        atomic_numbers: Union[Sequence[Union[int, str]], torch.Tensor],
        *,
        cutoff: Optional[float] = None,
        transforms: Optional[Sequence[NeighborListTransform]] = None,
        length_scale: float = _ANGSTROM_PER_NANOMETER,
        energy_scale: float = _EV_TO_KJMOL,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()

        model_like = load_models(model)
        self.model = EnsembleModel(model_like) if len(model_like) > 1 else model_like[0]
        self.model.eval()

        if device is None:
            device = next(self.model.parameters()).device
        else:
            device = torch.device(device)
            self.model = self.model.to(device)
        self.model_device = device

        cutoff_val = cutoff
        if cutoff_val is None:
            cutoff_val = find_layer_by_name_recursive(self.model, "cutoff")
        if cutoff_val is None:
            raise ValueError("A cutoff radius is required to build the neighbor list.")
        self.cutoff = float(cutoff_val)

        # store atomic numbers as buffer for TorchForce (OpenMM does not pass them)
        numbers = _to_atomic_number_tensor(atomic_numbers, device=device)
        self.register_buffer("atomic_numbers", numbers, persistent=True)

        # transforms must be a ModuleList for torch.jit.script
        self.transforms = torch.nn.ModuleList(list(transforms) if transforms else [])
        if not any(isinstance(t, NeighborListTransform) for t in self.transforms):
            self.transforms.append(
                TorchNeighborList(
                    cutoff=self.cutoff,
                    return_cell_displacements=True,
                )
            )

        self.length_scale = float(length_scale)
        self.energy_scale = float(energy_scale)
        self.force_scale = self.energy_scale * self.length_scale

        # cache dtype to avoid device/dtype mismatch when scripting
        self.model_dtype = next(self.model.parameters()).dtype
        self.to(self.model_device)

    def _prepare_inputs(
        self,
        positions_nm: torch.Tensor,
        box_vectors_nm: Optional[torch.Tensor],
        atomic_numbers: Optional[torch.Tensor],
        requires_grad: bool = False,
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Build model input dictionary from OpenMM tensors."""

        positions_nm = positions_nm.to(self.model_device)
        positions_nm = positions_nm.to(self.model_dtype)
        positions_nm.requires_grad_(requires_grad)
        pos = positions_nm * self.length_scale

        atoms_data: Dict[str, torch.Tensor] = {
            properties.n_atoms: torch.tensor(
                [pos.shape[0]], dtype=torch.long, device=pos.device
            ),
            properties.atomic_numbers: self.atomic_numbers.to(device=pos.device),
            properties.positions: pos,
            properties.image_idx: torch.zeros(
                (pos.shape[0],), dtype=torch.long, device=pos.device
            ),
        }

        if atomic_numbers is not None:
            atoms_data[properties.atomic_numbers] = atomic_numbers.to(
                dtype=torch.long, device=pos.device
            ).reshape(-1)

        if box_vectors_nm is not None:
            atoms_data[properties.cell] = (
                box_vectors_nm.to(pos.device, dtype=pos.dtype) * self.length_scale
            )

        for transform in self.transforms:
            atoms_data = transform(atoms_data)  # type: ignore[assignment]
        return atoms_data, positions_nm

    def _compute_energy(
        self,
        positions_nm: torch.Tensor,
        box_vectors_nm: Optional[torch.Tensor] = None,
        atomic_numbers: Optional[torch.Tensor] = None,
        requires_grad: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data, pos_handle = self._prepare_inputs(
            positions_nm,
            box_vectors_nm,
            atomic_numbers,
            requires_grad=requires_grad,
        )
        outputs = self.model(data)
        if "energy" not in outputs:
            raise KeyError("Model output does not contain an 'energy' key.")
        energy = outputs["energy"] * self.energy_scale
        return energy, pos_handle

    def forward(
        self,
        positions_nm: torch.Tensor,
        box_vectors_nm: Optional[torch.Tensor] = None,
        atomic_numbers: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return potential energy (kJ/mol) for TorchForce in OpenMM."""
        energy, _ = self._compute_energy(
            positions_nm, box_vectors_nm, atomic_numbers, requires_grad=True
        )
        return energy.sum()

    @torch.jit.export
    def forward_with_forces(
        self,
        positions_nm: torch.Tensor,
        box_vectors_nm: Optional[torch.Tensor] = None,
        atomic_numbers: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return energy and forces (both converted to OpenMM units) for debugging."""
        energy, pos_handle = self._compute_energy(
            positions_nm,
            box_vectors_nm,
            atomic_numbers,
            requires_grad=True,
        )
        energy_sum = energy.sum()
        grad = torch.autograd.grad(
            outputs=[energy_sum],
            inputs=[pos_handle],
            create_graph=False,
            allow_unused=False,
        )[0]
        if grad is None:
            raise RuntimeError("autograd.grad returned None for forces.")
        forces = -grad
        return {"energy": energy, "forces": forces}


def _to_atomic_number_tensor(
    atomic_numbers: Union[Sequence[Union[int, str]], torch.Tensor],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert atomic numbers or symbols to a tensor of Z values."""

    if isinstance(atomic_numbers, torch.Tensor):
        return atomic_numbers.to(device=device, dtype=torch.long)

    z_list = []
    for elem in atomic_numbers:
        if isinstance(elem, str):
            key = elem.capitalize()
            if key not in symbol_to_z:
                raise ValueError(f"Unknown element symbol '{elem}'.")
            z_list.append(symbol_to_z[key])
        elif isinstance(elem, int):
            if elem < 1 or elem >= len(chemical_symbols):
                raise ValueError(f"Atomic number out of range: {elem}")
            z_list.append(elem)
        else:
            raise TypeError(f"Unsupported atomic number entry: {elem}")

    return torch.tensor(z_list, dtype=torch.long, device=device)


def export_curator_to_openmm_torchscript(
    model: Union[torch.nn.Module, Sequence[Union[str, torch.nn.Module]], str],
    output_path: Union[str, Path],
    *,
    atomic_numbers: Union[Iterable[Union[int, str]], torch.Tensor],
    cutoff: Optional[float] = None,
    transforms: Optional[Sequence[NeighborListTransform]] = None,
    length_scale: float = _ANGSTROM_PER_NANOMETER,
    energy_scale: float = _EV_TO_KJMOL,
    device: Optional[Union[str, torch.device]] = None,
) -> Path:
    """Export a CURATOR model to TorchScript suitable for openmm-ml/OpenMM-Torch.

    Args:
        model: Path(s) or module(s) representing the trained CURATOR model.
        output_path: Destination for the scripted module.
        atomic_numbers: Per-atom atomic numbers (ints) or element symbols (str) for the target system.
        cutoff: Neighbor-list cutoff (Angstrom). If ``None``, inferred from the model.
        transforms: Optional neighbor-list transform overrides.
        length_scale: Conversion factor from nm (OpenMM) to Angstrom (model).
        energy_scale: Conversion factor from eV (model) to kJ/mol (OpenMM).
        device: Device to place the model on before scripting.

    Returns:
        Path to the saved TorchScript file.
    """
    wrapper = CuratorOpenMM(
        model=model,
        atomic_numbers=atomic_numbers,
        cutoff=cutoff,
        transforms=transforms,
        length_scale=length_scale,
        energy_scale=energy_scale,
        device=device,
    )
    scripted = torch.jit.script(wrapper)
    output_path = Path(output_path)
    scripted.save(output_path.as_posix())
    return output_path
