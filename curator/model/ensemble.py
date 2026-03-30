import torch
from torch import nn
from curator.data._uncertainty import UncertaintyModule
from .base import NeuralNetworkPotential
from typing import List, Dict, Callable, Optional
from curator.data import properties
try:
    from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min
except ImportError:
    from curator.utils import scatter_add, scatter_mean, scatter_max, scatter_min

class EnsembleModel(UncertaintyModule):
    """
    Ensemble model for evaluating uncertainties
    """
    _SCALAR_UNCERTAINTY_KEYS = (
        properties.e_max,
        properties.e_min,
        properties.e_var,
        properties.e_sd,
        properties.f_var,
        properties.f_sd,
    )
    _PER_ATOM_UNCERTAINTY_KEYS = (
        properties.force_sd_per_atom,
        properties.atomic_energy_sd,
    )

    def __init__(self, models: List[NeuralNetworkPotential], per_atom_uncertainty: bool=False) -> None:
        super().__init__()
        self.models = nn.ModuleList([model for model in models])
        self.compute_uncertainty = True if len(models) > 1 else False
        self.per_atom_uncertainty = per_atom_uncertainty
        self.model_outputs = []
        self.refresh_model_outputs()

    def refresh_model_outputs(self) -> None:
        self.model_outputs = []
        for model in self.models:
            for key in model.model_outputs:
                if key not in self.model_outputs:
                    self.model_outputs.append(key)
        scalar_uncertainty_keys = list(self._SCALAR_UNCERTAINTY_KEYS) if self.compute_uncertainty else []
        per_atom_uncertainty_keys = list(self._PER_ATOM_UNCERTAINTY_KEYS) if self.per_atom_uncertainty else []
        self.set_uncertainty_outputs(
            scalar_keys=scalar_uncertainty_keys,
            per_atom_keys=per_atom_uncertainty_keys,
        )
        for key in [*self.uncertainty_keys, *self.per_atom_uncertainty_keys]:
            if key not in self.model_outputs:
                self.model_outputs.append(key)

    @staticmethod
    def _reconstruct_local_forces(
        edge_index: torch.Tensor,
        edge_forces: torch.Tensor,
        natoms: int,
    ) -> torch.Tensor:
        if edge_index.dim() == 2 and edge_index.shape[0] == 2 and edge_index.shape[1] != 2:
            edge_index = edge_index.T
        forces = torch.zeros((natoms, 3), dtype=edge_forces.dtype, device=edge_forces.device)
        if edge_forces.numel() == 0 or natoms == 0:
            return forces

        forces.index_add_(0, edge_index[:, 0], edge_forces)
        local_j = edge_index[:, 1] < natoms
        if bool(local_j.any()):
            forces.index_add_(0, edge_index[local_j, 1], -edge_forces[local_j])
        return forces

    @staticmethod
    def _get_image_idx(
        data: properties.Type,
        natoms: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if properties.image_idx in data:
            image_idx = data[properties.image_idx]
            return image_idx[:natoms].to(device=device, dtype=dtype)

        if properties.n_atoms in data:
            n_atoms = data[properties.n_atoms].reshape(-1).to(device=device, dtype=dtype)
            if n_atoms.numel() == 1:
                return torch.zeros(natoms, dtype=dtype, device=device)
            return torch.repeat_interleave(
                torch.arange(n_atoms.numel(), dtype=dtype, device=device),
                n_atoms,
            )

        return torch.zeros(natoms, dtype=dtype, device=device)

    @staticmethod
    def _structure_energy_from_atomic(
        atomic_energy_stack: torch.Tensor,
        image_idx: torch.Tensor,
    ) -> torch.Tensor:
        per_model = []
        for atomic_energy in atomic_energy_stack:
            per_atom = atomic_energy.reshape(atomic_energy.shape[0], -1).sum(dim=1)
            per_model.append(scatter_add(per_atom, image_idx, dim=0))
        return torch.stack(per_model, dim=0)

    @staticmethod
    def _force_variance_from_force_stack(force_stack: torch.Tensor) -> torch.Tensor:
        return torch.var(force_stack, dim=0).mean(dim=1)

    def _force_variance_from_edge_stack(
        self,
        edge_forces_stack: torch.Tensor,
        edge_index: torch.Tensor,
        natoms: int,
    ) -> torch.Tensor:
        local_forces_stack = torch.stack(
            [
                self._reconstruct_local_forces(edge_index, edge_forces_stack[i], natoms)
                for i in range(edge_forces_stack.shape[0])
            ],
            dim=0,
        )
        return self._force_variance_from_force_stack(local_forces_stack)

    def _update_uncertainty_outputs(
        self,
        result_dict: Dict[str, torch.Tensor],
        *,
        energy_per_model: torch.Tensor,
        per_atom_force_var: torch.Tensor,
        image_idx: torch.Tensor,
        atomic_energy_sd: Optional[torch.Tensor] = None,
    ) -> None:
        result_dict[properties.e_max] = torch.max(energy_per_model, dim=0).values
        result_dict[properties.e_min] = torch.min(energy_per_model, dim=0).values
        result_dict[properties.e_var] = torch.var(energy_per_model, dim=0)
        result_dict[properties.e_sd] = torch.std(energy_per_model, dim=0)
        result_dict[properties.f_var] = scatter_mean(per_atom_force_var, image_idx, dim=0)
        result_dict[properties.f_sd] = result_dict[properties.f_var].sqrt()
        if self.per_atom_uncertainty:
            result_dict[properties.force_sd_per_atom] = per_atom_force_var.sqrt()
            if atomic_energy_sd is not None:
                result_dict[properties.atomic_energy_sd] = atomic_energy_sd

    def _collect_model_outputs(
        self,
        runner: Callable[[NeuralNetworkPotential], properties.Type],
    ) -> Dict[str, torch.Tensor]:
        model_outputs_lists: Dict[str, List[torch.Tensor]] = {}
        for model in self.models:
            out = runner(model)
            for key in self.model_outputs:
                if key not in out:
                    continue
                if key not in model_outputs_lists:
                    model_outputs_lists[key] = [out[key].detach()]
                else:
                    model_outputs_lists[key].append(out[key].detach())

        return {key: torch.stack(values) for key, values in model_outputs_lists.items()}

    def _aggregate_outputs(
        self,
        data: properties.Type,
        model_outputs_dict: Dict[str, torch.Tensor],
        n_local: Optional[int] = None,
    ) -> properties.Type:
        result_dict: Dict[str, torch.Tensor] = {
            key: value.mean(dim=0)
            for key, value in model_outputs_dict.items()
        }

        if not self.compute_uncertainty:
            return result_dict

        if properties.forces in model_outputs_dict:
            natoms = model_outputs_dict[properties.forces].shape[1]
            image_idx = self._get_image_idx(
                data,
                natoms=natoms,
                device=model_outputs_dict[properties.forces].device,
                dtype=torch.int64,
            )
            self._update_uncertainty_outputs(
                result_dict,
                energy_per_model=model_outputs_dict[properties.energy],
                per_atom_force_var=self._force_variance_from_force_stack(model_outputs_dict[properties.forces]),
                image_idx=image_idx,
            )
            return result_dict

        if properties.atomic_energy not in model_outputs_dict or properties.edge_forces not in model_outputs_dict:
            return result_dict

        atomic_energy_stack = model_outputs_dict[properties.atomic_energy]
        edge_forces_stack = model_outputs_dict[properties.edge_forces]
        natoms = n_local if n_local is not None else atomic_energy_stack.shape[1]
        natoms = int(natoms)
        edge_index = data[properties.edge_idx]
        if edge_index.device != edge_forces_stack.device:
            edge_index = edge_index.to(edge_forces_stack.device)
        image_idx = self._get_image_idx(
            data,
            natoms=natoms,
            device=atomic_energy_stack.device,
            dtype=torch.int64,
        )
        self._update_uncertainty_outputs(
            result_dict,
            energy_per_model=self._structure_energy_from_atomic(atomic_energy_stack[:, :natoms], image_idx),
            per_atom_force_var=self._force_variance_from_edge_stack(edge_forces_stack, edge_index, natoms),
            image_idx=image_idx,
            atomic_energy_sd=torch.std(atomic_energy_stack, dim=0),
        )

        return result_dict

    def forward(self, data: properties.Type) -> properties.Type:
        model_outputs_dict = self._collect_model_outputs(lambda model: model(data))
        result_dict = self._aggregate_outputs(data, model_outputs_dict)

        if properties.energy in data:
            # calculate errors
            if properties.forces in result_dict:
                natoms = result_dict[properties.forces].shape[0]
                device = result_dict[properties.forces].device
            else:
                natoms = int(data[properties.n_atoms].sum().item())
                device = data[properties.energy].device
            image_idx = self._get_image_idx(data, natoms, device, torch.int64)
            e_diff = result_dict[properties.energy] - data[properties.energy]
            f_diff = result_dict[properties.forces] - data[properties.forces]
            result_dict[properties.e_ae] = torch.abs(e_diff)
            result_dict[properties.e_se] = torch.square(e_diff)
            result_dict[properties.f_ae] = scatter_mean(torch.abs(f_diff).mean(dim=1), image_idx, dim=0)
            result_dict[properties.f_se] = scatter_mean(torch.square(f_diff).mean(dim=1), image_idx, dim=0)
            # currently torch scatter does not support jit script these two operations
            # result_dict[properties.f_maxe], _ = scatter_max(f_diff.square().sum(dim=1).sqrt(), data[properties.image_idx], dim=0)
            # result_dict[properties.f_mine], _ = scatter_min(f_diff.square().sum(dim=1).sqrt(), data[properties.image_idx], dim=0)

        return result_dict

    def forward_with_lammps(
        self,
        data: properties.Type,
        lammps_data=None,
        n_local: Optional[int] = None,
        n_ghost: Optional[int] = None,
    ) -> properties.Type:
        model_outputs_dict = self._collect_model_outputs(
            lambda model: model.forward_with_lammps(
                data,
                lammps_data=lammps_data,
                n_local=n_local,
                n_ghost=n_ghost,
            ),
        )
        return self._aggregate_outputs(data, model_outputs_dict, n_local=n_local)

class DropoutModel(nn.Module):
    pass
