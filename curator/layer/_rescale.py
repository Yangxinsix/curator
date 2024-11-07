import torch
from torch import nn
from typing import Optional, Dict, Union, List
from curator.data import properties
try:
    from torch_scatter import scatter_add
except ImportError:
    from curator.utils import scatter_add
from ase.data import atomic_numbers

# TODO: add __repr__ for modules
class GlobalRescaleShift(torch.nn.Module):
    def __init__(
        self,
        scale_by: Union[float,Dict[str, float],Dict[int, float],None] = None,                            # standard deviation used to rescale output
        shift_by: Union[float,Dict[str, float],Dict[int, float],None ]= None,                            # mean value used to shift output
        scale_trainable: bool=False,
        shift_trainable: bool=False,
        scale_keys: List[str] = ["energy"],
        shift_keys: List[str] = ["energy"],
        atomwise_normalization: bool=True,
        output_keys: List[str] = ["energy", "forces"],
        atomic_energies: Optional[Dict[int, Union[float, torch.Tensor]]] = None,
    ):
        super().__init__()
        self.scale_keys = scale_keys
        self.shift_keys = shift_keys
        self.output_keys = output_keys
        self.register_buffer("atomwise_normalization", torch.tensor(atomwise_normalization))
            
        if scale_by is None and shift_by is None:
            self._initialized = False
        else:
            self._initialized = True
            
        if scale_by is not None:
            self.has_scale = True
            if scale_trainable:
                self.register_parameter("scale_by", torch.tensor([scale_by]))
            else:
                self.register_buffer("scale_by", torch.tensor([scale_by]))
        else:
            self.has_scale = False
            self.register_buffer("scale_by", torch.tensor([1.0]))
            
        if shift_by is not None:
            self.has_shift = True
            if shift_trainable:
                self.register_parameter("shift_by", torch.tensor([shift_by]))
            else:
                self.register_buffer("shift_by", torch.tensor([shift_by]))
        else:
            self.has_shift = False
            self.register_buffer("shift_by", torch.tensor([0.0]))

        self.model_outputs = output_keys
        self._get_atomic_energies_list(atomic_energies)
        
    def forward(self, data: properties.Type) -> properties.Type:
        return self.scale(data, force_process=False)
    
    @torch.jit.export
    def scale(self, data: properties.Type, force_process: bool=False) -> properties.Type:
        data = data.copy()
        if not self.training or force_process:
            if self.has_scale:
                for key in self.scale_keys:
                    data[key] = data[key] * self.scale_by
            if self.has_shift:
                # add atomic energies
                shift_by = data[properties.n_atoms] * self.shift_by if self.atomwise_normalization else self.shift_by    
                if self.shift_by_E0:
                    node_e0 = self.atomic_energies[data[properties.Z]]
                    e0 = scatter_add(node_e0, data[properties.image_idx])
                    shift_by = shift_by + e0
                for key in self.shift_keys:
                    data[key] = data[key] + shift_by       
        return data
    
    @torch.jit.export
    def unscale(self, data: properties.Type, force_process: bool=False) -> properties.Type:
        data = data.copy()
        if self.training or force_process:
            # inverse scale and shift for unscale
            if self.has_shift:
                # add atomic energies
                shift_by = data[properties.n_atoms] * self.shift_by if self.atomwise_normalization else self.shift_by
                if self.shift_by_E0:
                    node_e0 = self.atomic_energies[data[properties.Z]]
                    e0 = scatter_add(node_e0, data[properties.image_idx])
                    shift_by = shift_by + e0
                for key in self.shift_keys:
                    data[key] = data[key] - shift_by
            if self.has_scale:
                for key in self.scale_keys:
                    data[key] = data[key] / self.scale_by
        return data
        
    def _get_atomic_energies_list(self, atomic_energies: Union[Dict[int, float], Dict[str, float], None]):
        if atomic_energies is not None:
            self.shift_by_E0 = True
            atomic_energies_dict = torch.zeros((119,), dtype=torch.float)
            if atomic_energies is not None:
                # convert chemical symbols to atomic numbers
                if isinstance(atomic_energies, Dict):
                    for k, v in atomic_energies.items():
                        if isinstance(k, str):
                            atomic_energies_dict[atomic_numbers[k]] = v
                        else:
                            atomic_energies_dict[k] = v
            self.register_buffer("atomic_energies", atomic_energies_dict)
        else:
            self.shift_by_E0 = False
            self.register_buffer("atomic_energies", torch.zeros((119,), dtype=torch.float))    # dummy buffer for torch script
        
    def datamodule(self, _datamodule):
        if not self._initialized:
            shift_by, scale_by = _datamodule._get_scale_shift()
            if scale_by is not None:
                self.has_scale = True
                self.scale_by = torch.tensor([scale_by])
            if shift_by is not None:
                self.has_shift = True
                self.shift_by = torch.tensor([shift_by])
                
            self.atomwise_normalization = torch.tensor(_datamodule.atomwise_normalization)
            scale_forces = _datamodule.scale_forces
            if scale_forces and "forces" not in self.scale_keys:
                self.scale_keys.append("forces")
            self._get_atomic_energies_list(_datamodule._get_average_E0())

    def __repr__(self):
        return (f"{self.__class__.__name__}(has_scale={self.has_scale}, has_shift={self.has_shift}, scale_by={self.scale_by}, shift_by={self.shift_by}"
            f", scale_keys={self.scale_keys}, shift_keys={self.shift_keys})"
        )
            
            
class PerSpeciesRescaleShift(torch.nn.Module):
    def __init__(
        self,
        scales: Dict[str, float] | Dict[int, float] | None = None,                            # standard deviation used to rescale output
        shifts: Dict[str, float] | Dict[int, float] | None = None,                            # mean value used to shift output
        scales_trainable: bool=False,
        shifts_trainable: bool=False,
        scales_keys: List[str] = ["atomic_energy"],
        shifts_keys: List[str] = ["atomic_energy"],
    ):
        super().__init__()
        self.scales_keys = scales_keys
        self.shifts_keys = shifts_keys
        if scales is None and shifts is None:
            self._initialized = False
        else:
            self._initialized = True
        
        if scales is not None:
            self.has_scales = True
            scales_dict = torch.ones((119,), dtype=torch.float)
            for k, v in scales.items():
                if isinstance(k, str):
                    scales_dict[atomic_numbers[k]] = v
                else:
                    scales_dict[k] = v
            scales = scales_dict
            if scales_trainable:
                self.register_parameter("scales", scales)
            else:
                self.register_buffer("scales", scales)
        else:
            self.has_scales = False
            self.register_buffer("scales", torch.ones((119,), dtype=torch.float))
        
        if shifts is not None:
            self.has_shifts = True
            shifts_dict = torch.zeros((119,), dtype=torch.float)
            for k, v in shifts.items():
                if isinstance(k, str):
                    shifts_dict[atomic_numbers[k]] = v
                else:
                    shifts_dict[k] = v
            shifts = shifts_dict
            if shifts_trainable:
                self.register_parameter("shifts", shifts)
            else:
                self.register_buffer("shifts", shifts)
        else:
            self.has_shifts = False
            self.register_buffer("shifts", torch.zeros((119,), dtype=torch.float))
            
    def forward(self,  data: properties.Type) -> properties.Type:
        if self.has_scales:
            for key in self.scales_keys:
                scales = self.scales[data[properties.Z]]
                data[key] = data[key] * scales
        if self.has_shifts:
            for key in self.shifts_keys:
                shifts = self.shifts[data[properties.Z]]
                data[key] = data[key] + shifts
        return data
    
    def datamodule(self, _datamodule):
        if not self._initialized:
            shifts, scales = _datamodule._get_per_species_scale_shift_()
            self.atomwise_normalization = torch.tensor(_datamodule.atomwise_normalization)

            if shifts is not None:
                self.has_shifts = True
                shifts_dict = torch.zeros((119,), dtype=torch.float)
                for k, v in shifts.items():
                    if isinstance(k, str):
                        shifts_dict[atomic_numbers[k]] = v
                    else:
                        shifts_dict[k] = v
                self.shifts = shifts_dict
            
            if scales is not None:
                self.has_scales = True
                scales_dict = torch.ones((119,), dtype=torch.float)
                for k, v in scales.items():
                    if isinstance(k, str):
                        scales_dict[atomic_numbers[k]] = v
                    else:
                        scales_dict[k] = v
                self.scales = scales_dict

    def __repr__(self):
        scale_shift_info = f'{self.__class__.__name__}(scale={self.scale:.6f}, shift={self.shift:.6f},atomwise={self.atomwise_normalization}'
        if self.shift_by_E0:
            formatted_energies = ", ".join([f"{x:.4f}" for x in self.atomic_energies])
            return scale_shift_info + f'\n, E0={formatted_energies})'
        else:
            return scale_shift_info + ')'