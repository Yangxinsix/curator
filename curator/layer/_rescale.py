import torch
from torch import nn
from typing import Optional, Dict, Union, List
from curator.data import properties
from collections import defaultdict
try:
    from torch_scatter import scatter_add
except ImportError:
    from curator.utils import scatter_add
from ase.data import atomic_numbers, chemical_symbols

# TODO: add __repr__ for modules
class GlobalRescaleShift(torch.nn.Module):
    def __init__(
        self,
        scale_by: Union[float, Dict[str, float], Dict[int, float], None] = None,                            # standard deviation used to rescale output
        shift_by: Union[float, Dict[str, float], Dict[int, float], None]= None,                            # mean value used to shift output
        scale_trainable: bool=False,
        shift_trainable: bool=False,
        scale_keys: List[str] = ["energy"],
        shift_keys: List[str] = ["energy"],
        atomwise_shift: bool=False,                   # if the value to be shifted is atomwise or structure-based
        atomwise_scale: bool=True,                    # True: energy scale = scale; False: energy scale = scale * n_atoms
        atomwise_normalization: bool=True,            # if the value to be shifted is normalized to each atom. This is only useful for structure-based properties.
        output_keys: List[str] = ["energy", "forces"],
        atomic_energies: Optional[Dict[int, Union[float, torch.Tensor]]] = None,
        atomic_charges: Optional[Dict[int, Union[float, torch.Tensor]]] = None,
    ):
        super().__init__()
        self.scale_keys = scale_keys
        self.shift_keys = shift_keys
        self.output_keys = output_keys
        self.register_buffer("atomwise_normalization", torch.tensor(atomwise_normalization))        # this should be in buffer because it cannot be changed when you are using the model
        self.atomwise_shift = atomwise_shift
        self.atomwise_scale = atomwise_scale
        
        if scale_by is None and shift_by is None:
            self._initialized = False
        else:
            self._initialized = True

        all_keys = list(dict.fromkeys(self.output_keys + self.scale_keys + self.shift_keys))
        if scale_by is not None and shift_by is not None:       
            self.scale_by = {}
            self.shift_by = {}
            self._register_scale_shift_buffer(keys=all_keys, scale_by=scale_by, shift_by=shift_by)

        self.model_outputs = output_keys
        self._get_atomic_energies_list(atomic_energies)
        self._get_atomic_charges_list(atomic_charges)
        
    def forward(self, data: properties.Type) -> properties.Type:
        return self.scale(data, force_process=False, keys=self.scale_keys)
    
    @torch.jit.export
    # From non-unit normalized standardized data to real unit data
    def scale(self, data: properties.Type, force_process: bool=False, keys: Union[properties.Type, List[properties.Type]]=None) -> properties.Type:
        data = data.copy()
        # make scale more robust. we can directly call scale(data), and the previous calls can be non-modified
        if keys is None:
            keys = self.scale_keys
        
        if not self.training or force_process:
            for key in keys:
                # Skip if key not in data or scale_by
                if key not in data or key not in self.scale_by:
                    continue
                # scale
                # For structure-level energy (not atomwise), scale by n_atoms
                # For atomic_energy or atomwise_scale, just multiply by scale factor
                is_structure_energy = (key == properties.energy and not self.atomwise_scale and self.atomwise_normalization)
                if is_structure_energy:
                    data[key] = data[key] * (self.scale_by[key] * data[properties.n_atoms])  # because scale_by is the std of per-atom energies
                else:
                    data[key] = data[key] * self.scale_by[key]
                # shift (forces should not be shifted, as they are derivatives of energy)
                if key not in self.shift_keys:
                    continue
                # For structure-level energy (not atomwise), shift by n_atoms * shift
                is_structure_shift = (key == properties.energy and not self.atomwise_shift and self.atomwise_normalization)
                if is_structure_shift:
                    shift_by = data[properties.n_atoms] * self.shift_by[key]
                else:
                    shift_by = self.shift_by[key]
                # get atomic energy and charge
                # E0 shift applies to both 'energy' (structure-level) and 'atomic_energy' (per-atom)
                is_energy_key = (key == properties.energy or key == properties.atomic_energy)
                if self.shift_by_E0 and is_energy_key:
                    node_e0 = self.atomic_energies[data[properties.Z]]
                    if self.atomwise_shift or key == properties.atomic_energy:
                        # For atomic_energy, always add per-atom E0
                        shift_by = shift_by + node_e0
                    else:
                        # For structure-level energy, aggregate E0
                        e0 = scatter_add(node_e0, data[properties.image_idx])
                        shift_by = shift_by + e0
                elif self.shift_by_q0 and key == properties.atomic_charge:
                    node_q0 = self.atomic_charges[data[properties.Z]]
                    shift_by = shift_by + node_q0
                data[key] = data[key] + shift_by
        return data
    
    @torch.jit.export
    # From real unit data to non-unit normalized standardized data
    def unscale(self, data: properties.Type, force_process: bool=False, keys: Union[properties.Type, List[properties.Type]]=None) -> properties.Type:
        data = data.copy()
        # make scale more robust. we can directly call scale(data), and the previous calls can be non-modified
        if keys is None:
            keys = self.scale_keys

        if self.training or force_process:
            # inverse scale and shift for unscale
            # First unshift, then unscale
            for key in keys:
                # Skip if key not in data or scale_by
                if key not in data or key not in self.scale_by:
                    continue
                # unshift (forces should not be shifted, as they are derivatives of energy)
                if key in self.shift_keys:
                    is_structure_shift = (key == properties.energy and not self.atomwise_shift and self.atomwise_normalization)
                    if is_structure_shift:
                        shift_by = data[properties.n_atoms] * self.shift_by[key]
                    else:
                        shift_by = self.shift_by[key]
                    # get atomic energy and charge
                    # E0 shift applies to both 'energy' (structure-level) and 'atomic_energy' (per-atom)
                    is_energy_key = (key == properties.energy or key == properties.atomic_energy)
                    if self.shift_by_E0 and is_energy_key:
                        node_e0 = self.atomic_energies[data[properties.Z]]
                        if self.atomwise_shift or key == properties.atomic_energy:
                            # For atomic_energy, always add per-atom E0
                            shift_by = shift_by + node_e0
                        else:
                            # For structure-level energy, aggregate E0
                            e0 = scatter_add(node_e0, data[properties.image_idx])
                            shift_by = shift_by + e0
                    elif self.shift_by_q0 and key == properties.atomic_charge:
                        node_q0 = self.atomic_charges[data[properties.Z]]
                        shift_by = shift_by + node_q0

                    data[key] = data[key] - shift_by
                # unscale
                is_structure_energy = (key == properties.energy and not self.atomwise_scale and self.atomwise_normalization)
                if is_structure_energy:
                    data[key] = data[key] / (self.scale_by[key] * data[properties.n_atoms])
                else:
                    data[key] = data[key] / self.scale_by[key]
        return data
        
    def _get_atomic_energies_list(self, atomic_energies: Union[Dict[int, float], Dict[str, float], None]):
        # from a non-zero dict to all element 119-length dict
        if atomic_energies is not None:
            if not hasattr(self, "shift_by_E0"):
                self.register_buffer("shift_by_E0", torch.tensor(True))
            else:
                self.shift_by_E0.copy_(torch.tensor(True))
            atomic_energies_dict = torch.zeros((119,), dtype=torch.float)
            if atomic_energies is not None:
                # convert chemical symbols to atomic numbers
                if isinstance(atomic_energies, Dict):
                    for k, v in atomic_energies.items():
                        if isinstance(k, str):
                            atomic_energies_dict[atomic_numbers[k]] = v
                        else:
                            atomic_energies_dict[k] = v
            if not hasattr(self, "atomic_energies"):
                self.register_buffer("atomic_energies", atomic_energies_dict)
            else:
                self.atomic_energies.copy_(atomic_energies_dict)
        else:
            if not hasattr(self, "shift_by_E0"):
                self.register_buffer("shift_by_E0", torch.tensor(False))
            else:
                self.shift_by_E0.copy_(torch.tensor(False))
            if not hasattr(self, "atomic_energies"):
                self.register_buffer("atomic_energies", torch.zeros((119,), dtype=torch.float))    # dummy buffer for torch script
            else:
                self.atomic_energies.copy_(torch.zeros((119,), dtype=torch.float))

    def _get_atomic_charges_list(self, atomic_charges: Union[Dict[int, float], Dict[str, float], None]):
        # from a non-zero dict to all element 119-length dict
        if atomic_charges is not None:
            if not hasattr(self, "shift_by_q0"):
                self.register_buffer("shift_by_q0", torch.tensor(True))
            else:
                self.shift_by_q0.copy_(torch.tensor(True))
            atomic_charges_dict = torch.zeros((119,), dtype=torch.float)
            if atomic_charges is not None:
                # convert chemical symbols to atomic numbers
                if isinstance(atomic_charges, Dict):
                    for k, v in atomic_charges.items():
                        if isinstance(k, str):
                            atomic_charges_dict[atomic_numbers[k]] = v
                        else:
                            atomic_charges_dict[k] = v
            if not hasattr(self, "atomic_charges"):
                self.register_buffer("atomic_charges", atomic_charges_dict)
            else:
                self.atomic_charges.copy_(atomic_charges_dict)
        else:
            if not hasattr(self, "shift_by_q0"):
                self.register_buffer("shift_by_q0", torch.tensor(False))
            else:
                self.shift_by_q0.copy_(torch.tensor(False))
            if not hasattr(self, "atomic_charges"):
                self.register_buffer("atomic_charges", torch.zeros((119,), dtype=torch.float))    # dummy buffer for torch script
            else:
                self.atomic_charges.copy_(torch.zeros((119,), dtype=torch.float))

    def _register_scale_shift_buffer(self, keys, scale_by=None, shift_by=None):
        # scale_by/shift_by: Dict[str, float] 或 None
        scale_by = scale_by or {}
        shift_by = shift_by or {}

        for k in keys:
            s = float(scale_by.get(k, 1.0))
            m = float(shift_by.get(k, 0.0))

            s_name = f"scale_by__{k}"
            m_name = f"shift_by__{k}"

            if not hasattr(self, s_name):
                self.register_buffer(s_name, torch.tensor(s))
            else:
                getattr(self, s_name).fill_(s)

            if not hasattr(self, m_name):
                self.register_buffer(m_name, torch.tensor(m))
            else:
                getattr(self, m_name).fill_(m)

            self.scale_by[k] = getattr(self, s_name)
            self.shift_by[k] = getattr(self, m_name)

    def _rebuild_scale_shift_dicts(self):
        """Rebuild scale_by and shift_by dicts from registered buffers after loading state_dict."""
        if not hasattr(self, 'scale_by') or self.scale_by is None:
            self.scale_by = {}
        if not hasattr(self, 'shift_by') or self.shift_by is None:
            self.shift_by = {}
        
        # Find all scale_by__* and shift_by__* buffers
        for name, buffer in self.named_buffers():
            if name.startswith("scale_by__"):
                key = name[len("scale_by__"):]
                self.scale_by[key] = buffer
            elif name.startswith("shift_by__"):
                key = name[len("shift_by__"):]
                self.shift_by[key] = buffer

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # First, register any missing scale_by__* and shift_by__* buffers from state_dict
        # This handles the case where model was initialized without scale_by/shift_by
        if not hasattr(self, 'scale_by') or self.scale_by is None:
            self.scale_by = {}
        if not hasattr(self, 'shift_by') or self.shift_by is None:
            self.shift_by = {}
            
        for key in state_dict:
            if key.startswith(prefix + "scale_by__"):
                buffer_name = key[len(prefix):]
                if not hasattr(self, buffer_name):
                    self.register_buffer(buffer_name, torch.tensor(0.0))
            elif key.startswith(prefix + "shift_by__"):
                buffer_name = key[len(prefix):]
                if not hasattr(self, buffer_name):
                    self.register_buffer(buffer_name, torch.tensor(0.0))
        
        # Call parent's _load_from_state_dict
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        # Rebuild the scale_by and shift_by dicts from buffers
        self._rebuild_scale_shift_dicts()
        self._initialized = True

    # Initialized in base.py
    def datamodule(self, _datamodule):
        if not self._initialized:
            self.scale_by = {}
            self.shift_by = {}
            shift_by, scale_by = _datamodule._get_scale_shift(self.scale_keys) # return Dict[str, float]

            if _datamodule.scale_forces and "forces" not in self.scale_keys:
                self.scale_keys.append("forces")

            all_keys = list(dict.fromkeys(self.output_keys + self.scale_keys + self.shift_keys))
            self._register_scale_shift_buffer(keys=all_keys, scale_by=scale_by, shift_by=shift_by)

            self.atomwise_normalization = torch.tensor(_datamodule.atomwise_normalization)
                
            self._get_atomic_energies_list(_datamodule._get_average_E0())
            self._get_atomic_charges_list(_datamodule._get_average_q0())
            
            self._initialized = True

    def __repr__(self):
        # Ensure dicts are rebuilt before repr
        if not self.scale_by or not self.shift_by:
            self._rebuild_scale_shift_dicts()
        sb = {k: float(v) for k, v in self.scale_by.items()}
        mb = {k: float(v) for k, v in self.shift_by.items()}
        atomic_energies_dict = {chemical_symbols[i]: self.atomic_energies[i] for i in self.atomic_energies.nonzero().squeeze().cpu().numpy()}
        atomic_charges_dict = {chemical_symbols[i]: self.atomic_charges[i] for i in self.atomic_charges.nonzero().squeeze().cpu().numpy()}
        return (f"{self.__class__.__name__}(scale_by={sb}, shift_by={mb}"
            f", shift_by_E0={self.shift_by_E0}, shift_by_q0={self.shift_by_q0}, atomic_energies={atomic_energies_dict}, atomic_charges={atomic_charges_dict}"
            f", scale_keys={self.scale_keys}, shift_keys={self.shift_keys}, atomwise_normalization={self.atomwise_normalization})"
        )
            
            
class PerSpeciesRescaleShift(torch.nn.Module):
    def __init__(
        self,
        scales: Union[Dict[str, float], Dict[int, float], None] = None,                            # standard deviation used to rescale output
        shifts: Union[Dict[str, float], Dict[int, float], None] = None,                            # mean value used to shift output
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
            self.register_buffer("scales", torch.ones((119,), dtype=torch.float))
        
        if shifts is not None:
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
            self.register_buffer("shifts", torch.zeros((119,), dtype=torch.float))
            
    def forward(self,  data: properties.Type) -> properties.Type:
        for key in self.scales_keys:
            scales = self.scales[data[properties.Z]]
            data[key] = data[key] * scales
        for key in self.shifts_keys:
            shifts = self.shifts[data[properties.Z]]
            data[key] = data[key] + shifts
        return data
    
    def datamodule(self, _datamodule):
        if not self._initialized:
            shifts, scales = _datamodule._get_per_species_scale_shift_()
            self.atomwise_normalization = torch.tensor(_datamodule.atomwise_normalization)

            if shifts is not None:
                shifts_dict = torch.zeros((119,), dtype=torch.float)
                for k, v in shifts.items():
                    if isinstance(k, str):
                        shifts_dict[atomic_numbers[k]] = v
                    else:
                        shifts_dict[k] = v
                self.shifts = shifts_dict
            
            if scales is not None:
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
