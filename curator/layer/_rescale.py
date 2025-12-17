import torch
from torch import nn
from typing import Optional, Dict, Union, List, Tuple
from curator.data import properties

try:
    from torch_scatter import scatter_add
except ImportError:  # pragma: no cover
    from curator.utils import scatter_add

from ase.data import atomic_numbers


# --------------------------------------------------------------------------- #
# Core transforms
# --------------------------------------------------------------------------- #
class ScaleTransform(nn.Module):
    """Scales specified properties by a scalar."""

    def __init__(self, keys: List[str], scale: Union[float, torch.Tensor], trainable: bool = False):
        super().__init__()
        scale = scale if torch.is_tensor(scale) else torch.tensor([scale], dtype=torch.float)
        if trainable:
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer("scale", scale)
        self.keys = keys

    def forward(self, data: properties.Type) -> properties.Type:
        for k in self.keys:
            if k in data:
                data[k] = data[k] * self.scale
        return data


class ShiftTransform(nn.Module):
    """
    Shifts specified properties by a scalar.
    - atomwise_shift: if True, properties are per-atom; if False, properties are per-structure.
    - atomwise_normalization: if True and shifting structure-level properties, multiply by n_atoms.
    """

    def __init__(
        self,
        keys: List[str],
        shift: Union[float, torch.Tensor],
        atomwise_shift: bool = False,
        atomwise_normalization: bool = True,
        trainable: bool = False,
    ):
        super().__init__()
        shift = shift if torch.is_tensor(shift) else torch.tensor([shift], dtype=torch.float)
        if trainable:
            self.shift = nn.Parameter(shift)
        else:
            self.register_buffer("shift", shift)
        self.keys = keys
        self.atomwise_shift = atomwise_shift
        self.atomwise_normalization = atomwise_normalization

    def compute_shift(self, data: properties.Type) -> torch.Tensor:
        s = self.shift
        if not self.atomwise_shift and self.atomwise_normalization:
            s = data[properties.n_atoms] * s
        return s

    def forward(self, data: properties.Type) -> properties.Type:
        s = self.compute_shift(data)
        for k in self.keys:
            if k in data:
                data[k] = data[k] + s
        return data


class AtomicEnergyShift(ShiftTransform):
    """
    Adds atomic energy reference per species. Acts like a shift term.
    """

    def __init__(
        self,
        keys: List[str],
        atomic_energies: Optional[Dict[int, float]] = None,
        atomwise_shift: bool = False,
        atomwise_normalization: bool = True,
    ):
        super().__init__(keys=keys, shift=0.0, atomwise_shift=atomwise_shift, atomwise_normalization=atomwise_normalization)
        atomic_energies_dict = torch.zeros((119,), dtype=torch.float)
        if atomic_energies is not None:
            for k, v in atomic_energies.items():
                idx = atomic_numbers[k] if isinstance(k, str) else k
                atomic_energies_dict[idx] = v
        self.register_buffer("atomic_energies", atomic_energies_dict)
        self.register_buffer("shift_by_E0", torch.tensor(atomic_energies is not None))

    def forward(self, data: properties.Type) -> properties.Type:
        if not self.shift_by_E0:
            return data
        node_e0 = self.atomic_energies[data[properties.Z]]
        if self.atomwise_shift:
            shift_term = node_e0
        else:
            shift_term = scatter_add(node_e0, data[properties.image_idx])
            if self.atomwise_normalization:
                shift_term = shift_term  # already per-structure
        for k in self.keys:
            if k in data:
                data[k] = data[k] + shift_term
        return data


# --------------------------------------------------------------------------- #
# Domain-aware wrapper
# --------------------------------------------------------------------------- #
class DomainRescale(nn.Module):
    """
    Holds per-domain scale and shift transforms. Applies domain-specific transforms if present,
    otherwise falls back to default.
    """

    def __init__(
        self,
        default_scales: List[ScaleTransform],
        default_shifts: List[nn.Module],
        domain_key: Optional[str] = None,
    ):
        super().__init__()
        self.domain_key = domain_key
        self.default_scales = nn.ModuleList(default_scales)
        self.default_shifts = nn.ModuleList(default_shifts)
        self.domain_scales = nn.ModuleDict()
        self.domain_shifts = nn.ModuleDict()

    def set_domain(self, domain: str, scales: List[ScaleTransform], shifts: List[nn.Module]):
        self.domain_scales[domain] = nn.ModuleList(scales)
        self.domain_shifts[domain] = nn.ModuleList(shifts)

    def _get_domain(self, data: properties.Type) -> Optional[str]:
        if self.domain_key is None or self.domain_key not in data:
            return None
        dom = data[self.domain_key]
        if torch.is_tensor(dom):
            if dom.numel() == 0:
                return None
            dom = dom.view(-1)[0].item()
        return str(dom)

    def forward(self, data: properties.Type) -> properties.Type:
        dom = self._get_domain(data)
        scales = self.default_scales
        shifts = self.default_shifts
        if dom is not None:
            if dom in self.domain_scales:
                scales = self.domain_scales[dom]
            if dom in self.domain_shifts:
                shifts = self.domain_shifts[dom]
        for s in scales:
            data = s(data)
        for sh in shifts:
            data = sh(data)
        return data


# --------------------------------------------------------------------------- #
# High-level modules
# --------------------------------------------------------------------------- #
class GlobalRescaleShift(nn.Module):
    """
    Rescale/shift outputs with optional atomwise normalization, atomic energies,
    and domain-specific overrides.
    """

    def __init__(
        self,
        scale_by: Union[float, Dict[str, float], None] = None,
        shift_by: Union[float, Dict[str, float], None] = None,
        scale_trainable: bool = False,
        shift_trainable: bool = False,
        scale_keys: List[str] = ["energy"],
        shift_keys: List[str] = ["energy"],
        atomwise_shift: bool = False,
        atomwise_normalization: bool = True,
        output_keys: List[str] = ["energy", "forces"],
        atomic_energies: Optional[Dict[int, Union[float, torch.Tensor]]] = None,
        domain_key: Optional[str] = None,
    ):
        super().__init__()
        self.scale_keys = scale_keys
        self.shift_keys = shift_keys
        self.output_keys = output_keys
        self.atomwise_shift = atomwise_shift
        self.atomwise_normalization = atomwise_normalization
        self.domain_key = domain_key

        # default transforms
        default_scale_val = 1.0 if scale_by is None else (scale_by.get("default", 1.0) if isinstance(scale_by, dict) else scale_by)
        default_scales = [
            ScaleTransform(scale_keys, default_scale_val, trainable=scale_trainable)
        ]
        default_shifts: List[nn.Module] = [
            ShiftTransform(shift_keys, 0.0 if shift_by is None else (shift_by.get("default", 0.0) if isinstance(shift_by, dict) else shift_by), atomwise_shift, atomwise_normalization, trainable=shift_trainable)
        ]
        # atomic energies as shift
        default_shifts.append(AtomicEnergyShift(keys=shift_keys, atomic_energies=atomic_energies, atomwise_shift=atomwise_shift, atomwise_normalization=atomwise_normalization))

        self.rescaler = DomainRescale(default_scales, default_shifts, domain_key=domain_key)
        self._initialized = not (scale_by is None and shift_by is None and atomic_energies is None)

    def forward(self, data: properties.Type) -> properties.Type:
        data = data.copy()
        data = self.rescaler(data)
        return data

    def unscale(self, data: properties.Type, force_process: bool = False) -> properties.Type:
        # best-effort inverse assuming single scale/shift per property; for domain-aware cases,
        # domain selection is handled the same way.
        data = data.copy()
        dom = None
        if isinstance(self.rescaler, DomainRescale):
            dom = self.rescaler._get_domain(data)
            scales = self.rescaler.domain_scales.get(dom, self.rescaler.default_scales)
            shifts = self.rescaler.domain_shifts.get(dom, self.rescaler.default_shifts)
        else:
            scales = self.rescaler.default_scales
            shifts = self.rescaler.default_shifts
        # undo shifts first (excluding atomic energy shift which is not easily invertible without recomputation)
        for sh in shifts:
            if isinstance(sh, AtomicEnergyShift):
                continue
            if isinstance(sh, ShiftTransform):
                s = sh.compute_shift(data)
                for k in sh.keys:
                    if k in data:
                        data[k] = data[k] - s
        # undo scales
        for sc in scales:
            for k in sc.keys:
                if k in data:
                    data[k] = data[k] / sc.scale
        return data

    def datamodule(self, _datamodule):
        # populate from datamodule statistics (single or multi-domain)
        if hasattr(_datamodule, "domain_modules") and isinstance(_datamodule.domain_modules, dict):
            # per-domain stats
            for name, dm in _datamodule.domain_modules.items():
                shift_by, scale_by = dm._get_scale_shift()
                scales = [ScaleTransform(self.scale_keys, scale_by if scale_by is not None else 1.0)]
                shifts: List[nn.Module] = [
                    ShiftTransform(self.shift_keys, shift_by if shift_by is not None else 0.0, self.atomwise_shift, self.atomwise_normalization)
                ]
                atomic_energies = dm._get_average_E0()
                shifts.append(
                    AtomicEnergyShift(self.shift_keys, atomic_energies=atomic_energies, atomwise_shift=self.atomwise_shift, atomwise_normalization=self.atomwise_normalization)
                )
                self.rescaler.set_domain(str(name), scales=scales, shifts=shifts)
            if getattr(_datamodule, "scale_forces", False) and "forces" not in self.scale_keys:
                self.scale_keys.append("forces")
            self._initialized = True
        else:
            if not self._initialized:
                shift_by, scale_by = _datamodule._get_scale_shift()
                # update default transforms
                self.rescaler.default_scales[0].scale.copy_(torch.tensor([1.0 if scale_by is None else scale_by]))
                self.rescaler.default_shifts[0].shift.copy_(torch.tensor([0.0 if shift_by is None else shift_by]))
                atomic_energies = _datamodule._get_average_E0()
                if atomic_energies is not None:
                    ae_tensor = self.rescaler.default_shifts[1].atomic_energies
                    ae_tensor.zero_()
                    for k, v in atomic_energies.items():
                        idx = atomic_numbers[k] if isinstance(k, str) else k
                        ae_tensor[idx] = v
                    self.rescaler.default_shifts[1].shift_by_E0.copy_(torch.tensor(True))
                if getattr(_datamodule, "scale_forces", False) and "forces" not in self.scale_keys:
                    self.scale_keys.append("forces")
                self._initialized = True

    # compatibility helpers
    @property
    def scale_by(self) -> torch.Tensor:
        return self.rescaler.default_scales[0].scale

    @property
    def shift_by(self) -> torch.Tensor:
        return self.rescaler.default_shifts[0].shift

    @property
    def atomic_energies(self) -> torch.Tensor:
        # default atomic energies
        for sh in self.rescaler.default_shifts:
            if isinstance(sh, AtomicEnergyShift):
                return sh.atomic_energies
        return torch.zeros((119,), dtype=torch.float)

    @property
    def shift_by_E0(self) -> torch.Tensor:
        for sh in self.rescaler.default_shifts:
            if isinstance(sh, AtomicEnergyShift):
                return sh.shift_by_E0
        return torch.tensor(False)


class PerSpeciesRescaleShift(nn.Module):
    """
    Per-species scaling/shifting for atomic properties (e.g., atomic_energy).
    """

    def __init__(
        self,
        scales: Union[Dict[str, float], Dict[int, float], None] = None,
        shifts: Union[Dict[str, float], Dict[int, float], None] = None,
        scales_trainable: bool = False,
        shifts_trainable: bool = False,
        scales_keys: List[str] = ["atomic_energy"],
        shifts_keys: List[str] = ["atomic_energy"],
    ):
        super().__init__()
        self.scales_keys = scales_keys
        self.shifts_keys = shifts_keys
        self._initialized = not (scales is None and shifts is None)

        scales_dict = torch.ones((119,), dtype=torch.float)
        if scales is not None:
            for k, v in scales.items():
                idx = atomic_numbers[k] if isinstance(k, str) else k
                scales_dict[idx] = v
        if scales_trainable:
            self.register_parameter("scales", nn.Parameter(scales_dict))
        else:
            self.register_buffer("scales", scales_dict)

        shifts_dict = torch.zeros((119,), dtype=torch.float)
        if shifts is not None:
            for k, v in shifts.items():
                idx = atomic_numbers[k] if isinstance(k, str) else k
                shifts_dict[idx] = v
        if shifts_trainable:
            self.register_parameter("shifts", nn.Parameter(shifts_dict))
        else:
            self.register_buffer("shifts", shifts_dict)

    def forward(self, data: properties.Type) -> properties.Type:
        for key in self.scales_keys:
            if key in data:
                data[key] = data[key] * self.scales[data[properties.Z]]
        for key in self.shifts_keys:
            if key in data:
                data[key] = data[key] + self.shifts[data[properties.Z]]
        return data

    def datamodule(self, _datamodule):
        if not self._initialized:
            _ = _datamodule._get_average_E0()
            self.shifts.copy_(torch.tensor(_datamodule.atomic_energies, dtype=torch.float))

    def __repr__(self):
        return f"{self.__class__.__name__}(scales_keys={self.scales_keys}, shifts_keys={self.shifts_keys})"
