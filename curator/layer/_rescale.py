from typing import Optional, Dict, Union, List
import torch
from torch import nn
from curator.data import properties
from curator.data.datamodule import DataContext
from curator.data.properties import HeadConfig, HEAD_PRESETS, resolve_heads, normalize_head_flag

try:
    from torch_scatter import scatter_add
except ImportError:  # pragma: no cover
    from curator.utils import scatter_add

from ase.data import atomic_numbers


# --------------------------------------------------------------------------- #
# Basic transforms
# --------------------------------------------------------------------------- #
class ScaleTransform(nn.Module):
    def __init__(self, key: str, scale: Union[float, torch.Tensor], trainable: bool = False):
        super().__init__()
        scale = scale if torch.is_tensor(scale) else torch.tensor([scale], dtype=torch.float)
        if trainable:
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer("scale", scale)
        self.key = key

    def forward(self, data: properties.Type) -> properties.Type:
        if self.key in data:
            data[self.key] = data[self.key] * self.scale
        return data


class ShiftTransform(nn.Module):
    """Simple shift: add scalar shift to a property."""

    def __init__(self, key: str, shift: Union[float, torch.Tensor], trainable: bool = False):
        super().__init__()
        shift = shift if torch.is_tensor(shift) else torch.tensor([shift], dtype=torch.float)
        if trainable:
            self.shift = nn.Parameter(shift)
        else:
            self.register_buffer("shift", shift)
        self.key = key

    def forward(self, data: properties.Type) -> properties.Type:
        if self.key not in data:
            return data
        data[self.key] = data[self.key] + self.shift
        return data


class AtomwiseShift(nn.Module):
    """
    Shift with atomwise control:
    - atomwise_shift=True: apply a per-atom shift (broadcast) to atomwise property.
    - atomwise_shift=False: apply structure-level shift; if atomwise_normalization=True, multiply by n_atoms.
    """

    def __init__(
        self,
        key: str,
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
        self.key = key
        self.atomwise_shift = atomwise_shift
        self.atomwise_normalization = atomwise_normalization

    def compute_shift(self, data: properties.Type) -> torch.Tensor:
        if self.atomwise_shift:
            return self.shift
        if self.atomwise_normalization:
            return data[properties.n_atoms] * self.shift
        return self.shift

    def forward(self, data: properties.Type) -> properties.Type:
        if self.key not in data:
            return data
        if self.atomwise_shift:
            data[self.key] = data[self.key] + self.shift
            return data

        s = self.compute_shift(data)
        data[self.key] = data[self.key] + s
        return data


class PerSpeciesShift(nn.Module):
    """
    Universal per-species shift for a property. Values are specified per atomic number.

    If `atomwise_shift=True`, adds values per atom.
    Otherwise, sums per-atom values into per-structure shift using `image_idx`.
    """

    def __init__(
        self,
        key: str,
        values: Optional[Dict[int, float]],
        atomwise_shift: bool = False,
    ):
        super().__init__()
        values_dict = torch.zeros((119,), dtype=torch.float)
        if values is not None:
            for k, v in values.items():
                idx = atomic_numbers[k] if isinstance(k, str) else k
                values_dict[idx] = v
        self.register_buffer("values", values_dict)
        self.register_buffer("enabled", torch.tensor(values is not None))
        self.key = key
        self.atomwise_shift = atomwise_shift

    def load_values(self, values: Dict[int, float]):
        self.values.zero_()
        for k, v in values.items():
            idx = atomic_numbers[k] if isinstance(k, str) else k
            self.values[idx] = v
        self.enabled.copy_(torch.tensor(True))

    def forward(self, data: properties.Type) -> properties.Type:
        return self.apply(data, sign=1.0)

    def apply(
        self,
        data: properties.Type,
        sign: float = 1.0,
    ) -> properties.Type:
        if not self.enabled or self.key not in data:
            return data
        per_atom = self.values[data[properties.Z]]
        if self.atomwise_shift:
            data[self.key] = data[self.key] + sign * per_atom
            return data

        # structure-level: sum over atoms
        shift_term = scatter_add(per_atom, data[properties.image_idx])
        data[self.key] = data[self.key] + sign * shift_term
        return data

# --------------------------------------------------------------------------- #
# High-level modules
# --------------------------------------------------------------------------- #
class GlobalRescaleShift(nn.Module):
    """
    Rescale/shift outputs with per-property HeadConfig. Domain-specific handling is provided
    by MultiDomainRescaleShift.
    """

    def __init__(
        self,
        heads: Optional[List[HeadConfig]] = None,
        scale_trainable: bool = False,
        shift_trainable: bool = False,
    ):
        super().__init__()
        if heads is not None and hasattr(heads, "get") and "heads" in heads:
            heads = heads.get("heads")
        if heads is None:
            heads = [HEAD_PRESETS["energy"]]
        # ensure heads are HeadConfig instances
        self.heads = resolve_heads([h.key if isinstance(h, HeadConfig) and h.key in HEAD_PRESETS else h for h in heads])
        # build per-head transforms
        scales = []
        shifts = []
        per_species_shifts = []
        self._atomwise_output_keys: Dict[str, bool] = {}
        for h in self.heads:
            self._atomwise_output_keys[h.key] = bool(h.reduction is None)
            # scale
            s_val = 1.0
            if isinstance(h.scale_by, (int, float)) and not isinstance(h.scale_by, bool):
                s_val = h.scale_by
            scales.append(ScaleTransform(h.key, s_val, trainable=scale_trainable))
            # shift
            sh_val = 0.0
            if isinstance(h.shift_by, (int, float)) and not isinstance(h.shift_by, bool):
                sh_val = h.shift_by
            if h.atomwise_shift or h.atomwise_normalization:
                shifts.append(AtomwiseShift(h.key, sh_val, atomwise_shift=h.atomwise_shift, atomwise_normalization=h.atomwise_normalization, trainable=shift_trainable))
            else:
                shifts.append(ShiftTransform(h.key, sh_val, trainable=shift_trainable))

            # per-species shift (e.g., atomic energies), optional
            # per-species shift: create with initial values if provided; else None (enable via context if "auto")
            init_values = None
            if isinstance(h.per_species_shift, dict):
                init_values = h.per_species_shift
            per_species_shifts.append(PerSpeciesShift(h.key, values=init_values, atomwise_shift=h.atomwise_shift))

        self.scales = nn.ModuleList(scales)
        self.shifts = nn.ModuleList(shifts)
        self.atomic_shifts = nn.ModuleList(per_species_shifts)
        self._initialized = not any(
            normalize_head_flag(h.scale_by) in ("default", "rms")
            or normalize_head_flag(h.shift_by) == "default"
            for h in self.heads
        )

    def forward(self, data: properties.Type) -> properties.Type:
        return self.scale(data, force_process=False)

    def scale(
        self,
        data: properties.Type,
        force_process: bool = False,
    ) -> properties.Type:
        data = data.copy()
        if self.training and not force_process:
            return data
        # scale then shifts
        for sc in self.scales:
            data = sc(data)
        for sh in self.atomic_shifts:
            data = sh(data)
        for sh in self.shifts:
            if isinstance(sh, AtomwiseShift):
                data = sh(data)
            else:
                data = sh(data)
        return data

    def unscale(
        self,
        data: properties.Type,
        force_process: bool = False,
    ) -> properties.Type:
        data = data.copy()
        if not self.training and not force_process:
            return data

        # undo shifts (structure + atomwise)
        for sh in self.shifts:
            if sh.key not in data:
                continue
            if isinstance(sh, AtomwiseShift):
                if sh.atomwise_shift:
                    data[sh.key] = data[sh.key] - sh.shift
                else:
                    s = sh.compute_shift(data)
                    data[sh.key] = data[sh.key] - s
            else:
                data[sh.key] = data[sh.key] - sh.shift

        for sh in self.atomic_shifts:
            data = sh.apply(data, sign=-1.0)

        # undo scale
        for sc in self.scales:
            if sc.key in data:
                data[sc.key] = data[sc.key] / sc.scale

        return data

    def setup_from_context(self, ctx: DataContext):
        if self._initialized:
            return
        for i, head in enumerate(self.heads):
            stats = ctx.head_scale_shift.get(head.key, {"mean": 0.0, "std": 1.0})
            scale_mode = normalize_head_flag(head.scale_by)
            shift_mode = normalize_head_flag(head.shift_by)

            if scale_mode in ("default", "rms"):
                scale_by = stats.get("std", 1.0)
            elif isinstance(scale_mode, (int, float)) and not isinstance(scale_mode, bool):
                scale_by = float(scale_mode)
            else:
                scale_by = 1.0

            if shift_mode == "default":
                shift_by = stats.get("mean", 0.0)
            elif isinstance(shift_mode, (int, float)) and not isinstance(shift_mode, bool):
                shift_by = float(shift_mode)
            else:
                shift_by = 0.0

            self.scales[i].scale.copy_(torch.tensor([scale_by], dtype=self.scales[i].scale.dtype))

            shift_module = self.shifts[i]
            if hasattr(shift_module, "shift"):
                shift_module.shift.copy_(torch.tensor([shift_by], dtype=shift_module.shift.dtype))

        # per-species shift handling
        if isinstance(head.per_species_shift, dict):
            self.atomic_shifts[i].load_values(head.per_species_shift)
        elif isinstance(head.per_species_shift, str) and head.per_species_shift == "auto":
            atomic_values = ctx.head_species_shift.get(head.key, None)
            if atomic_values is not None:
                self.atomic_shifts[i].load_values(atomic_values)
        self._initialized = True

    def setup_from_datamodule(self, dm):
        try:
            ctx = dm.build_context(self.heads)
        except Exception:
            return
        return self.setup_from_context(ctx)

    # compatibility helpers for legacy access (first head)
    @property
    def scale_by(self) -> torch.Tensor:
        return self.scales[0].scale

    @property
    def shift_by(self) -> torch.Tensor:
        shift0 = self.shifts[0]
        return shift0.shift if hasattr(shift0, "shift") else torch.tensor([0.0])

    @property
    def atomic_energies(self) -> torch.Tensor:
        # legacy name; values are per-species shift table
        return self.atomic_shifts[0].values

    @property
    def shift_by_E0(self) -> torch.Tensor:
        # legacy name; enabled flag for per-species shift
        return self.atomic_shifts[0].enabled

    def __repr__(self):
        return f"{self.__class__.__name__}(heads={[h.key for h in self.heads]})"


class MultiDomainRescaleShift(nn.Module):
    """
    Domain-aware wrapper holding one GlobalRescaleShift per domain. Selects by properties.domain at forward.
    """

    def __init__(self, heads: List[HeadConfig]):
        super().__init__()
        if hasattr(heads, "get") and "heads" in heads:
            heads = heads.get("heads")
        self.heads = resolve_heads(heads)
        self.domain_modules = nn.ModuleDict()

    def setup_from_datamodule(self, dm):
        if hasattr(dm, "build_contexts") and hasattr(dm, "domain_modules"):
            contexts = dm.build_contexts(self.heads)
            for dom_id, ctx in contexts.items():
                if dom_id == "global":
                    continue
                # filter heads for this domain if domains are specified
                filtered_heads = []
                for h in self.heads:
                    if h.domains is None:
                        filtered_heads.append(h)
                    else:
                        ds = [str(d) for d in h.domains]
                        if dom_id in ds:
                            filtered_heads.append(h)
                if not filtered_heads:
                    continue
                grs = GlobalRescaleShift(heads=filtered_heads)
                grs.setup_from_context(ctx)
                self.domain_modules[str(dom_id)] = grs
        else:
            grs = GlobalRescaleShift(heads=self.heads)
            grs.setup_from_datamodule(dm)
            self.domain_modules["0"] = grs

    def _get_domain(self, data: properties.Type) -> str:
        dom = None
        if properties.domain in data:
            dom = data[properties.domain]
        elif properties.domain_atom in data:
            dom = data[properties.domain_atom]
        if dom is None:
            if "0" in self.domain_modules:
                return "0"
            return next(iter(self.domain_modules.keys()))
        if torch.is_tensor(dom):
            if dom.numel() == 0:
                if "0" in self.domain_modules:
                    return "0"
                return next(iter(self.domain_modules.keys()))
            dom = dom.view(-1)[0].item()
        dom = str(dom)
        if dom in self.domain_modules:
            return dom
        # fallback to first available domain
        return next(iter(self.domain_modules.keys()))

    def forward(self, data: properties.Type) -> properties.Type:
        return self.scale(data, force_process=False)

    def scale(self, data: properties.Type, force_process: bool = False) -> properties.Type:
        if self.training and not force_process:
            return data.copy()
        dom = self._get_domain(data)
        return self.domain_modules[dom].scale(data, force_process=force_process)

    def unscale(self, data: properties.Type, force_process: bool = False) -> properties.Type:
        if not self.training and not force_process:
            return data.copy()
        dom = self._get_domain(data)
        return self.domain_modules[dom].unscale(data, force_process=force_process)


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
        scales_keys: Optional[List[str]] = None,
        shifts_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self.scales_keys = scales_keys if scales_keys is not None else ["atomic_energy"]
        self.shifts_keys = shifts_keys if shifts_keys is not None else ["atomic_energy"]
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

    def setup_from_datamodule(self, _datamodule):
        # no legacy support
        raise NotImplementedError("Use setup_from_context instead.")

    def __repr__(self):
        return f"{self.__class__.__name__}(scales_keys={self.scales_keys}, shifts_keys={self.shifts_keys})"
