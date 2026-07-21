from typing import Optional, Dict, Union, List, Any
import torch
from torch import nn
from curator.data import properties
from curator.data.datamodule import DataContext
from curator.data.properties import HeadConfig, HEAD_PRESETS, resolve_heads, normalize_head_flag

try:
    from torch_scatter import scatter_add, scatter_mean
except ImportError:  # pragma: no cover
    from curator.utils import scatter_add, scatter_mean

from ase.data import atomic_numbers, chemical_symbols


def _effective_rescale_heads(dm) -> List[HeadConfig]:
    specs: List[Any] = [properties.energy]
    if getattr(dm, "scale_forces", False):
        specs.append(properties.forces)
    specs.extend(getattr(dm, "rescale_shift_heads", None) or [])

    by_key: Dict[str, HeadConfig] = {}
    for spec in specs:
        head = resolve_heads([spec])[0]
        by_key[head.key] = head
    return list(by_key.values())


# --------------------------------------------------------------------------- #
# Basic transforms
# --------------------------------------------------------------------------- #
class ScaleTransform(nn.Module):
    def __init__(
        self,
        key: str,
        scale: Union[float, torch.Tensor],
        trainable: bool = False,
        data_key: Optional[str] = None,
    ):
        super().__init__()
        scale = scale if torch.is_tensor(scale) else torch.tensor([scale], dtype=torch.float)
        if trainable:
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer("scale", scale)
        self.key = key
        self.data_key = data_key or key

    def forward(self, data: properties.Type) -> properties.Type:
        target_key = self.data_key if self.data_key in data else self.key
        if target_key in data:
            data[target_key] = data[target_key] * self.scale
        return data


class ShiftTransform(nn.Module):
    """Simple shift: add scalar shift to a property."""

    def __init__(
        self,
        key: str,
        shift: Union[float, torch.Tensor],
        trainable: bool = False,
        data_key: Optional[str] = None,
    ):
        super().__init__()
        shift = shift if torch.is_tensor(shift) else torch.tensor([shift], dtype=torch.float)
        if trainable:
            self.shift = nn.Parameter(shift)
        else:
            self.register_buffer("shift", shift)
        self.key = key
        self.data_key = data_key or key

    def forward(self, data: properties.Type) -> properties.Type:
        target_key = self.data_key if self.data_key in data else self.key
        if target_key not in data:
            return data
        data[target_key] = data[target_key] + self.shift
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
        data_key: Optional[str] = None,
        atomwise_data_key: bool = False,
    ):
        super().__init__()
        shift = shift if torch.is_tensor(shift) else torch.tensor([shift], dtype=torch.float)
        if trainable:
            self.shift = nn.Parameter(shift)
        else:
            self.register_buffer("shift", shift)
        self.key = key
        self.data_key = data_key or key
        self.atomwise_shift = atomwise_shift
        self.atomwise_normalization = atomwise_normalization
        self.atomwise_data_key = atomwise_data_key

    def _resolve_key(self, data: properties.Type) -> str:
        return self.data_key if self.data_key in data else self.key

    def _use_atomwise_mode(self, target_key: str) -> bool:
        return self.atomwise_data_key and target_key == self.data_key and self.data_key != self.key

    def compute_shift(self, data: properties.Type, target_key: Optional[str] = None) -> torch.Tensor:
        target_key = self._resolve_key(data) if target_key is None else target_key
        if self.atomwise_shift or self._use_atomwise_mode(target_key):
            return self.shift
        if self.atomwise_normalization:
            return data[properties.n_atoms] * self.shift
        return self.shift

    def forward(self, data: properties.Type) -> properties.Type:
        target_key = self._resolve_key(data)
        if target_key not in data:
            return data
        if self.atomwise_shift or self._use_atomwise_mode(target_key):
            data[target_key] = data[target_key] + self.shift
            return data

        s = self.compute_shift(data, target_key=target_key)
        data[target_key] = data[target_key] + s
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
        data_key: Optional[str] = None,
        atomwise_data_key: bool = False,
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
        self.data_key = data_key or key
        self.atomwise_shift = atomwise_shift
        self.atomwise_data_key = atomwise_data_key

    def _resolve_key(self, data: properties.Type) -> str:
        return self.data_key if self.data_key in data else self.key

    def _use_atomwise_mode(self, target_key: str) -> bool:
        return self.atomwise_shift or (
            self.atomwise_data_key and target_key == self.data_key and self.data_key != self.key
        )

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
        target_key = self._resolve_key(data)
        if not self.enabled or target_key not in data:
            return data
        per_atom = self.values[data[properties.Z]]
        while per_atom.dim() < data[target_key].dim():
            per_atom = per_atom.unsqueeze(-1)
        if self._use_atomwise_mode(target_key):
            data[target_key] = data[target_key] + sign * per_atom
            return data

        # structure-level: sum over atoms
        shift_term = scatter_add(per_atom, data[properties.image_idx])
        data[target_key] = data[target_key] + sign * shift_term
        return data


class PerSpeciesScale(nn.Module):
    """
    Per-species multiplicative scaling for atomwise properties.
    """

    def __init__(
        self,
        key: str,
        values: Optional[Dict[int, float]],
        data_key: Optional[str] = None,
    ):
        super().__init__()
        values_dict = torch.ones((119,), dtype=torch.float)
        if values is not None:
            for k, v in values.items():
                idx = atomic_numbers[k] if isinstance(k, str) else k
                values_dict[idx] = v
        self.register_buffer("values", values_dict)
        self.register_buffer("enabled", torch.tensor(values is not None))
        self.key = key
        self.data_key = data_key or key

    def load_values(self, values: Dict[int, float]):
        self.values.fill_(1.0)
        for k, v in values.items():
            idx = atomic_numbers[k] if isinstance(k, str) else k
            self.values[idx] = v
        self.enabled.copy_(torch.tensor(True))

    def forward(self, data: properties.Type) -> properties.Type:
        target_key = self.data_key if self.data_key in data else self.key
        if not self.enabled or target_key not in data:
            return data
        if self.data_key != self.key and target_key == self.key:
            return data
        factors = self.values[data[properties.Z]]
        while factors.dim() < data[target_key].dim():
            factors = factors.unsqueeze(-1)
        data[target_key] = data[target_key] * factors
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
        # ensure heads are HeadConfig instances (keep explicit HeadConfig overrides)
        self.heads = resolve_heads(heads)
        self._initialize_transforms(scale_trainable=scale_trainable, shift_trainable=shift_trainable)

    def _initialize_transforms(
        self,
        *,
        scale_trainable: bool = False,
        shift_trainable: bool = False,
    ) -> None:
        # build per-head transforms
        scales = []
        shifts = []
        per_species_scales = []
        per_species_shifts = []
        self._atomwise_output_keys: Dict[str, bool] = {}
        for h in self.heads:
            self._atomwise_output_keys[h.key] = bool(h.reduction is None)
            target_key = self._preferred_data_key(h)
            atomwise_data_key = target_key != h.key
            # scale
            s_val = 1.0
            if isinstance(h.scale_by, (int, float)) and not isinstance(h.scale_by, bool):
                s_val = h.scale_by
            scales.append(
                ScaleTransform(
                    h.key,
                    s_val,
                    trainable=scale_trainable,
                    data_key=target_key,
                )
            )
            # shift
            shift_mode = normalize_head_flag(h.shift_by)
            sh_val = 0.0
            if isinstance(h.shift_by, (int, float)) and not isinstance(h.shift_by, bool):
                sh_val = h.shift_by
            if shift_mode is not None and (h.atomwise_shift or h.atomwise_normalization):
                shifts.append(
                    AtomwiseShift(
                        h.key,
                        sh_val,
                        atomwise_shift=h.atomwise_shift,
                        atomwise_normalization=h.atomwise_normalization,
                        trainable=shift_trainable,
                        data_key=target_key,
                        atomwise_data_key=atomwise_data_key,
                    )
                )
            else:
                shifts.append(
                    ShiftTransform(
                        h.key,
                        sh_val,
                        trainable=shift_trainable,
                        data_key=target_key,
                    )
                )

            init_scale_values = None
            if isinstance(h.per_species_scale, dict):
                init_scale_values = h.per_species_scale
            per_species_scales.append(
                PerSpeciesScale(
                    h.key,
                    values=init_scale_values,
                    data_key=target_key,
                )
            )

            # per-species shift (e.g., atomic energies), optional
            # per-species shift: create with initial values if provided; else None (enable via context if "auto")
            init_values = None
            if isinstance(h.per_species_shift, dict):
                init_values = h.per_species_shift
            per_species_shifts.append(
                PerSpeciesShift(
                    h.key,
                    values=init_values,
                    atomwise_shift=h.atomwise_shift,
                    data_key=target_key,
                    atomwise_data_key=atomwise_data_key,
                )
            )

        self.scales = nn.ModuleList(scales)
        self.shifts = nn.ModuleList(shifts)
        self.atomic_scales = nn.ModuleList(per_species_scales)
        self.atomic_shifts = nn.ModuleList(per_species_shifts)
        self._configure_sync_reduced_outputs()
        self._initialized = not any(
            normalize_head_flag(h.scale_by) in ("default", "rms")
            or normalize_head_flag(h.shift_by) == "default"
            for h in self.heads
        )

    @torch.jit.unused
    def _configure_sync_reduced_outputs(self) -> None:
        head_keys: List[str] = []
        atomwise_keys: List[str] = []
        reductions: List[str] = []
        for head in self.heads:
            atomwise_key = getattr(head, "atomwise_key", None)
            reduction = getattr(head, "reduction", None)
            if not head.is_atomwise or atomwise_key is None or atomwise_key == head.key:
                continue
            if reduction != "sum" and reduction != "mean":
                continue
            head_keys.append(head.key)
            atomwise_keys.append(atomwise_key)
            reductions.append(reduction)
        self._sync_head_keys = torch.jit.annotate(List[str], head_keys)
        self._sync_atomwise_keys = torch.jit.annotate(List[str], atomwise_keys)
        self._sync_reductions = torch.jit.annotate(List[str], reductions)

    @staticmethod
    def _format_scalar(val: Any):
        if torch.is_tensor(val):
            v = val.detach().cpu().reshape(-1).tolist()
            val = v[0] if len(v) == 1 else v
        return val

    @staticmethod
    def _preferred_data_key(head: HeadConfig) -> str:
        if head.is_atomwise and head.atomwise_key:
            return head.atomwise_key
        return head.key

    @staticmethod
    def _ensure_image_idx(data: properties.Type, key: str) -> bool:
        if properties.image_idx in data:
            return True
        if properties.n_atoms not in data:
            return False
        n_atoms = data[properties.n_atoms]
        if n_atoms.numel() == 0:
            return False
        n_atoms_val = int(n_atoms.reshape(-1)[0].item())
        device = n_atoms.device
        if key in data:
            device = data[key].device
        data[properties.image_idx] = torch.zeros(n_atoms_val, dtype=torch.long, device=device)
        return True

    def _sync_reduced_outputs(self, data: properties.Type) -> properties.Type:
        for i in range(len(self._sync_head_keys)):
            head_key = self._sync_head_keys[i]
            atomwise_key = self._sync_atomwise_keys[i]
            reduction = self._sync_reductions[i]
            if atomwise_key not in data:
                continue
            if not self._ensure_image_idx(data, atomwise_key):
                continue
            if reduction == "sum":
                data[head_key] = scatter_add(data[atomwise_key], data[properties.image_idx], dim=0)
            else:
                data[head_key] = scatter_mean(data[atomwise_key], data[properties.image_idx], dim=0)
        return data

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
        for sc in self.atomic_scales:
            data = sc(data)
        for sh in self.atomic_shifts:
            data = sh(data)
        for sh in self.shifts:
            data = sh(data)
        return self._sync_reduced_outputs(data)

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
            target_key = sh.data_key if getattr(sh, "data_key", sh.key) in data else sh.key
            if target_key not in data:
                continue
            if isinstance(sh, AtomwiseShift):
                if sh.atomwise_shift or sh._use_atomwise_mode(target_key):
                    data[target_key] = data[target_key] - sh.shift
                else:
                    s = sh.compute_shift(data, target_key=target_key)
                    data[target_key] = data[target_key] - s
            else:
                data[target_key] = data[target_key] - sh.shift

        for sh in self.atomic_shifts:
            data = sh.apply(data, sign=-1.0)

        for sc in self.atomic_scales:
            target_key = sc.data_key if getattr(sc, "data_key", sc.key) in data else sc.key
            if not sc.enabled or target_key not in data:
                continue
            if getattr(sc, "data_key", sc.key) != sc.key and target_key == sc.key:
                continue
            factors = sc.values[data[properties.Z]]
            while factors.dim() < data[target_key].dim():
                factors = factors.unsqueeze(-1)
            data[target_key] = data[target_key] / factors

        # undo scale
        for sc in self.scales:
            target_key = sc.data_key if getattr(sc, "data_key", sc.key) in data else sc.key
            if target_key in data:
                data[target_key] = data[target_key] / sc.scale

        return self._sync_reduced_outputs(data)

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

            if isinstance(head.per_species_scale, dict):
                self.atomic_scales[i].load_values(head.per_species_scale)
            elif isinstance(head.per_species_scale, str) and head.per_species_scale == "auto":
                atomic_scale_values = ctx.head_species_scale.get(head.key, None)
                if atomic_scale_values is not None:
                    self.atomic_scales[i].load_values(atomic_scale_values)

            if isinstance(head.per_species_shift, dict):
                self.atomic_shifts[i].load_values(head.per_species_shift)
            elif isinstance(head.per_species_shift, str) and head.per_species_shift == "auto":
                atomic_values = ctx.head_species_shift.get(head.key, None)
                if atomic_values is not None:
                    self.atomic_shifts[i].load_values(atomic_values)
        self._initialized = True

    def setup_from_datamodule(self, dm):
        if hasattr(dm, "domain_modules"):
            return
        try:
            scale_trainable = any(isinstance(sc.scale, nn.Parameter) for sc in getattr(self, "scales", []))
            shift_trainable = any(
                hasattr(sh, "shift") and isinstance(sh.shift, nn.Parameter)
                for sh in getattr(self, "shifts", [])
            )
            self.heads = _effective_rescale_heads(dm)
            self._initialize_transforms(
                scale_trainable=scale_trainable,
                shift_trainable=shift_trainable,
            )
            ctx = dm.build_context(self.heads)
        except Exception:
            return
        return self.setup_from_context(ctx)

    # inspection helpers for exporter / debugging
    @property
    @torch.jit.unused
    def scale_by(self) -> List[Any]:
        return [self._format_scalar(sc.scale) for sc in self.scales]

    @property
    @torch.jit.unused
    def shift_by(self) -> List[Any]:
        return [self._format_scalar(sh.shift) if hasattr(sh, "shift") else 0.0 for sh in self.shifts]

    @property
    @torch.jit.unused
    def atomic_energies(self) -> Optional[torch.Tensor]:
        for i, h in enumerate(self.heads):
            if h.key == properties.energy:
                return self.atomic_shifts[i].values
        return None

    @property
    @torch.jit.unused
    def per_species_shifts(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for shift in self.atomic_shifts:
            enabled = shift.enabled
            if torch.is_tensor(enabled) and not bool(enabled.item()):
                continue
            values = shift.values.detach().cpu()
            nz = torch.nonzero(values, as_tuple=True)[0]
            mapping = {}
            for z in nz.tolist():
                if z == 0:
                    continue
                mapping[chemical_symbols[z]] = float(values[z].item())
            if mapping:
                out[shift.key] = mapping
        return out

    @property
    @torch.jit.unused
    def per_species_scales(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for scale in self.atomic_scales:
            enabled = scale.enabled
            if torch.is_tensor(enabled) and not bool(enabled.item()):
                continue
            values = scale.values.detach().cpu()
            nz = torch.nonzero(values != 1.0, as_tuple=True)[0]
            mapping = {}
            for z in nz.tolist():
                if z == 0:
                    continue
                mapping[chemical_symbols[z]] = float(values[z].item())
            if mapping:
                out[scale.key] = mapping
        return out

    @torch.jit.unused
    def __repr__(self):
        scales = [float(f"{x:.3g}") if isinstance(x, (int, float)) else x for x in self.scale_by]
        shifts = [float(f"{x:.3g}") if isinstance(x, (int, float)) else x for x in self.shift_by]
        return f"{self.__class__.__name__}(heads={[h.key for h in self.heads]}, scale={scales}, shift_by={shifts})"


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
            merged_heads: Dict[str, HeadConfig] = {}
            for domain_name, domain_dm in dm.domain_modules.items():
                dom_id = str(getattr(dm, "domain_to_id", {}).get(domain_name, domain_name))
                heads = _effective_rescale_heads(domain_dm)
                grs = GlobalRescaleShift(heads=heads)
                grs.setup_from_datamodule(domain_dm)
                self.domain_modules[str(dom_id)] = grs
                for head in heads:
                    merged_heads[head.key] = head
            self.heads = list(merged_heads.values()) or [HEAD_PRESETS["energy"]]
        else:
            heads = _effective_rescale_heads(dm)
            grs = GlobalRescaleShift(heads=heads)
            grs.setup_from_datamodule(dm)
            self.domain_modules["0"] = grs
            self.heads = heads

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
        # Converted official checkpoints can arrive with per-species scales/shifts
        # already materialized. In that case initialization is complete and this
        # hook should be a no-op during model.initialize_modules().
        if self._initialized:
            return
        # There is currently no datamodule-driven auto-fit path for this module.
        return

    def __repr__(self):
        return f"{self.__class__.__name__}(scales_keys={self.scales_keys}, shifts_keys={self.shifts_keys})"
