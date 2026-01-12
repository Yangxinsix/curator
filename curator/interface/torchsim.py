from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Union
import warnings

import torch

from curator.data import (
    AseDataReader,
    NeighborListTransform,
    TorchNeighborList,
    BatchNeighborList,
    Transform,
    properties,
)
from curator.layer import find_layer_by_name_recursive
from curator.model import EnsembleModel
from curator.utils import load_models
from curator.interface.plumed import Plumed
try:
    from torch_sim.models.interface import ModelInterface
    from torch_sim.state import SimState
    _HAS_TORCHSIM = True
except ImportError:  # optional dependency
    ModelInterface = object  # type: ignore
    SimState = Any  # type: ignore
    _HAS_TORCHSIM = False

try:
    from ase import Atoms
except Exception:  # pragma: no cover - optional dependency
    Atoms = None


TensorDict = Mapping[str, Any]


class CuratorTorchSimAdapter(ModelInterface):
    """
    Lightweight wrapper to call CURATOR models inside torchsim graphs.

    - Accepts either an `ase.Atoms` instance or a tensor dictionary with
      keys from `curator.data.properties` (positions, atomic_numbers, cell...).
    - Builds neighbor lists (TorchNeighborList) when requested.
    - Returns a plain tensor dictionary that can be consumed by torchsim nodes
      such as LambdaNode/PyTorchNode.
    """

    def __init__(
        self,
        model: Union[str, torch.nn.Module, Sequence[Union[str, torch.nn.Module]]],
        *,
        cutoff: Optional[float] = None,
        compute_neighbor_list: bool = True,
        transforms: Optional[Sequence[Transform]] = None,
        device: Optional[torch.device] = None,
        load_compiled: bool = True,
        load_weights_only: bool = False,
        return_cell_displacements: bool = False,
        outputs: Optional[Sequence[str]] = None,
        energy_scale: float = 1.0,
        forces_scale: float = 1.0,
        stress_scale: float = 1.0,
        plumed_bias: Optional[Plumed] = None,
        detach: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if not _HAS_TORCHSIM:
            raise ImportError("torch-sim is required for CuratorTorchSimAdapter. Install `torch-sim` to use this feature.")
        super().__init__()
        # TorchSim interface metadata
        resolved_device = self._resolve_device(device)
        models = load_models(
            model,
            device=resolved_device,
            load_compiled=load_compiled,
            load_weights_only=load_weights_only,
        )
        self.model = EnsembleModel(models) if len(models) > 1 else models[0]
        self.model.eval()
        self._device = next(self.model.parameters()).device
        self._dtype = dtype or next(self.model.parameters()).dtype or torch.get_default_dtype()
        self._compute_stress = outputs is None or properties.stress in (outputs or []) or properties.virial in (outputs or [])
        self._compute_forces = True

        self.detach = detach
        self.outputs = set(outputs) if outputs is not None else None
        self.compute_neighbor_list = compute_neighbor_list
        self.energy_scale = energy_scale
        self.forces_scale = forces_scale
        self.stress_scale = stress_scale
        self.plumed_bias = plumed_bias

        self.transforms = list(transforms) if transforms is not None else []
        self._move_transforms_to_device(self._device)
        self.cutoff = float(cutoff) if cutoff is not None else None
        if self.compute_neighbor_list:
            if self.cutoff is None:
                self.cutoff = find_layer_by_name_recursive(self.model, "cutoff")
            if self.cutoff is None:
                raise ValueError("cutoff must be provided or discoverable on the model.")
            for t in self.transforms:
                if hasattr(t, "requires_grad") and self._compute_forces:
                    t.requires_grad = True
            if not any(isinstance(t, NeighborListTransform) for t in self.transforms):
                # BatchNeighborList handles both single and multi-system inputs
                self.transforms.append(
                    BatchNeighborList(
                        cutoff=self.cutoff,
                        requires_grad=self._compute_forces,
                        return_distance=False,
                        neighbor_list=TorchNeighborList(
                            cutoff=self.cutoff,
                            requires_grad=self._compute_forces,
                            return_cell_displacements=return_cell_displacements,
                        ),
                    )
                )

        self._ase_reader = AseDataReader(
            cutoff=self.cutoff,
            compute_neighbor_list=self.compute_neighbor_list,
            transforms=list(self.transforms),
            return_cell_displacements=return_cell_displacements,
            default_dtype=self._dtype,
        )
        self._move_transforms_to_device(self._device)

    def forward(
        self,
        inputs: Union["Atoms", TensorDict, SimState],
        *,
        detach: Optional[bool] = None,
        to_cpu: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch = self._prepare_inputs(inputs)
        batch = self._to_device(batch, self.device)
        try:
            outputs = self.model(batch)
        except RuntimeError as exc:
            if "does not require grad" in str(exc):
                outputs = self._forward_with_manual_forces(batch)
            else:
                raise
        processed = self._postprocess(outputs, batch, detach=detach, to_cpu=to_cpu)
        self.last_outputs = processed
        return processed

    def __call__(self, inputs, **kwargs):
        return self.forward(inputs, **kwargs)

    def _forward_with_manual_forces(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Fallback path when the checkpoint's GradientOutput autograd fails (e.g., detached energy)."""
        data = batch.copy()
        # Run input modules
        for m in self.model.input_modules:
            data = m(data)
        # Representation
        data = self.model.representation(data)

        post_scale_modules = []
        # Apply output modules except GradientOutput; defer GlobalRescaleShift until forces exist.
        from curator.layer._grad_output import GradientOutput  # local import to avoid cycle
        from curator.layer._rescale import GlobalRescaleShift

        for m in self.model.output_modules:
            if isinstance(m, GradientOutput):
                continue
            if isinstance(m, GlobalRescaleShift):
                post_scale_modules.append(m)
                continue
            data = m(data)

        # Manual forces via autograd on positions
        pos = data[properties.positions]
        if not pos.requires_grad:
            pos = pos.requires_grad_()
            data[properties.positions] = pos
        energy = data[properties.energy]
        forces = -torch.autograd.grad(energy.sum(), pos, create_graph=False, retain_graph=False)[0]
        data[properties.forces] = forces

        # Apply deferred scaling
        for m in post_scale_modules:
            data = m.scale(data, force_process=True)

        return data

    def _prepare_inputs(
        self, inputs: Union["Atoms", TensorDict, SimState]
    ) -> Dict[str, torch.Tensor]:
        if Atoms is not None and isinstance(inputs, Atoms):
            self._move_transforms_to_device(self.device)
            return self._ase_reader(inputs)
        if isinstance(inputs, SimState):
            return self._prepare_from_state(inputs)
        if isinstance(inputs, Mapping):
            return self._prepare_from_mapping(inputs)
        raise TypeError("inputs must be an ase.Atoms, torch_sim.SimState, or a mapping of tensors.")

    def _prepare_from_mapping(self, raw: TensorDict) -> Dict[str, torch.Tensor]:
        data: Dict[str, Any] = dict(raw)

        # Aliases for common field names
        alias_map = {
            properties.positions: ("pos", "r"),
            properties.Z: ("z", "atomic_numbers", "numbers"),
            properties.cell: ("cells",),
        }
        for canonical, aliases in alias_map.items():
            if canonical in data:
                continue
            for alias in aliases:
                if alias in data:
                    data[canonical] = data.pop(alias)
                    break

        if properties.positions not in data or properties.Z not in data:
            raise ValueError("positions and atomic_numbers are required to run the model.")

        pos = torch.as_tensor(data[properties.positions], dtype=self._dtype)
        z = torch.as_tensor(data[properties.Z], dtype=torch.long)
        if self._compute_forces:
            pos.requires_grad_()
        data[properties.positions] = pos
        data[properties.Z] = z
        if properties.cell in data:
            data[properties.cell] = torch.as_tensor(
                data[properties.cell], dtype=self._dtype, device=pos.device
            )

        if properties.n_atoms not in data:
            data[properties.n_atoms] = torch.tensor([pos.shape[0]], dtype=torch.long, device=pos.device)
        if properties.image_idx not in data:
            data[properties.image_idx] = torch.zeros(
                (data[properties.n_atoms].item(),), dtype=torch.long, device=pos.device
            )

        for transform in self.transforms:
            data = transform(data)

        return data

    def _prepare_from_state(self, state: SimState) -> Dict[str, torch.Tensor]:
        # TorchSim stores cell as column vectors; CURATOR expects row vectors.
        cell_tensor = state.cell
        if cell_tensor.ndim == 3:
            cell_row = cell_tensor.transpose(-2, -1)
        else:
            cell_row = cell_tensor.mT
        if cell_row.ndim == 3:
            cell_row = cell_row.reshape(-1, 3)
        mapping = {
            properties.positions: state.positions.requires_grad_() if self._compute_forces else state.positions,
            properties.Z: state.atomic_numbers,
            properties.cell: cell_row,
            properties.image_idx: state.system_idx,
            properties.n_atoms: state.n_atoms_per_system if hasattr(state, "n_atoms_per_system") else torch.tensor([state.n_atoms], device=state.device),
        }
        if hasattr(state, "masses"):
            mapping["masses"] = state.masses
        if hasattr(state, "charges"):
            mapping[properties.atomic_charge] = state.charges

        data = {k: v for k, v in mapping.items() if v is not None}
        for transform in self.transforms:
            data = transform(data)
        return data

    def _postprocess(
        self,
        outputs: Dict[str, Any],
        inputs: Dict[str, torch.Tensor],
        *,
        detach: Optional[bool],
        to_cpu: bool,
    ) -> Dict[str, torch.Tensor]:
        should_detach = self.detach if detach is None else detach
        result: Dict[str, torch.Tensor] = {}

        for key, value in outputs.items():
            if self.outputs is not None and key not in self.outputs:
                continue
            tensor = value
            if isinstance(tensor, torch.Tensor):
                if key == properties.energy:
                    tensor = tensor * self.energy_scale
                elif key == properties.forces:
                    tensor = tensor * self.forces_scale
                elif key == properties.stress:
                    tensor = tensor * self.stress_scale
                if should_detach:
                    tensor = tensor.detach()
                if to_cpu:
                    tensor = tensor.cpu()
            result[key] = tensor

        if (
            properties.virial in outputs
            and properties.stress not in result
            and (self.outputs is None or properties.stress in self.outputs)
        ):
            virial = outputs[properties.virial]
            if isinstance(virial, torch.Tensor) and properties.cell in inputs:
                volume = torch.det(inputs[properties.cell])
                stress = -(virial / volume).detach() if should_detach else -virial / volume
                result[properties.stress] = self._voigt6_to_full(stress, to_cpu=to_cpu)

        # Ensure shapes for TorchSim (energy: [n_systems], forces: [n_atoms, 3], stress: [n_systems, 3, 3])
        if properties.energy in result and isinstance(result[properties.energy], torch.Tensor):
            e = result[properties.energy]
            result[properties.energy] = e.view(-1)
        if properties.forces in result and isinstance(result[properties.forces], torch.Tensor):
            f = result[properties.forces]
            result[properties.forces] = f.view(f.shape[0], 3)
        if properties.stress in result and isinstance(result[properties.stress], torch.Tensor):
            s = result[properties.stress]
            if s.ndim == 2 and s.shape[-1] == 6:
                s = self._voigt6_to_full(s, to_cpu=to_cpu)
            if s.ndim == 2 and s.shape[0] == 9:
                s = s.view(1, 3, 3)
            result[properties.stress] = s
        if self.plumed_bias is not None:
            result = self.plumed_bias.apply_batch(result, inputs)
        return result

    @staticmethod
    def _to_device(data: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

    # --- TorchSim interface helpers ---
    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @staticmethod
    def _resolve_device(device_like: Optional[Union[str, torch.device]]) -> torch.device:
        if device_like is None:
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

        dev = torch.device(device_like)
        if dev.type == "cuda":
            try:
                torch.cuda.init()
                torch.cuda.device(dev)
            except Exception as exc:
                warnings.warn(f"CUDA device '{dev}' unavailable ({exc}); falling back to CPU.")
                dev = torch.device("cpu")
        return dev

    def _move_transforms_to_device(self, device: torch.device) -> None:
        for t in getattr(self, "transforms", []):
            if hasattr(t, "to"):
                try:
                    t.to(device)
                except Exception:
                    pass

    @staticmethod
    def _voigt6_to_full(tensor: torch.Tensor, to_cpu: bool = False) -> torch.Tensor:
        """
        Convert a (..., 6) Voigt-form tensor to (..., 3, 3).
        Order assumed: xx, yy, zz, yz, xz, xy (ASE convention).
        """
        if tensor.ndim == 1:
            tensor = tensor.view(1, -1)
        if tensor.shape[-1] != 6:
            return tensor.cpu() if to_cpu and isinstance(tensor, torch.Tensor) else tensor
        xx, yy, zz, yz, xz, xy = tensor.unbind(-1)
        full = torch.stack(
            [
                torch.stack([xx, xy, xz], dim=-1),
                torch.stack([xy, yy, yz], dim=-1),
                torch.stack([xz, yz, zz], dim=-1),
            ],
            dim=-2,
        )
        return full.cpu() if to_cpu else full


def build_torchsim_callable(
    adapter: CuratorTorchSimAdapter,
    *,
    output_keys: Optional[Sequence[str]] = None,
    to_cpu: bool = False,
) -> Any:
    """
    Helper to plug the adapter into torchsim nodes that expect a callable.

    Examples
    --------
    >>> adapter = CuratorTorchSimAdapter("model.pt", cutoff=5.0)
    >>> step_fn = build_torchsim_callable(adapter, output_keys=["energy", "forces"])
    >>> # In torchsim: LambdaNode(step_fn) or PyTorchNode(step_fn)
    """

    def _fn(tensors: TensorDict):
        outputs = adapter(tensors, to_cpu=to_cpu)
        if output_keys is None:
            return outputs
        return {k: outputs[k] for k in output_keys if k in outputs}

    return _fn
