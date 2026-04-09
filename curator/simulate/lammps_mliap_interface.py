# The code is copied from https://github.com/ACEsuit/mace/blob/mace-mliap/mace/calculators/lammps_mliap_mace.py with modifications

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Dict, Optional, List, Union
import numpy as np
from curator.data import properties
from curator.data._uncertainty import collect_uncertainty_outputs
from curator.layer import GradientOutput, GlobalRescaleShift, PairwiseDistance
from curator.model.base import NeuralNetworkPotential, LitNNP
from curator.model.ensemble import EnsembleModel

import torch
from ase.data import atomic_numbers

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
except ImportError:

    class MLIAPUnified:
        def __init__(self):
            pass

class CURATORLammpsConfig:
    """Configuration settings for CURATOR-LAMMPS integration."""

    def __init__(self):
        self.debug_time = self._get_env_bool("CURATOR_TIME", False)
        self.debug_profile = self._get_env_bool("CURATOR_PROFILE", False)
        self.debug_dump_dir = os.environ.get("CURATOR_MLIAP_DUMP_DIR", "").strip()
        self.profile_start_step = int(os.environ.get("CURATOR_PROFILE_START", "5"))
        self.profile_end_step = int(os.environ.get("CURATOR_PROFILE_END", "10"))
        self.allow_cpu = self._get_env_bool("CURATOR_ALLOW_CPU", False)
        self.force_cpu = self._get_env_bool("CURATOR_FORCE_CPU", False)

    @staticmethod
    def _get_env_bool(var_name: str, default: bool) -> bool:
        return os.environ.get(var_name, str(default)).lower() in (
            "true",
            "1",
            "t",
            "yes",
        )

@contextmanager
def timer(name: str, enabled: bool = True):
    """Context manager for timing code blocks."""
    if not enabled:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logging.info(f"Timer - {name}: {elapsed*1000:.3f} ms")

class LAMMPS_MLIAP(MLIAPUnified):
    """CURATOR integration for LAMMPS using the MLIAP interface."""
    def __init__(
            self, 
            model: Union[NeuralNetworkPotential, LitNNP],
            element_types: Optional[List[str]] = None,
            **kwargs,
        ):
        super().__init__()
        base_model = model.model if isinstance(model, LitNNP) else model
        reference_model = base_model.models[0] if isinstance(base_model, EnsembleModel) else base_model
        self.config = CURATORLammpsConfig()
        self.element_types = element_types or reference_model.representation.species
        self.num_species = len(self.element_types)
        self.rcutfac = 0.5 * float(reference_model.representation.cutoff)
        self.ndescriptors = 1
        self.nparams = 1
        self.dtype = torch.get_default_dtype()
        self.device = "cpu"
        self.initialized = False
        self.step = 0
        models_to_convert = list(base_model.models) if isinstance(base_model, EnsembleModel) else [base_model]
        for submodel in models_to_convert:
            self._convert_model(submodel)
        if isinstance(base_model, EnsembleModel):
            base_model.refresh_model_outputs()
        self.model = base_model
        (
            self.scalar_uncertainty_output_keys,
            self.per_atom_uncertainty_output_keys,
        ) = collect_uncertainty_outputs(base_model)
        # LAMMPS MLIAP provides integer element types (0..N-1 or 1..N) rather than atomic numbers.
        # Keep a type -> atomic-number table from pair_coeff element order.
        self._elem_type_to_z = torch.tensor(
            [atomic_numbers[s] for s in self.element_types], dtype=torch.int64
        )

    @staticmethod
    def _local_rank() -> int:
        for name in (
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MV2_COMM_WORLD_LOCAL_RANK",
            "SLURM_LOCALID",
            "LOCAL_RANK",
            "PMI_LOCAL_RANK",
        ):
            value = os.environ.get(name)
            if value is not None:
                try:
                    return int(value)
                except ValueError:
                    continue
        return 0

    @staticmethod
    def _is_cuda_array(value) -> bool:
        if isinstance(value, torch.Tensor):
            return value.is_cuda
        return hasattr(value, "__cuda_array_interface__")

    @staticmethod
    def _to_torch(value, dtype=None, device=None):
        if isinstance(value, torch.Tensor):
            tensor = value
        elif hasattr(value, "__dlpack__"):
            tensor = torch.utils.dlpack.from_dlpack(value)
        else:
            tensor = torch.as_tensor(value)

        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        if device is not None and tensor.device != device:
            tensor = tensor.to(device=device)
        return tensor

    @staticmethod
    def _supports_ghost_exchange(data) -> bool:
        return hasattr(data, "forward_exchange") and hasattr(data, "reverse_exchange")

    def _maybe_dump_debug(self, name: str, payload: Dict, arrays: Optional[Dict[str, np.ndarray]] = None):
        dump_root = getattr(self.config, "debug_dump_dir", "")
        if not dump_root:
            return

        dump_dir = os.path.join(dump_root, f"rank{self._local_rank()}")
        os.makedirs(dump_dir, exist_ok=True)
        stem = os.path.join(dump_dir, f"step{self.step:03d}_{name}")
        with open(stem + ".json", "w") as f:
            json.dump(payload, f, indent=2)
        if arrays:
            np.savez_compressed(stem + ".npz", **arrays)

    def _map_element_types(self, elems) -> torch.Tensor:
        elems = self._to_torch(elems, dtype=torch.int64)
        if elems.numel() == 0:
            return elems

        emin = int(elems.min().item())
        emax = int(elems.max().item())
        mapping = self._elem_type_to_z.to(elems.device)
        if emin >= 0 and emax < self.num_species:
            return mapping[elems]
        if emin >= 1 and emax <= self.num_species:
            return mapping[elems - 1]
        return elems

    @staticmethod
    def _move_kernel_projectors_to_device(model: torch.nn.Module, device: torch.device) -> None:
        for module in model.modules():
            kernels = getattr(module, "kernels", None)
            if not kernels:
                continue
            for kernel in kernels:
                projector = getattr(kernel, "projector", None)
                if isinstance(projector, torch.nn.Module):
                    projector.to(device)
        
    @staticmethod
    def _retarget_rescale_module(module):
        if not all(hasattr(module, attr) for attr in ("heads", "scales", "shifts", "atomic_shifts")):
            return

        for i, head in enumerate(module.heads):
            if head.key == properties.energy:
                new_key = properties.atomic_energy
                head.key = new_key
                if hasattr(head, "atomwise_key"):
                    head.atomwise_key = properties.atomic_energy
                if hasattr(head, "atomwise_shift"):
                    head.atomwise_shift = True
                if hasattr(head, "atomwise_normalization"):
                    head.atomwise_normalization = False
            elif head.key == properties.forces:
                new_key = properties.edge_forces
                head.key = new_key
                if hasattr(head, "atomwise_key"):
                    head.atomwise_key = properties.edge_forces
            else:
                continue

            module.scales[i].key = new_key
            module.shifts[i].key = new_key
            module.atomic_shifts[i].key = new_key

            if new_key == properties.atomic_energy and hasattr(module.shifts[i], "atomwise_shift"):
                module.shifts[i].atomwise_shift = True
            if new_key == properties.atomic_energy and hasattr(module.shifts[i], "atomwise_normalization"):
                module.shifts[i].atomwise_normalization = False

    @staticmethod
    def _convert_model(model):
        keep_outputs = [
            key
            for key in getattr(model, "model_outputs", [])
            if key not in {
                properties.energy,
                properties.forces,
                properties.atomic_energy,
                properties.edge_forces,
            }
        ]
        model.model_outputs = [properties.atomic_energy, properties.edge_forces] + keep_outputs

        # output atomic energy
        readout = model.representation.readout
        readout_modules = []
        if hasattr(readout, "model_outputs") and hasattr(readout, "per_atom_flags"):
            readout_modules = [readout]
        elif hasattr(readout, "domain_modules"):
            readout_modules = [
                module
                for module in readout.domain_modules.values()
                if hasattr(module, "model_outputs") and hasattr(module, "per_atom_flags")
            ]

        for module in readout_modules:
            for i, key in enumerate(module.model_outputs):
                if key == properties.energy:
                    module.per_atom_flags[i] = True
                    module.per_atom_keys[i] = properties.atomic_energy

        # output edge forces
        for m in model.output_modules:
            if isinstance(m, GradientOutput):
                m.compute_edge_forces = True
                m.compute_edge_forces_only = True
                m.model_outputs = [properties.forces, properties.edge_forces]
            elif isinstance(m, GlobalRescaleShift):
                m.atomwise_shift = True
                m.scale_keys = [properties.atomic_energy, properties.edge_forces]
                m.shift_keys = [properties.atomic_energy]
                m.model_outputs = [properties.atomic_energy, properties.edge_forces]
            elif hasattr(m, "domain_modules") and hasattr(m, "heads"):
                LAMMPS_MLIAP._retarget_rescale_module(m)
                for submodule in m.domain_modules.values():
                    LAMMPS_MLIAP._retarget_rescale_module(submodule)

        # LAMMPS unified interface already provides edge index and edge vectors.
        for m in model.input_modules:
            if isinstance(m, PairwiseDistance):
                m.compute_neighbor_list = False
                m.batch_nl = None
                m.compute_distance_from_R = False
                m.compute_forces = True
        
        model.eval()

    def _initialize_device(self, data):
        try:
            elems = data.elems
        except Exception as exc:
            raise RuntimeError(
                "Failed to inspect MLIAP data backend. "
                "If you are using KOKKOS GPU mode, make sure the Python side can access the returned arrays "
                "(for example by installing cupy or another DLPack-capable bridge)."
            ) from exc

        if self._is_cuda_array(elems) and not self.config.force_cpu:
            if not torch.cuda.is_available():
                if not self.config.allow_cpu:
                    raise ValueError(
                        "LAMMPS provided GPU-backed MLIAP data, but torch.cuda is unavailable. "
                        "Set CURATOR_ALLOW_CPU=true to force CPU execution."
                    )
                device = torch.device("cpu")
            else:
                local_rank = self._local_rank() % max(torch.cuda.device_count(), 1)
                torch.cuda.set_device(local_rank)
                device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")

        self.device = device
        self.model = self.model.to(device)
        self._move_kernel_projectors_to_device(self.model, device)
        logging.info(f"CURATOR model initialized on device: {device}")
        self.initialized = True

    def compute_forces(self, data):
        natoms = data.nlocal
        npairs = data.npairs

        if not self.initialized:
            self._initialize_device(data)

        self.step += 1
        self._manage_profiling()

        if natoms == 0 or npairs <= 1:
            return

        with timer("total_step", enabled=self.config.debug_time):
            use_ghost_exchange = self._supports_ghost_exchange(data)
            n_ghost = int(getattr(data, "ntotal", natoms) - natoms)

            with timer("prepare_batch", enabled=self.config.debug_time):
                batch = self._prepare_batch(data, use_ghost_exchange)

            with timer("model_forward", enabled=self.config.debug_time):
                out = self._forward_model(
                    batch=batch,
                    data=data,
                    natoms=natoms,
                    n_ghost=n_ghost,
                    use_ghost_exchange=use_ghost_exchange,
                )
                atom_energies, pair_forces = out['atomic_energy'], out['edge_forces']

                if self.device.type != "cpu":
                    torch.cuda.synchronize()

                self._maybe_dump_debug(
                    "outputs",
                    {
                        "use_ghost_exchange": bool(use_ghost_exchange),
                        "atomic_energy_shape": list(atom_energies.shape),
                        "edge_forces_shape": list(pair_forces.shape),
                        "local_energy_sum": float(torch.sum(atom_energies[:natoms]).item()),
                        "scalar_uncertainty_keys": list(self.scalar_uncertainty_output_keys),
                        "per_atom_uncertainty_keys": list(self.per_atom_uncertainty_output_keys),
                    },
                    arrays={
                        "atomic_energy": atom_energies.detach().cpu().numpy(),
                        "edge_forces": pair_forces.detach().cpu().numpy(),
                    },
                )

            with timer("update_lammps", enabled=self.config.debug_time):
                self._write_core_outputs(data, out, natoms)
                self._write_uncertainties(data, out, natoms)

    def _call_model(self, model, batch, data, natoms: int, n_ghost: int, use_ghost_exchange: bool):
        if use_ghost_exchange:
            return model.forward_with_lammps(
                batch,
                lammps_data=data,
                n_local=natoms,
                n_ghost=n_ghost,
            )
        return model(batch)

    def _forward_model(self, batch, data, natoms: int, n_ghost: int, use_ghost_exchange: bool):
        return self._call_model(self.model, batch, data, natoms, n_ghost, use_ghost_exchange)

    @staticmethod
    def _to_scalar_uncertainty(value) -> Optional[float]:
        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            return float(value.detach().reshape(-1)[0].item())
        if isinstance(value, (float, int)):
            return float(value)
        return None

    def _prepare_batch(self, data, use_ghost_exchange: bool):
        """Prepare the input batch for the CURATOR model."""
        nlocal = int(data.nlocal)
        if use_ghost_exchange:
            elems = self._map_element_types(data.elems[:nlocal]).to(self.device)
            pair_i = self._to_torch(data.pair_i, dtype=torch.int64, device=self.device)
            pair_j = self._to_torch(data.pair_j, dtype=torch.int64, device=self.device)
            edge_index = torch.stack([pair_i, pair_j], dim=0).T
        else:
            elems = self._map_element_types(data.elems[:nlocal]).to(self.device)
            pair_i = self._to_torch(data.pair_i, dtype=torch.int64)
            pair_j = self._to_torch(data.pair_j, dtype=torch.int64)
            tags = self._to_torch(data.tags, dtype=torch.int64)

            local_tags = tags[:nlocal]
            owned_index_by_tag = {int(tag): idx for idx, tag in enumerate(local_tags.cpu().tolist())}
            mapped_j = []
            for j in pair_j.cpu().tolist():
                mapped = owned_index_by_tag.get(int(tags[j].item()))
                if mapped is None:
                    raise ValueError(
                        "MLIAP batch contains ghost neighbors whose tags are not owned locally; "
                        "the current non-KOKKOS mliap fallback only supports local-owned graph indexing."
                    )
                mapped_j.append(mapped)

            edge_index = torch.stack(
                [
                    pair_i.to(self.device),
                    torch.as_tensor(mapped_j, dtype=torch.int64, device=self.device),
                ],
                dim=0,
            ).T

        arrays = {
            "raw_pair_i": self._to_torch(data.pair_i, dtype=torch.int64).cpu().numpy(),
            "raw_pair_j": self._to_torch(data.pair_j, dtype=torch.int64).cpu().numpy(),
            "raw_elems": self._to_torch(data.elems, dtype=torch.int64).cpu().numpy(),
            "edge_index": edge_index.detach().cpu().numpy(),
            "edge_diff": self._to_torch(data.rij, dtype=self.dtype).cpu().numpy(),
            "atomic_numbers": elems.detach().cpu().numpy(),
        }
        if hasattr(data, "tags"):
            arrays["raw_tags"] = self._to_torch(data.tags, dtype=torch.int64).cpu().numpy()

        self._maybe_dump_debug(
            "batch",
            {
                "use_ghost_exchange": bool(use_ghost_exchange),
                "device": str(self.device),
                "nlocal": int(nlocal),
                "ntotal": int(getattr(data, "ntotal", nlocal)),
                "npairs": int(data.npairs),
                "edge_i_min": int(edge_index[:, 0].min().item()) if edge_index.numel() else 0,
                "edge_i_max": int(edge_index[:, 0].max().item()) if edge_index.numel() else 0,
                "edge_j_min": int(edge_index[:, 1].min().item()) if edge_index.numel() else 0,
                "edge_j_max": int(edge_index[:, 1].max().item()) if edge_index.numel() else 0,
                "edge_i_ge_nlocal": int((edge_index[:, 0] >= nlocal).sum().item()) if edge_index.numel() else 0,
                "edge_j_ge_nlocal": int((edge_index[:, 1] >= nlocal).sum().item()) if edge_index.numel() else 0,
                "has_tags": bool(hasattr(data, "tags")),
            },
            arrays=arrays,
        )

        return {
            "n_atoms": torch.as_tensor(nlocal, dtype=torch.int64, device=self.device).unsqueeze(0),
            "_n_pairs": torch.as_tensor(data.npairs, dtype=torch.int64, device=self.device).unsqueeze(0),
            "_edge_index": edge_index,
            "_edge_difference": self._to_torch(data.rij, dtype=self.dtype, device=self.device),
            "atomic_numbers": elems,
        }

    def _write_core_outputs(self, data, output, natoms):
        """Write energies and forces back to the MLIAP data object."""
        atom_energies = output[properties.atomic_energy]
        pair_forces = output[properties.edge_forces]
        if self.dtype == torch.float32:
            pair_forces = pair_forces.double()

        atom_energies_local = atom_energies[:natoms].detach()
        if self.device.type == "cuda":
            pair_forces_out = pair_forces.detach()
        else:
            pair_forces_out = pair_forces.detach().cpu().double().numpy()

        try:
            if self.device.type == "cuda":
                data.eatoms = atom_energies_local
            else:
                data.eatoms = atom_energies_local.cpu().double().numpy()
        except Exception:
            logging.debug("Skipping per-atom energy writeback for current MLIAP backend.")

        data.energy = float(torch.sum(atom_energies[:natoms]).item())
        data.update_pair_forces(pair_forces_out)

    def _write_uncertainties(self, data, output, natoms):
        if hasattr(data, "clear_uncertainties"):
            data.clear_uncertainties()
            for key in self.scalar_uncertainty_output_keys:
                if key not in output:
                    continue
                value = self._to_scalar_uncertainty(output[key])
                if value is not None:
                    data.set_uncertainty(key, value)
        if hasattr(data, "clear_uncertainty_arrays"):
            data.clear_uncertainty_arrays()
            for key in self.per_atom_uncertainty_output_keys:
                if key not in output:
                    continue
                value = output[key]
                if not torch.is_tensor(value):
                    continue
                data.set_uncertainty_array(
                    key,
                    value[:natoms].detach().cpu().double().numpy(),
                )

    def _manage_profiling(self):
        if not self.config.debug_profile:
            return

        if self.step == self.config.profile_start_step:
            logging.info(f"Starting CUDA profiler at step {self.step}")
            torch.cuda.profiler.start()

        if self.step == self.config.profile_end_step:
            logging.info(f"Stopping CUDA profiler at step {self.step}")
            torch.cuda.profiler.stop()
            logging.info("Profiling complete. Exiting.")
            sys.exit()

    def compute_descriptors(self, data):
        pass

    def compute_gradients(self, data):
        pass
