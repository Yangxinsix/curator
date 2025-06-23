# The code is copied from https://github.com/ACEsuit/mace/blob/mace-mliap/mace/calculators/lammps_mliap_mace.py with modifications

import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Dict, Tuple, Optional, List, Union
from curator.model.base import NeuralNetworkPotential, LitNNP

import torch
from ase.data import chemical_symbols

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
        self.config = CURATORLammpsConfig()
        self.element_types = element_types or model.representation.species
        self.num_species = len(self.element_types)
        self.rcutfac = 0.5 * float(model.representation.cutoff)
        self.ndescriptors = 1
        self.nparams = 1
        self.dtype = torch.get_default_dtype()
        self.device = "cpu"
        self.initialized = False
        self.step = 0

        self.model = model['model'] if model.__class__.__name__ == 'LitNNP' else model
        self._convert_model(model)
        
    @staticmethod
    def _convert_model(model):
        model.model_outputs = ['atomic_energy', 'edge_forces']

        # output atomic energy
        for spec in model.representation.readout.output_specs:
            if spec.key == 'energy':
                spec.per_atom = True
                spec.per_atom_key = 'atomic_energy'
        # output edge forces
        model.output_modules.gradient_output.compute_edge_forces = True
        model.output_modules.gradient_output.compute_edge_forces_only = True

    def _initialize_device(self, data):
        using_kokkos = "kokkos" in data.__class__.__module__.lower()

        if using_kokkos and not self.config.force_cpu:
            device = torch.as_tensor(data.elems).device
            if device.type == "cpu" and not self.config.allow_cpu:
                raise ValueError(
                    "GPU requested but tensor is on CPU. Set CURATOR_ALLOW_CPU=true to allow CPU computation."
                )
        else:
            device = torch.device("cpu")

        self.device = device
        self.model = self.model.to(device)
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
            with timer("prepare_batch", enabled=self.config.debug_time):
                batch = self._prepare_batch(data)

            with timer("model_forward", enabled=self.config.debug_time):
                out = self.model(batch)
                atom_energies, pair_forces = out['atomic_energy'], out['edge_forces']

                if self.device.type != "cpu":
                    torch.cuda.synchronize()

            with timer("update_lammps", enabled=self.config.debug_time):
                self._update_lammps_data(data, atom_energies, pair_forces, natoms)

    def _prepare_batch(self, data):
        """Prepare the input batch for the CURATOR model."""
        
        return {
            "n_atoms": torch.as_tensor(data.nlocal, dtype=torch.int64).unsqueeze(0),
            "_n_pairs": torch.as_tensor(data.npairs, dtype=torch.int64).unsqueeze(0),
            "_edge_index": torch.stack(
                [
                    torch.as_tensor(data.pair_j, dtype=torch.int64).to(self.device),
                    torch.as_tensor(data.pair_i, dtype=torch.int64).to(self.device),
                ],
                dim=0,
            ).T,
            "_edge_difference": torch.as_tensor(data.rij).to(self.dtype).to(self.device),
            "atomic_numbers": torch.as_tensor(data.elems, dtype=torch.int64).to(self.device),
            "lammps_data": data,
            "n_local": data.nlocal,
            "n_ghost": data.ntotal - data.nlocal,
        }

    def _update_lammps_data(self, data, atom_energies, pair_forces, natoms):
        """Update LAMMPS data structures with computed energies and forces."""
        if self.dtype == torch.float32:
            pair_forces = pair_forces.double()
        eatoms = torch.as_tensor(data.eatoms)
        eatoms.copy_(atom_energies[:natoms])
        data.energy = torch.sum(atom_energies[:natoms])
        data.update_pair_forces_gpu(pair_forces)

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
