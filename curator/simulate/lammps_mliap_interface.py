# The code is copied from https://github.com/ACEsuit/mace/blob/mace-mliap/mace/calculators/lammps_mliap_mace.py with modifications

import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Dict, Tuple, Optional, List, Union
from curator.data import properties
from curator.layer import GradientOutput, GlobalRescaleShift
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

        self.model = model.model if isinstance(model, LitNNP) else model
        self._convert_model(self.model)
        
    @staticmethod
    def _convert_model(model):
        model.model_outputs = [properties.atomic_energy, properties.edge_forces]

        # output atomic energy
        for i, key in enumerate(model.representation.readout.model_outputs):
            if key == properties.energy:
                model.representation.readout.per_atom_flags[i] = True
                model.representation.readout.per_atom_keys[i] = properties.atomic_energy

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
        
        model.eval()

    def _initialize_device(self, data):
        using_kokkos = "kokkos" in data.__class__.__module__.lower()

        if using_kokkos and not self.config.force_cpu:
            device = torch.device("cuda")
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
        }, data, data.nlocal, data.ntotal - data.nlocal

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


class LAMMPS_MLIAP_QEQ(MLIAPUnified):
    """
    Curator-QEQ integration for LAMMPS using ML-IAP + kspace.
    
    This class handles the short-range ML forces and predicts atomic charges.
    The long-range Coulomb forces are computed by LAMMPS kspace (Ewald/PPPM).
    
    Architecture:
    - ML model predicts: atomic_energy, edge_forces, atomic_charge
    - Short-range forces: returned to LAMMPS via update_pair_forces_gpu()
    - Atomic charges: written to data.charges (requires extended MLIAP interface)
    - Long-range forces: computed by LAMMPS kspace->compute() using atom->q
    """
    
    def __init__(
            self, 
            model: Union[NeuralNetworkPotential, LitNNP],
            element_types: Optional[List[str]] = None,
            use_lammps_kspace: bool = True,
            charge_normalization: bool = True,
            total_charge: float = 0.0,
            **kwargs,
        ):
        """
        Initialize the QEQ MLIAP interface.
        
        Args:
            model: Curator MACE-QEQ model
            element_types: List of element symbols
            use_lammps_kspace: If True, only compute short-range forces and let 
                              LAMMPS kspace handle long-range. If False, compute 
                              full Ewald in Python (slower but doesn't need LAMMPS mods)
            charge_normalization: Normalize charges to ensure sum = total_charge
            total_charge: Target total charge for the system
        """
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
        
        # QEQ specific settings
        self.use_lammps_kspace = use_lammps_kspace
        self.charge_normalization = charge_normalization
        self.total_charge = total_charge

        self.model = model.model if isinstance(model, LitNNP) else model
        self._convert_model_qeq(self.model)
        
    @staticmethod
    def _convert_model_qeq(model):
        """
        Convert model to output for LAMMPS integration:
        
        Energy/Force components:
        - short_energy: pure ML short-range energy (no Ewald)
        - residual_energy: χ·q + η·q² electronegativity/hardness terms
        - edge_forces: per-pair forces (short-range ML only)
        - residual_forces: forces from residual energy (should be ~0)
        - atomic_charge: predicted atomic charges
        
        NOT included (computed by LAMMPS):
        - ewald_energy: real + recip + self (computed by coul/long + kspace)
        - ewald_forces: forces from Ewald energy
        
        Architecture:
        - LAMMPS receives: (short_energy + residual_energy), (edge_forces + residual_forces), charges
        - LAMMPS computes: Ewald energy/forces using charges via coul/long + kspace ewald/pppm
        """
        # Set model outputs - we need short energy, residual energy, and charges
        model.model_outputs = [
            properties.atomic_energy,      # Will be converted to short_energy
            properties.edge_forces,        # Short-range ML forces
            properties.atomic_charge,      # For LAMMPS kspace
            properties.residual_energy,    # χ·q + η·q² terms
            properties.residual_forces,    # Forces from residual (should be ~0)
        ]

        # Output atomic energy (will become short_energy before ChargeEquilibration adds Ewald)
        for i, key in enumerate(model.representation.readout.model_outputs):
            if key == properties.energy:
                model.representation.readout.per_atom_flags[i] = True
                model.representation.readout.per_atom_keys[i] = properties.atomic_energy

        # Output edge forces
        for m in model.output_modules:
            if isinstance(m, GradientOutput):
                m.compute_edge_forces = True
                m.compute_edge_forces_only = True
                m.model_outputs = [properties.forces, properties.edge_forces]
            elif isinstance(m, GlobalRescaleShift):
                m.atomwise_shift = True
                m.scale_keys = [properties.atomic_energy, properties.edge_forces]
                m.shift_keys = [properties.atomic_energy]
                m.model_outputs = [properties.atomic_energy, properties.edge_forces, properties.atomic_charge]
        
        model.eval()

    def _initialize_device(self, data):
        using_kokkos = "kokkos" in data.__class__.__module__.lower()

        if using_kokkos and not self.config.force_cpu:
            device = torch.device("cuda")
            if device.type == "cpu" and not self.config.allow_cpu:
                raise ValueError(
                    "GPU requested but tensor is on CPU. Set CURATOR_ALLOW_CPU=true to allow CPU computation."
                )
        else:
            device = torch.device("cpu")

        self.device = device
        self.model = self.model.to(device)
        logging.info(f"CURATOR-QEQ model initialized on device: {device}")
        logging.info(f"use_lammps_kspace: {self.use_lammps_kspace}")
        self.initialized = True

    def compute_forces(self, data):
        """
        Main compute function called by LAMMPS pair_mliap.
        
        Energy/Force flow:
        1. ML model computes: short_energy, edge_forces, atomic_charge
        2. QEQ layer adds: residual_energy (χ·q + η·q²), residual_forces (~0)
        3. We output to LAMMPS: (short_energy + residual_energy), edge_forces, charges
        4. LAMMPS kspace computes: Ewald real + recip + self using our charges
        
        Note: We do NOT include ewald_energy/ewald_forces since LAMMPS handles that.
        """
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
                
                # Short-range ML outputs
                atom_energies = out[properties.atomic_energy]
                pair_forces = out[properties.edge_forces]
                
                # QEQ outputs
                predicted_charges = out.get(properties.atomic_charge, None)
                residual_energy = out.get(properties.residual_energy, None)
                residual_forces = out.get(properties.residual_forces, None)
                
                # We explicitly DO NOT use:
                # - out[properties.ewald_energy] (LAMMPS kspace will compute this)
                # - out[properties.ewald_forces] (LAMMPS kspace will compute this)

                if self.device.type != "cpu":
                    torch.cuda.synchronize()

            with timer("update_lammps", enabled=self.config.debug_time):
                self._update_lammps_data(
                    data, 
                    atom_energies, 
                    pair_forces, 
                    predicted_charges, 
                    natoms,
                    residual_energy=residual_energy,
                    residual_forces=residual_forces,
                )

    def _prepare_batch(self, data):
        """Prepare the input batch for the CURATOR-QEQ model."""
        batch = {
            properties.n_atoms: torch.as_tensor(data.nlocal, dtype=torch.int64).unsqueeze(0),
            properties.n_pairs: torch.as_tensor(data.npairs, dtype=torch.int64).unsqueeze(0),
            properties.edge_idx: torch.stack(
                [
                    torch.as_tensor(data.pair_j, dtype=torch.int64).to(self.device),
                    torch.as_tensor(data.pair_i, dtype=torch.int64).to(self.device),
                ],
                dim=0,
            ).T,
            properties.edge_diff: torch.as_tensor(data.rij).to(self.dtype).to(self.device),
            properties.atomic_numbers: torch.as_tensor(data.elems, dtype=torch.int64).to(self.device),
        }
        
        # If we have existing charges from LAMMPS, use them as initial guess
        if hasattr(data, 'charges') and data.charges is not None:
            batch[properties.atomic_charge] = torch.as_tensor(data.charges).to(self.dtype).to(self.device)
        
        return batch, data, data.nlocal, data.ntotal - data.nlocal

    def _normalize_charges(self, charges: torch.Tensor, natoms: int) -> torch.Tensor:
        """Normalize charges to ensure sum = total_charge."""
        if not self.charge_normalization:
            return charges
        
        local_charges = charges[:natoms]
        current_sum = local_charges.sum()
        correction = (self.total_charge - current_sum) / natoms
        charges[:natoms] = local_charges + correction
        return charges

    def _update_lammps_data(self, data, atom_energies, pair_forces, predicted_charges, natoms, residual_energy=None, residual_forces=None):
        """
        Update LAMMPS data structures with computed energies, forces, and charges.
        
        Energy breakdown:
        - LAMMPS receives: E_short (ML) + E_residual (χ·q + η·q²)
        - LAMMPS will add: E_coul_real (pair coul/long) + E_recip + E_self (kspace)
        
        Force breakdown:
        - LAMMPS receives: F_short (edge forces) (F_residual should be ~0 for well-trained model)
        - LAMMPS will add: F_coul_real (pair coul/long) + F_recip (kspace)
        
        Charges:
        - Written to atom->q for LAMMPS kspace to use
        """
        # Convert to double if needed
        if self.dtype == torch.float32:
            pair_forces = pair_forces.double()
        
        # Update atomic energies (short-range ML energy)
        eatoms = torch.as_tensor(data.eatoms)
        local_energies = atom_energies[:natoms].detach()
        
        # Add residual energy if provided (distributed evenly across atoms)
        if residual_energy is not None:
            # residual_energy is per-structure, distribute to atoms
            residual_per_atom = residual_energy.detach() / natoms
            local_energies = local_energies + residual_per_atom
        
        eatoms.copy_(local_energies)
        data.energy = local_energies.sum().item()
        
        # Update pair forces (short-range ML forces)
        # Note: residual_forces should be ~0 for well-trained QEQ model
        # They come from χ·q + η·q² which has zero gradient w.r.t. positions
        # if charges are predicted correctly (charge equilibration condition)
        data.update_pair_forces_gpu(pair_forces)
        
        # Update atomic charges for LAMMPS kspace
        if predicted_charges is not None and self.use_lammps_kspace:
            # Normalize charges to ensure sum = total_charge
            predicted_charges = self._normalize_charges(predicted_charges, natoms)
            
            # Write charges to LAMMPS atom->q
            # This requires the extended MLIAP interface with charges property
            if hasattr(data, 'charges'):
                charges_np = predicted_charges[:natoms].detach().cpu().numpy()
                if self.dtype == torch.float32:
                    charges_np = charges_np.astype('float64')
                data.charges[:natoms] = charges_np
                logging.debug(f"Step {self.step}: Updated {natoms} atomic charges, sum={charges_np.sum():.6f}")
            else:
                if self.step == 1:
                    logging.warning(
                        "MLIAP data object does not have 'charges' property. "
                        "Long-range Coulomb forces will not be computed correctly. "
                        "Please ensure LAMMPS is compiled with the extended MLIAP interface."
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
