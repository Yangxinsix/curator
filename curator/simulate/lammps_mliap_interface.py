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


def prepare_model_for_qeq_inference(
    model: Union[NeuralNetworkPotential, LitNNP],
    return_all: bool = False,
) -> NeuralNetworkPotential:
    """
    Prepare a QEQ model for inference, exposing all energy components.
    
    This function modifies the model to output all necessary components for QEQ
    energy decomposition and LAMMPS MLIAP integration:
    
    Per-atom outputs (needed for LAMMPS MLIAP):
    - atomic_energy: per-atom ML short-range energy (E_ML_i)
    - atomic_residual_energy: per-atom residual energy (χ·q_i + η·q_i²)
    - atomic_charge: equilibrated atomic charges
    - edge_forces: per-edge short-range forces
    
    Per-structure outputs (for analysis):
    - short_energy: total ML short-range energy = sum(atomic_energy)
    - residual_energy: total residual energy = sum(atomic_residual_energy)  
    - ewald_energy: Ewald summation energy
    - energy: total energy (short + residual + ewald)
    - forces: total forces
    
    For LAMMPS MLIAP, the per-atom energy should be:
        eatoms[i] = atomic_energy[i] + atomic_residual_energy[i]
    
    And LAMMPS kspace will add the Ewald energy separately.
    
    Args:
        model: Curator MACE-QEQ model (NeuralNetworkPotential or LitNNP)
        return_all: If True, set model._return_all_outputs = True to bypass 
                   extract_outputs filtering entirely
    
    Returns:
        Configured model
        
    Example:
        >>> model = prepare_model_for_qeq_inference(model)
        >>> outputs = model(batch)
        >>> # For LAMMPS MLIAP:
        >>> eatoms = outputs['atomic_energy'] + outputs['atomic_residual_energy']
    """
    # Handle LitNNP wrapper
    if isinstance(model, LitNNP):
        model = model.model
    
    # Set comprehensive model outputs for QEQ analysis
    model.model_outputs = [
        # Per-atom outputs (needed for LAMMPS MLIAP)
        properties.atomic_energy,           # Per-atom ML short-range energy
        properties.atomic_residual_energy,  # Per-atom residual energy (χ·q + η·q²)
        properties.atomic_charge,           # Equilibrated atomic charges
        properties.edge_forces,             # Per-edge short-range forces
        # Per-structure outputs (for analysis)
        properties.short_energy,            # Total ML short-range energy
        properties.residual_energy,         # Total residual energy
        properties.ewald_energy,            # Ewald summation energy
        properties.ewald_forces,            # Forces from Ewald (for debugging)
        properties.residual_forces,         # Forces from residual (should be ~0)
        properties.energy,                  # Total energy
        properties.forces,                  # Total forces
        properties.virial,                  # Virial tensor (if computed)
    ]
    
    # Optionally return ALL data without filtering
    if return_all:
        model._return_all_outputs = True
    
    # Configure readout to output per-atom energies
    for i, key in enumerate(model.representation.readout.model_outputs):
        if key == properties.energy:
            model.representation.readout.per_atom_flags[i] = True
            model.representation.readout.per_atom_keys[i] = properties.atomic_energy
    
    # Configure output modules for LAMMPS mode
    from curator.layer._charge_equilibration import ChargeEquilibration
    for m in model.output_modules:
        if isinstance(m, GradientOutput):
            # CRITICAL: For LAMMPS mode, we compute gradients via edge_diff, not positions
            # LAMMPS provides edge_diff directly (rij vectors), so we don't have positions
            m.grad_on_edge_diff = True
            m.grad_on_positions = False
            m.compute_edge_forces = True
            # MUST set compute_edge_forces_only=True for LAMMPS!
            # Otherwise index_add_ will fail because edge_idx contains ghost atom indices
            # LAMMPS will compute atomic forces from edge forces itself
            m.compute_edge_forces_only = True
            m.model_outputs = [properties.edge_forces]
        elif isinstance(m, ChargeEquilibration):
            # For LAMMPS mode with kspace, don't compute forces in Python
            # LAMMPS kspace will compute Ewald forces using the charges we provide
            # This also avoids the need for 'positions' and 'forces' in data
            m.compute_forces = False
        elif isinstance(m, GlobalRescaleShift):
            m.atomwise_shift = True
            # Note: scale_by/shift_by dict uses 'energy'/'forces' keys, but we output 'atomic_energy'/'edge_forces'
            # Add aliases for the per-atom/per-edge versions
            # Use .clone() to avoid shared tensor references
            if properties.energy in m.scale_by:
                m.scale_by[properties.atomic_energy] = m.scale_by[properties.energy].clone()
            if properties.forces in m.scale_by:
                m.scale_by[properties.edge_forces] = m.scale_by[properties.forces].clone()
            # Also add shift_by alias for atomic_energy
            if properties.energy in m.shift_by:
                m.shift_by[properties.atomic_energy] = m.shift_by[properties.energy].clone()
            m.scale_keys = [properties.energy, properties.atomic_energy, properties.edge_forces]
            m.shift_keys = [properties.energy, properties.atomic_energy]
            m.model_outputs = [
                properties.atomic_energy, 
                properties.atomic_residual_energy,
                properties.edge_forces, 
                properties.atomic_charge,
                properties.short_energy,
                properties.residual_energy,
                properties.ewald_energy,
            ]
    
    model.eval()
    return model


# Module exports
__all__ = [
    "LAMMPS_MLIAP",
    "LAMMPS_MLIAP_QEQ", 
    "prepare_model_for_qeq_inference",
    "CURATORLammpsConfig",
]


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
        # Get dtype from model parameters, not from default
        # This ensures input data matches model's trained precision
        self._model_ref = model.model if isinstance(model, LitNNP) else model
        self.dtype = next(self._model_ref.parameters()).dtype
        self.device = "cpu"
        self.initialized = False
        self.step = 0
        
        # Create mapping from LAMMPS type index to atomic number
        # LAMMPS type indices are 0-based, element_types is ordered as in pair_coeff
        from ase.data import atomic_numbers as ase_atomic_numbers
        self.type_to_atomic_number = torch.tensor(
            [ase_atomic_numbers[elem] for elem in self.element_types],
            dtype=torch.int64
        )

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
                # CRITICAL: For LAMMPS mode, we compute gradients via edge_diff, not positions
                # LAMMPS provides edge_diff directly (rij vectors), so we don't have positions
                m.grad_on_edge_diff = True
                m.grad_on_positions = False
                m.compute_edge_forces = True
                m.compute_edge_forces_only = True
                m.model_outputs = [properties.forces, properties.edge_forces]
            elif isinstance(m, GlobalRescaleShift):
                m.atomwise_shift = True
                if properties.energy in m.scale_by:
                    m.scale_by[properties.atomic_energy] = m.scale_by[properties.energy].clone()
                if properties.forces in m.scale_by:
                    m.scale_by[properties.edge_forces] = m.scale_by[properties.forces].clone()
                if properties.energy in m.shift_by:
                    m.shift_by[properties.atomic_energy] = m.shift_by[properties.energy].clone()
                m.scale_keys = [properties.energy, properties.atomic_energy, properties.edge_forces]
                m.shift_keys = [properties.energy, properties.atomic_energy]
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
                atom_energies, pair_forces = out[properties.atomic_energy], out[properties.edge_forces]

                if self.device.type != "cpu":
                    torch.cuda.synchronize()

            with timer("update_lammps", enabled=self.config.debug_time):
                self._update_lammps_data(data, atom_energies, pair_forces, natoms)

    def _prepare_batch(self, data):
        """Prepare the input batch for the CURATOR model.
        
        For multi-GPU LAMMPS simulations, this includes:
        - Standard batch data (positions, edge indices, etc.)
        - LAMMPS data object for forward/reverse exchange operations
        - Local/ghost atom counts for message passing
        """
        n_local = data.nlocal
        n_ghost = data.ntotal - data.nlocal
        
        # Convert LAMMPS type indices to atomic numbers
        # data.elems contains LAMMPS type indices (0-based)
        lammps_types = torch.as_tensor(data.elems, dtype=torch.int64)
        atomic_numbers = self.type_to_atomic_number.to(self.device)[lammps_types.to(self.device)]
        
        batch = {
            properties.n_atoms: torch.as_tensor(n_local, dtype=torch.int64).unsqueeze(0),
            properties.n_pairs: torch.as_tensor(data.npairs, dtype=torch.int64).unsqueeze(0),
            properties.edge_idx: torch.stack(
                [
                    torch.as_tensor(data.pair_j, dtype=torch.int64).to(self.device),
                    torch.as_tensor(data.pair_i, dtype=torch.int64).to(self.device),
                ],
                dim=0,
            ).T,
            properties.edge_diff: torch.as_tensor(data.rij).to(self.dtype).to(self.device),
            properties.atomic_numbers: atomic_numbers,
            # LAMMPS multi-GPU support
            properties.lammps_data: data,
            properties.n_local: n_local,
            properties.n_ghost: n_ghost,
        }
        return batch

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
        # Get dtype from model parameters, not from default
        # This ensures input data matches model's trained precision  
        self._model_ref = model.model if isinstance(model, LitNNP) else model
        self.dtype = next(self._model_ref.parameters()).dtype
        self.device = "cpu"
        self.initialized = False
        self.step = 0
        
        # Create mapping from LAMMPS type index to atomic number
        # LAMMPS type indices are 0-based, element_types is ordered as in pair_coeff
        from ase.data import atomic_numbers as ase_atomic_numbers
        self.type_to_atomic_number = torch.tensor(
            [ase_atomic_numbers[elem] for elem in self.element_types],
            dtype=torch.int64
        )
        
        # QEQ specific settings
        self.use_lammps_kspace = use_lammps_kspace
        self.charge_normalization = charge_normalization
        self.total_charge = total_charge

        self.model = model.model if isinstance(model, LitNNP) else model
        
        # Use the shared function to configure model outputs
        prepare_model_for_qeq_inference(self.model, return_all=False)
        
        # For LAMMPS MLIAP, we only need minimal outputs (not all analysis outputs)
        # Override to keep only what LAMMPS needs
        self.model.model_outputs = [
            properties.atomic_energy,           # Per-atom ML short-range energy
            properties.atomic_residual_energy,  # Per-atom χ·q + η·q²
            properties.edge_forces,             # Per-edge short-range forces
            properties.atomic_charge,           # Equilibrated charges for LAMMPS kspace
        ]

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
        
        Energy/Force flow for LAMMPS MLIAP with QEQ:
        
        1. ML model computes:
           - atomic_energy[i]: per-atom ML short-range energy
           - edge_forces[ij]: per-edge ML forces
           - atomic_charge[i]: equilibrated charges
           
        2. ChargeEquilibration adds:
           - atomic_residual_energy[i]: per-atom χ·q_i + η·q_i²
           
        3. We output to LAMMPS:
           - eatoms[i] = atomic_energy[i] + atomic_residual_energy[i]
           - pair_forces[ij] = edge_forces[ij]
           - charges[i] = atomic_charge[i]
           
        4. LAMMPS kspace computes:
           - Ewald energy (real + recip + self) using our charges
           - Ewald forces
        
        Total energy in LAMMPS = sum(eatoms) + E_coul_real + E_recip + E_self
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

            # DEBUG: print batch shapes
            if self.step <= 2:
                print(f"\n=== DEBUG QEQ STEP {self.step} ===")
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                    else:
                        print(f"  {k}: type={type(v).__name__}")
                print(f"  data.nlocal={data.nlocal}, data.ntotal={data.ntotal}, data.npairs={data.npairs}")

            with timer("model_forward", enabled=self.config.debug_time):
                out = self.model(batch)
                
                # DEBUG: print output keys
                if self.step <= 2:
                    print(f"\n=== DEBUG MODEL OUTPUT (step {self.step}) ===")
                    print(f"  output keys: {list(out.keys())}")
                    for k, v in out.items():
                        if isinstance(v, torch.Tensor):
                            print(f"  {k}: shape={v.shape}")
                
                # Per-atom ML short-range energies
                atomic_energy = out[properties.atomic_energy]
                
                # Per-atom residual energy (χ·q + η·q²)
                atomic_residual_energy = out.get(properties.atomic_residual_energy, None)
                
                # Per-edge forces
                edge_forces = out[properties.edge_forces]
                
                # Equilibrated charges for LAMMPS kspace
                atomic_charge = out.get(properties.atomic_charge, None)

                if self.device.type != "cpu":
                    torch.cuda.synchronize()

            with timer("update_lammps", enabled=self.config.debug_time):
                self._update_lammps_data(
                    data, 
                    atomic_energy=atomic_energy,
                    atomic_residual_energy=atomic_residual_energy,
                    edge_forces=edge_forces, 
                    atomic_charge=atomic_charge, 
                    natoms=natoms,
                )

    def _prepare_batch(self, data):
        """Prepare the input batch for the CURATOR-QEQ model.
        
        For multi-GPU LAMMPS simulations, this includes:
        - Standard batch data (positions, edge indices, etc.)
        - LAMMPS data object for forward/reverse exchange operations
        - Local/ghost atom counts for message passing
        """
        n_local = data.nlocal
        n_ghost = data.ntotal - data.nlocal
        n_total = data.ntotal
        
        # Convert LAMMPS type indices to atomic numbers
        # data.elems contains LAMMPS type indices (0-based)
        lammps_types = torch.as_tensor(data.elems, dtype=torch.int64)
        atomic_numbers = self.type_to_atomic_number.to(self.device)[lammps_types.to(self.device)]
        
        batch = {
            properties.n_atoms: torch.as_tensor(n_local, dtype=torch.int64, device=self.device).unsqueeze(0),
            properties.n_pairs: torch.as_tensor(data.npairs, dtype=torch.int64, device=self.device).unsqueeze(0),
            properties.edge_idx: torch.stack(
                [
                    torch.as_tensor(data.pair_j, dtype=torch.int64).to(self.device),
                    torch.as_tensor(data.pair_i, dtype=torch.int64).to(self.device),
                ],
                dim=0,
            ).T,
            properties.edge_diff: torch.as_tensor(data.rij).to(self.dtype).to(self.device),
            properties.atomic_numbers: atomic_numbers,
            # image_idx: all atoms belong to the same image (batch index 0)
            # This is required by scatter operations in the model
            properties.image_idx: torch.zeros(n_total, dtype=torch.int64, device=self.device),
            # LAMMPS multi-GPU support
            properties.lammps_data: data,
            properties.n_local: n_local,
            properties.n_ghost: n_ghost,
        }
        
        # If we have existing charges from LAMMPS, use them as initial guess
        # Note: data.charges may be a cupy array on GPU or None
        if hasattr(data, 'charges') and data.charges is not None:
            try:
                # Try to convert charges (handles both numpy and cupy arrays)
                charges_arr = data.charges
                if hasattr(charges_arr, 'get'):  # cupy array
                    charges_arr = charges_arr.get()  # Convert to numpy
                batch[properties.atomic_charge] = torch.as_tensor(charges_arr).to(self.dtype).to(self.device)
            except Exception as e:
                logging.debug(f"Could not convert LAMMPS charges to tensor: {e}")
                # Charges will be computed fresh by the model
        
        return batch

    def _normalize_charges(self, charges: torch.Tensor, natoms: int) -> torch.Tensor:
        """Normalize charges to ensure sum = total_charge."""
        if not self.charge_normalization:
            return charges
        
        local_charges = charges[:natoms]
        current_sum = local_charges.sum()
        correction = (self.total_charge - current_sum) / natoms
        charges[:natoms] = local_charges + correction
        return charges

    def _update_lammps_data(self, data, atomic_energy, atomic_residual_energy, edge_forces, atomic_charge, natoms):
        """
        Update LAMMPS data structures with computed energies, forces, and charges.
        
        Per-atom energy for LAMMPS:
            eatoms[i] = atomic_energy[i] + atomic_residual_energy[i]
        
        Where:
        - atomic_energy: ML short-range per-atom energy
        - atomic_residual_energy: χ·q_i + η·q_i² per-atom
        
        LAMMPS kspace will add Ewald energy (real + recip + self) using the charges.
        
        Forces:
        - We provide edge_forces (per-edge ML short-range forces)
        - LAMMPS kspace will add Coulomb forces
        
        Charges:
        - Written to atom->q for LAMMPS kspace to use
        """
        # Convert to double if needed
        if self.dtype == torch.float32:
            edge_forces = edge_forces.double()
        
        # Compute per-atom energies: E_ML + E_residual
        eatoms = torch.as_tensor(data.eatoms)
        local_ml_energy = atomic_energy[:natoms].detach()
        
        if atomic_residual_energy is not None:
            local_residual_energy = atomic_residual_energy[:natoms].detach()
            local_total_energy = local_ml_energy + local_residual_energy
        else:
            local_total_energy = local_ml_energy
        
        eatoms.copy_(local_total_energy)
        data.energy = local_total_energy.sum().item()
        
        # Update pair forces (short-range ML forces only)
        # LAMMPS kspace will add Coulomb forces
        data.update_pair_forces_gpu(edge_forces)
        
        # Update atomic charges for LAMMPS kspace
        if atomic_charge is not None and self.use_lammps_kspace:
            # Normalize charges to ensure sum = total_charge
            atomic_charge = self._normalize_charges(atomic_charge, natoms)
            
            # Write charges to LAMMPS atom->q
            if hasattr(data, 'charges'):
                local_charges = atomic_charge[:natoms].detach()
                charges_arr = data.charges
                if hasattr(charges_arr, 'get'):
                    charges_np = local_charges.cpu().numpy()
                    if self.dtype == torch.float32:
                        charges_np = charges_np.astype('float64')
                    charges_np = charges_np.copy(order='C')
                    charges_arr[:natoms].set(charges_np)
                    charge_sum = float(charges_np.sum())
                else:
                    charges_np = local_charges.cpu().numpy()
                    if self.dtype == torch.float32:
                        charges_np = charges_np.astype('float64')
                    charges_arr[:natoms] = charges_np
                    charge_sum = float(charges_np.sum())
                if hasattr(data, 'modified_charges_device'):
                    data.modified_charges_device()
                if hasattr(data, 'forward_comm_charges'):
                    data.forward_comm_charges()
                if hasattr(data, 'sync_charges_host'):
                    data.sync_charges_host()
                if hasattr(data, 'update_kspace_qsum_qsq'):
                    data.update_kspace_qsum_qsq()
                logging.debug(f"Step {self.step}: Updated {natoms} atomic charges, sum={charge_sum:.6f}")
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
