from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from collections.abc import Mapping as AbcMapping, Sequence as AbcSequence
import logging
import time

import numpy as np

from ase import units
from curator.data import properties
try:  # optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None


class Plumed:
    """
    Minimal PLUMED bias wrapper for postprocessing energies/forces.

    - Designed to be engine-agnostic (ASE, torch-sim, etc.).
    - Supports batched systems by keeping one PLUMED instance per system.
    """

    def __init__(
        self,
        *,
        plumed_kwargs: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
        use_charge: bool = False,
        update_charge: bool = False,
        debug: bool = False,
        debug_interval: int = 1,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if isinstance(plumed_kwargs, AbcMapping):
            self._plumed_configs = [dict(plumed_kwargs)]
        elif isinstance(plumed_kwargs, AbcSequence) and not isinstance(plumed_kwargs, (str, bytes)):
            if not plumed_kwargs:
                raise ValueError("plumed_kwargs must contain at least one configuration.")
            self._plumed_configs = [dict(cfg) for cfg in plumed_kwargs]
        else:
            raise TypeError("plumed_kwargs must be a dict or a list of dicts.")

        self.use_charge = bool(use_charge)
        self.update_charge = bool(update_charge)
        self.debug = bool(debug)
        self.debug_interval = max(1, int(debug_interval))
        self.logr = logger or logging.getLogger(__name__)

        self._plumed: List[object] = []
        self._natoms: List[int] = []
        self._steps: List[int] = []
        self._transfer_cpu_s = 0.0
        self._transfer_dev_s = 0.0
        self._plumed_s = 0.0
        self._transfer_calls = 0
        self._plumed_calls = 0

    def _config_for_system(self, system_index: int) -> Dict[str, Any]:
        if len(self._plumed_configs) == 1:
            return self._plumed_configs[0]
        if system_index >= len(self._plumed_configs):
            raise IndexError(
                f"plumed_kwargs has {len(self._plumed_configs)} entries, "
                f"but system_index {system_index} was requested."
            )
        return self._plumed_configs[system_index]

    def _new_plumed(self, natoms: int, system_index: int, config: Dict[str, Any]) -> object:
        try:
            from plumed import Plumed as _Plumed
        except Exception as exc:
            raise ImportError("plumed is required to use Plumed.") from exc

        plumed = _Plumed()

        input_lines = config.get("input_lines", []) or []
        if isinstance(input_lines, str):
            input_lines = [input_lines]
        timestep = float(config.get("timestep", 1.0))
        kT = float(config.get("kT", 1.0))
        log = config.get("log", "")
        log = "" if log is None else str(log)
        restart = bool(config.get("restart", False))

        # Unit setup: align with ASE Plumed wrapper (eV, Angstrom, ps).
        ps = 1000 * units.fs
        plumed.cmd("setMDEnergyUnits", units.mol / units.kJ)
        plumed.cmd("setMDLengthUnits", 1 / units.nm)
        plumed.cmd("setMDTimeUnits", 1 / ps)
        plumed.cmd("setMDChargeUnits", 1.0)
        plumed.cmd("setMDMassUnits", 1.0)

        plumed.cmd("setNatoms", int(natoms))
        plumed.cmd("setMDEngine", "CURATOR")
        if log:
            log_name = log.format(i=system_index) if "{i}" in log else log
            plumed.cmd("setLogFile", log_name)
        plumed.cmd("setTimestep", timestep)
        plumed.cmd("setRestart", restart)
        plumed.cmd("setKbT", kT)
        plumed.cmd("init")
        for line in input_lines:
            plumed.cmd("readInputLine", line)
        return plumed

    def _get_plumed(self, natoms: int, system_index: int) -> object:
        while len(self._plumed) <= system_index:
            self._plumed.append(None)
            self._natoms.append(-1)
            self._steps.append(0)
        config = self._config_for_system(system_index)
        if self._plumed[system_index] is None or self._natoms[system_index] != natoms:
            if self._plumed[system_index] is not None:
                self.logr.warning(
                    "Reinitializing PLUMED for system %d (natoms %d -> %d).",
                    system_index,
                    self._natoms[system_index],
                    natoms,
                )
            self._plumed[system_index] = self._new_plumed(natoms, system_index, config)
            self._natoms[system_index] = natoms
            self._steps[system_index] = 0
        return self._plumed[system_index]

    def _next_step(self, system_index: int, step: Optional[int]) -> int:
        if step is not None:
            return int(step)
        while len(self._steps) <= system_index:
            self._steps.append(0)
        cur = self._steps[system_index]
        self._steps[system_index] = cur + 1
        return cur

    def _validate_charges(self, charges: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if not self.use_charge:
            return None
        if charges is None:
            self.logr.warning("PLUMED charge requested but charges are missing; skipping setCharges.")
            return None
        return np.asarray(charges)

    def _compute_bias(
        self,
        positions: np.ndarray,
        energy: float,
        masses: Optional[np.ndarray],
        charges: Optional[np.ndarray],
        cell: Optional[np.ndarray],
        step: int,
        system_index: int,
    ) -> Tuple[float, np.ndarray]:
        natoms = int(len(positions))
        plumed = self._get_plumed(natoms, system_index)

        plumed.cmd("setStep", int(step))

        if self.use_charge:
            charge_arr = self._validate_charges(charges)
            if charge_arr is not None:
                plumed.cmd("setCharges", np.asarray(charge_arr, dtype=float))

        if cell is not None:
            plumed.cmd("setBox", np.asarray(cell, dtype=float))

        plumed.cmd("setPositions", np.asarray(positions, dtype=float))
        plumed.cmd("setEnergy", float(energy))
        if masses is not None:
            plumed.cmd("setMasses", np.asarray(masses, dtype=float))

        forces_bias = np.zeros_like(positions, dtype=float)
        plumed.cmd("setForces", forces_bias)
        virial = np.zeros((3, 3), dtype=float)
        plumed.cmd("setVirial", virial)

        plumed.cmd("prepareCalc")
        plumed.cmd("performCalc")
        energy_bias = np.zeros((1,), dtype=float)
        plumed.cmd("getBias", energy_bias)
        return float(energy_bias[0]), forces_bias

    def _apply_impl(
        self,
        positions: np.ndarray,
        energy: np.ndarray,
        forces: np.ndarray,
        *,
        cell: Optional[np.ndarray],
        masses: Optional[np.ndarray],
        charges: Optional[np.ndarray],
        step: Optional[int],
        system_index: int = 0,
    ) -> Tuple[float, np.ndarray]:
        """Apply PLUMED bias for a single system (numpy inputs)."""
        pos = np.asarray(positions, dtype=float)
        frc = np.asarray(forces, dtype=float)
        e = float(np.asarray(energy, dtype=float).reshape(-1)[0])

        step_val = self._next_step(system_index, step)
        energy_bias, forces_bias = self._compute_bias(
            pos,
            e,
            masses=masses,
            charges=charges,
            cell=cell,
            step=step_val,
            system_index=system_index,
        )
        return e + energy_bias, frc + forces_bias

    def _tensor_to_numpy(self, value: Optional[object]) -> Optional[np.ndarray]:
        """Convert torch tensors to numpy arrays for PLUMED inputs."""
        if value is None:
            return None
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def apply_batch(
        self,
        outputs,
        inputs,
        *,
        step: Optional[int] = None,
    ):
        """Apply PLUMED bias for batched torchsim-style input/output dicts."""
        energy = outputs.get(properties.energy)
        forces = outputs.get(properties.forces)
        if energy is None or forces is None:
            return outputs
        if torch is None or not (isinstance(energy, torch.Tensor) and isinstance(forces, torch.Tensor)):
            return outputs

        positions = inputs.get(properties.positions)
        if positions is None or not isinstance(positions, torch.Tensor):
            return outputs

        cell = inputs.get(properties.cell)
        masses = inputs.get("masses")
        charges = inputs.get(properties.atomic_charge) if properties.atomic_charge in inputs else inputs.get("charges")
        image_idx = inputs.get(properties.image_idx)
        n_atoms = inputs.get(properties.n_atoms)

        t0 = time.perf_counter()
        pos_np = self._tensor_to_numpy(positions)
        e_np = self._tensor_to_numpy(energy)
        f_np = self._tensor_to_numpy(forces)
        cell_np = self._tensor_to_numpy(cell)
        masses_np = self._tensor_to_numpy(masses)
        charges_np = self._tensor_to_numpy(charges)
        image_np = self._tensor_to_numpy(image_idx)
        n_atoms_np = self._tensor_to_numpy(n_atoms)
        t1 = time.perf_counter()

        e_flat = np.asarray(e_np, dtype=float).reshape(-1)
        n_sys = len(e_flat)
        if image_np is None and n_atoms_np is None:
            if n_sys > 1:
                raise ValueError("Batch bias requires image_idx or n_atoms_per_system.")
            indices = [np.arange(len(pos_np))]
        elif image_np is None and n_atoms_np is not None:
            n_atoms_seq = [int(x) for x in np.atleast_1d(n_atoms_np).tolist()]
            indices = []
            offset = 0
            for n in n_atoms_seq:
                indices.append(np.arange(offset, offset + n))
                offset += n
        else:
            image_np = np.asarray(image_np, dtype=int)
            indices = [np.where(image_np == i)[0] for i in range(n_sys)]

        t2 = time.perf_counter()
        for sys_idx, idx in enumerate(indices):
            if idx.size == 0:
                continue
            cell_i = None
            if cell_np is not None:
                cell_arr = np.asarray(cell_np)
                if cell_arr.ndim == 3:
                    cell_i = cell_arr[sys_idx]
                elif cell_arr.ndim == 2 and cell_arr.shape[0] == n_sys * 3 and cell_arr.shape[1] == 3:
                    start = sys_idx * 3
                    cell_i = cell_arr[start:start + 3]
                else:
                    cell_i = cell_arr
            masses_i = None
            if masses_np is not None:
                m_arr = np.asarray(masses_np)
                if m_arr.ndim > 1 and m_arr.shape[0] == n_sys:
                    masses_i = m_arr[sys_idx]
                else:
                    masses_i = m_arr[idx]
            charges_i = None
            if charges_np is not None:
                c_arr = np.asarray(charges_np)
                if c_arr.ndim > 1 and c_arr.shape[0] == n_sys:
                    charges_i = c_arr[sys_idx]
                else:
                    charges_i = c_arr[idx]

            e_val, f_val = self._apply_impl(
                positions=pos_np[idx],
                energy=np.asarray(e_flat[sys_idx], dtype=float),
                forces=f_np[idx],
                cell=cell_i,
                masses=masses_i,
                charges=charges_i,
                step=step,
                system_index=sys_idx,
            )
            e_flat[sys_idx] = float(e_val)
            f_np[idx] = f_val
        t3 = time.perf_counter()

        e_t = torch.as_tensor(e_flat, device=energy.device, dtype=energy.dtype)
        f_t = torch.as_tensor(f_np, device=forces.device, dtype=forces.dtype)
        t4 = time.perf_counter()

        self._record_transfer_time(t1 - t0, t4 - t3)
        self._record_plumed_time(t3 - t2)

        outputs[properties.energy] = e_t
        outputs[properties.forces] = f_t
        return outputs

    def apply_atoms(self, atoms, energy: float, forces: np.ndarray, *, step: Optional[int] = None):
        """Apply PLUMED bias using an ASE Atoms container (single system)."""
        positions = atoms.get_positions()
        cell = atoms.cell[:] if atoms is not None else None
        masses = atoms.get_masses() if atoms is not None else None

        charges = None
        if self.use_charge:
            if self.update_charge and atoms.calc is not None and hasattr(atoms.calc, "get_charges"):
                charges = atoms.calc.get_charges(atoms=atoms.copy())
            else:
                try:
                    charges = atoms.get_initial_charges()
                except Exception:
                    charges = None

        t0 = time.perf_counter()
        result = self._apply_impl(
            positions=np.asarray(positions, dtype=float),
            energy=np.asarray(energy, dtype=float),
            forces=np.asarray(forces, dtype=float),
            cell=cell,
            masses=masses,
            charges=charges,
            step=step,
            system_index=0,
        )
        self._record_plumed_time(time.perf_counter() - t0)
        return result

    def finalize(self) -> None:
        for p in self._plumed:
            if p is None:
                continue
            try:
                p.finalize()
            except Exception:
                pass

    def _record_transfer_time(self, to_cpu_s: float, to_device_s: float) -> None:
        if not self.debug:
            return
        self._transfer_cpu_s += float(to_cpu_s)
        self._transfer_dev_s += float(to_device_s)
        self._transfer_calls += 1
        self._maybe_log_stats()

    def _record_plumed_time(self, plumed_s: float) -> None:
        if not self.debug:
            return
        self._plumed_s += float(plumed_s)
        self._plumed_calls += 1
        self._maybe_log_stats()

    def _maybe_log_stats(self) -> None:
        # Intentionally no-op: stats are logged at the end via log_stats().
        return

    def log_stats(self) -> None:
        """Log averaged timing statistics after a run (if debug is enabled)."""
        if not self.debug:
            return
        total_calls = max(self._plumed_calls, self._transfer_calls)
        if total_calls == 0:
            return
        total_cpu = self._transfer_cpu_s
        total_dev = self._transfer_dev_s
        total_plumed = self._plumed_s
        self.logr.info(
            "PLUMED total timings: cpu_transfer=%.6f s, device_transfer=%.6f s, plumed_compute=%.6f s over %d calls",
            total_cpu,
            total_dev,
            total_plumed,
            total_calls,
        )
