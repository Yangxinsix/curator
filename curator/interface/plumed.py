from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union
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
        input_lines: Sequence[str],
        timestep: float,
        *,
        kT: float = 1.0,
        log: str = "",
        restart: bool = False,
        use_charge: bool = False,
        update_charge: bool = False,
        debug: bool = False,
        debug_interval: int = 1,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.input_lines = list(input_lines)
        self.timestep = float(timestep)
        self.kT = float(kT)
        self.log = str(log)
        self.restart = bool(restart)
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

    def _new_plumed(self, natoms: int, system_index: int) -> object:
        try:
            from plumed import Plumed as _Plumed
        except Exception as exc:
            raise ImportError("plumed is required to use Plumed.") from exc

        plumed = _Plumed()

        # Unit setup: align with ASE Plumed wrapper (eV, Angstrom, ps).
        ps = 1000 * units.fs
        plumed.cmd("setMDEnergyUnits", units.mol / units.kJ)
        plumed.cmd("setMDLengthUnits", 1 / units.nm)
        plumed.cmd("setMDTimeUnits", 1 / ps)
        plumed.cmd("setMDChargeUnits", 1.0)
        plumed.cmd("setMDMassUnits", 1.0)

        plumed.cmd("setNatoms", int(natoms))
        plumed.cmd("setMDEngine", "CURATOR")
        if self.log:
            log_name = self.log.format(i=system_index) if "{i}" in self.log else self.log
            plumed.cmd("setLogFile", log_name)
        plumed.cmd("setTimestep", float(self.timestep))
        plumed.cmd("setRestart", self.restart)
        plumed.cmd("setKbT", float(self.kT))
        plumed.cmd("init")
        for line in self.input_lines:
            plumed.cmd("readInputLine", line)
        return plumed

    def _get_plumed(self, natoms: int, system_index: int) -> object:
        while len(self._plumed) <= system_index:
            self._plumed.append(None)
            self._natoms.append(-1)
            self._steps.append(0)
        if self._plumed[system_index] is None or self._natoms[system_index] != natoms:
            if self._plumed[system_index] is not None:
                self.logr.warning(
                    "Reinitializing PLUMED for system %d (natoms %d -> %d).",
                    system_index,
                    self._natoms[system_index],
                    natoms,
                )
            self._plumed[system_index] = self._new_plumed(natoms, system_index)
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
        indices: Sequence[np.ndarray],
        step: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply PLUMED bias on pre-sliced numpy arrays (core implementation)."""
        t0 = time.perf_counter()
        e = np.asarray(energy, dtype=float).reshape(-1)
        frc = np.asarray(forces, dtype=float)
        pos = np.asarray(positions, dtype=float)

        for sys_idx, idx in enumerate(indices):
            if idx.size == 0:
                continue
            cell_i = None
            if cell is not None:
                cell_arr = np.asarray(cell)
                cell_i = cell_arr[sys_idx] if cell_arr.ndim == 3 else cell_arr
            masses_i = None
            if masses is not None:
                m_arr = np.asarray(masses)
                if m_arr.ndim > 1 and m_arr.shape[0] == len(indices):
                    masses_i = m_arr[sys_idx]
                else:
                    masses_i = m_arr[idx]
            charges_i = None
            if charges is not None:
                c_arr = np.asarray(charges)
                if c_arr.ndim > 1 and c_arr.shape[0] == len(indices):
                    charges_i = c_arr[sys_idx]
                else:
                    charges_i = c_arr[idx]

            step_val = self._next_step(sys_idx, step)
            energy_bias, forces_bias = self._compute_bias(
                pos[idx],
                float(e[sys_idx]),
                masses=masses_i,
                charges=charges_i,
                cell=cell_i,
                step=step_val,
                system_index=sys_idx,
            )
            e[sys_idx] = float(e[sys_idx]) + energy_bias
            frc[idx] = frc[idx] + forces_bias

        self._record_plumed_time(time.perf_counter() - t0)
        return e, frc

    def apply(
        self,
        positions: Union[np.ndarray, Sequence[Sequence[float]]],
        energy: Union[float, np.ndarray],
        forces: Union[np.ndarray, Sequence[Sequence[float]]],
        *,
        cell: Optional[np.ndarray] = None,
        masses: Optional[np.ndarray] = None,
        charges: Optional[np.ndarray] = None,
        step: Optional[int] = None,
        system_index: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply PLUMED bias for a single system (numpy inputs)."""
        pos = np.asarray(positions, dtype=float)
        idx = np.arange(len(pos))
        e, f = self._apply_impl(
            positions=pos,
            energy=np.asarray(energy, dtype=float),
            forces=np.asarray(forces, dtype=float),
            cell=cell,
            masses=masses,
            charges=charges,
            indices=[idx],
            step=step,
        )
        return np.asarray(e[0]), f

    def apply_batch(
        self,
        positions: np.ndarray,
        energy: np.ndarray,
        forces: np.ndarray,
        *,
        cell: Optional[np.ndarray] = None,
        masses: Optional[np.ndarray] = None,
        charges: Optional[np.ndarray] = None,
        image_idx: Optional[np.ndarray] = None,
        n_atoms_per_system: Optional[Iterable[int]] = None,
        step: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply PLUMED bias for batched systems (numpy inputs)."""
        pos = np.asarray(positions, dtype=float)
        n_sys = len(np.asarray(energy, dtype=float).reshape(-1))
        if image_idx is None and n_atoms_per_system is None:
            if n_sys > 1:
                raise ValueError("Batch bias requires image_idx or n_atoms_per_system.")
            indices = [np.arange(len(pos))]
        elif image_idx is None and n_atoms_per_system is not None:
            n_atoms_seq = list(int(n) for n in n_atoms_per_system)
            indices = []
            offset = 0
            for n in n_atoms_seq:
                indices.append(np.arange(offset, offset + n))
                offset += n
        else:
            image_idx = np.asarray(image_idx, dtype=int)
            indices = [np.where(image_idx == i)[0] for i in range(n_sys)]
        return self._apply_impl(
            positions=pos,
            energy=np.asarray(energy, dtype=float),
            forces=np.asarray(forces, dtype=float),
            cell=cell,
            masses=masses,
            charges=charges,
            indices=indices,
            step=step,
        )

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

        return self.apply(
            positions,
            energy,
            forces,
            cell=cell,
            masses=masses,
            charges=charges,
            step=step,
        )

    def apply_outputs(self, outputs, inputs):
        """Apply PLUMED bias to torchsim-style output/input dicts."""
        energy = outputs.get(properties.energy)
        forces = outputs.get(properties.forces)
        if energy is None or forces is None:
            return outputs
        if torch is None or not (isinstance(energy, torch.Tensor) and isinstance(forces, torch.Tensor)):
            return outputs

        positions = inputs.get(properties.positions)
        if positions is None or not isinstance(positions, torch.Tensor):
            return outputs

        cell = inputs.get("plumed_cell") if "plumed_cell" in inputs else inputs.get(properties.cell)
        masses = inputs.get("masses")
        charges = inputs.get(properties.atomic_charge) if properties.atomic_charge in inputs else inputs.get("charges")

        image_idx = inputs.get(properties.image_idx)
        n_atoms = inputs.get(properties.n_atoms)

        e_new, f_new = self.apply_tensors(
            positions,
            energy,
            forces,
            cell=cell,
            masses=masses,
            charges=charges,
            image_idx=image_idx,
            n_atoms_per_system=n_atoms,
        )
        outputs[properties.energy] = e_new
        outputs[properties.forces] = f_new
        return outputs

    def apply_tensors(
        self,
        positions,
        energy,
        forces,
        *,
        cell=None,
        masses=None,
        charges=None,
        image_idx=None,
        n_atoms_per_system=None,
        step: Optional[int] = None,
    ):
        """Apply PLUMED bias with torch tensors, handling CPU/GPU transfers."""
        if torch is None:
            raise RuntimeError("apply_tensors requires torch to be installed.")
        if not (isinstance(positions, torch.Tensor) and isinstance(energy, torch.Tensor) and isinstance(forces, torch.Tensor)):
            raise TypeError("apply_tensors expects torch tensors for positions/energy/forces.")

        t0 = time.perf_counter()
        pos_np = positions.detach().cpu().numpy()
        e_np = energy.detach().cpu().numpy()
        f_np = forces.detach().cpu().numpy()
        cell_np = cell.detach().cpu().numpy() if isinstance(cell, torch.Tensor) else None
        masses_np = masses.detach().cpu().numpy() if isinstance(masses, torch.Tensor) else None
        charges_np = charges.detach().cpu().numpy() if isinstance(charges, torch.Tensor) else None
        image_np = image_idx.detach().cpu().numpy() if isinstance(image_idx, torch.Tensor) else None
        n_atoms_np = n_atoms_per_system.detach().cpu().numpy() if isinstance(n_atoms_per_system, torch.Tensor) else None
        t1 = time.perf_counter()

        if image_np is None and n_atoms_np is not None and len(np.atleast_1d(e_np)) > 1:
            n_atoms_seq = [int(x) for x in np.atleast_1d(n_atoms_np).tolist()]
        else:
            n_atoms_seq = None

        e_np, f_np = self.apply_batch(
            pos_np,
            e_np,
            f_np,
            cell=cell_np,
            masses=masses_np,
            charges=charges_np,
            image_idx=image_np,
            n_atoms_per_system=n_atoms_seq,
            step=step,
        )

        t2 = time.perf_counter()
        e_t = torch.as_tensor(e_np, device=energy.device, dtype=energy.dtype)
        f_t = torch.as_tensor(f_np, device=forces.device, dtype=forces.dtype)
        t3 = time.perf_counter()

        self._record_transfer_time(t1 - t0, t3 - t2)
        return e_t, f_t

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
        if not self.debug:
            return
        total_calls = max(self._plumed_calls, self._transfer_calls)
        if total_calls == 0 or total_calls % self.debug_interval != 0:
            return
        avg_cpu = self._transfer_cpu_s / self._transfer_calls if self._transfer_calls else 0.0
        avg_dev = self._transfer_dev_s / self._transfer_calls if self._transfer_calls else 0.0
        avg_plumed = self._plumed_s / self._plumed_calls if self._plumed_calls else 0.0
        self.logr.info(
            "PLUMED avg timings: cpu=%.6f s, device=%.6f s, plumed=%.6f s over %d calls",
            avg_cpu,
            avg_dev,
            avg_plumed,
            total_calls,
        )
