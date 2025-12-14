from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
from ase import Atoms
from ase import units
from ..core.callbacks import Callback
from ..core.context import SimContext


class MDThermoLogger(Callback):
    """
    LAMMPS-like thermo logger as a Callback.
    - Works with step-aware engines (MD/optimizer/NEB) via Simulator's on_step dispatch.
    - Header is printed at sim start; rows are printed every `interval` steps.
    - Variables can be chosen by name list or by preset key in `default_combinations`.
    """

    default_combinations: Dict[str, List[str]] = {
        "basic_energies": ["step", "epot", "ekin", "etot"],
        "temp_pressure": ["step", "temp", "pressure"],
        "full_thermodynamics": ["step", "epot", "ekin", "etot", "temp", "stress"],
        "structural_properties": ["step", "volume", "density"],
        "energy_per_atom": ["step", "epot_per_atom", "ekin_per_atom"],
        "dynamic_properties": ["step", "temp", "pressure", "epot", "ekin"],
    }

    def __init__(
        self,
        variables: Optional[Union[str, List[str]]] = None,
        header: bool = True,
        interval: int = 1,
        logger: Optional[logging.Logger] = None,
        custom_functions: Optional[Dict[str, Callable[[SimContext], Any]]] = None,
    ):
        self.header = header
        self.interval = max(1, int(interval))
        # Default to project logger so messages reach simulation.log/console.
        self.log = logger or logging.getLogger("curator")
        if self.log.level == logging.NOTSET:
            self.log.setLevel(logging.INFO)

        # Resolve variables
        if isinstance(variables, str) and variables in self.default_combinations:
            self.variables = list(self.default_combinations[variables])
        elif variables is None:
            self.variables = list(self.default_combinations["basic_energies"])  # sensible default
        else:
            self.variables = list(variables)

        # Map variable name -> callable(ctx: SimContext) -> Any
        self.variable_funcs: Dict[str, Callable[[SimContext], Any]] = {
            "step": self.get_step,
            "epot": self.get_epot,
            "ekin": self.get_ekin,
            "etot": self.get_etot,
            "epot_per_atom": self.get_epot_per_atom,
            "ekin_per_atom": self.get_ekin_per_atom,
            "temp": self.get_temp,
            "stress": self.get_stress,
            "pressure": self.get_pressure,
            "volume": self.get_volume,
            "density": self.get_density,
            "natoms": self.get_natoms,
        }
        if custom_functions:
            self.variable_funcs.update(custom_functions)

    def _format_value(self, val: Any) -> str:
        """Pretty-print thermo values consistently."""
        if isinstance(val, float):
            return f"{val:15.5f}"
        if isinstance(val, (list, tuple, np.ndarray)):
            arr = np.asarray(val).reshape(-1)
            show = ",".join(f"{x:.3f}" for x in arr[:6])
            return f"{show:>15}"
        return f"{str(val):>15}"

    # ---- lifecycle hooks ----
    def on_sim_start(self, ctx: SimContext):
        if self.header:
            header_line = "".join(f"{var:>15}" for var in self.variables)
            self.log.info(header_line)

    def on_step(self, ctx: SimContext):
        # Simulator handles interval dispatching; but guard again for safety
        if (ctx.step % self.interval) != 0:
            return
        values: List[str] = []
        for var in self.variables:
            func = self.variable_funcs.get(var)
            if func is None:
                values.append(self._format_value("N/A"))
                continue
            try:
                val = func(ctx)
            except Exception:
                val = "N/A"
            values.append(self._format_value(val))
        self.log.info("".join(values))

    # ---- variable getters (consume ctx directly) ----
    def get_step(self, ctx: SimContext) -> int:
        return int(ctx.step)

    def get_epot(self, ctx: SimContext) -> float:
        return float(ctx.atoms.get_potential_energy())

    def get_ekin(self, ctx: SimContext) -> float:
        try:
            return float(ctx.atoms.get_kinetic_energy())
        except Exception:
            return 0.0

    def get_etot(self, ctx: SimContext) -> float:
        return self.get_epot(ctx) + self.get_ekin(ctx)

    def get_temp(self, ctx: SimContext) -> float:
        # Simple ideal-gas-like estimator: T = 2 E_kin / (kB * dof)
        # NOTE: does not correct for constraints/fixed atoms by default.
        try:
            dof = 3 * ctx.atoms.get_global_number_of_atoms()
            ekin = self.get_ekin(ctx)
            return 2.0 * ekin / (dof * units.kB) if dof > 0 else 0.0
        except Exception:
            return 0.0

    def get_stress(self, ctx: SimContext):
        try:
            return ctx.atoms.get_stress()  # Voigt (xx, yy, zz, yz, xz, xy) in eV/Å^3
        except Exception:
            return "N/A"

    def get_pressure(self, ctx: SimContext) -> float:
        try:
            s = np.asarray(ctx.atoms.get_stress())
            sxx, syy, szz = float(s[0]), float(s[1]), float(s[2])
            return - (sxx + syy + szz) / 3.0  # same units as stress (eV/Å^3)
        except Exception:
            return 0.0

    def get_volume(self, ctx: SimContext) -> float:
        try:
            return float(ctx.atoms.get_volume())
        except Exception:
            return 0.0

    def get_density(self, ctx: SimContext) -> float:
        try:
            m = float(ctx.atoms.get_masses().sum())  # amu
            V = float(ctx.atoms.get_volume())        # Å^3
            return m / V if V > 0 else 0.0           # amu/Å^3
        except Exception:
            return 0.0

    def get_epot_per_atom(self, ctx: SimContext) -> float:
        n = ctx.atoms.get_global_number_of_atoms()
        return self.get_epot(ctx) / n if n > 0 else 0.0

    def get_ekin_per_atom(self, ctx: SimContext) -> float:
        n = ctx.atoms.get_global_number_of_atoms()
        return self.get_ekin(ctx) / n if n > 0 else 0.0

    def get_natoms(self, ctx: SimContext) -> int:
        return int(ctx.atoms.get_global_number_of_atoms())
