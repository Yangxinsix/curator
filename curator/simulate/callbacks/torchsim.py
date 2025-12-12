from __future__ import annotations

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Union, Callable
import torch
import torch_sim as ts

from .thermo import MDThermoLogger
from ..core.context import SimContext


class TorchSimThermoLogger(MDThermoLogger):
    """
    TorchSim-aware thermo logger. Reads from ctx.state['sim_state'] (a torch_sim SimState/MDState)
    and falls back to ctx.atoms if present.
    """

    def __init__(
        self,
        variables: Optional[Union[str, List[str]]] = None,
        header: bool = True,
        interval: int = 1,
        logger: Optional[logging.Logger] = None,
        custom_functions: Optional[Dict[str, Callable[[SimContext], Any]]] = None,
    ):
        super().__init__(variables=variables, header=header, interval=interval, logger=logger, custom_functions=custom_functions)
        # override variable funcs to use SimState
        self.variable_funcs.update(
            {
                "epot": self.get_epot,
                "ekin": self.get_ekin,
                "etot": self.get_etot,
                "temp": self.get_temp,
                "stress": self.get_stress,
                "pressure": self.get_pressure,
                "volume": self.get_volume,
                "density": self.get_density,
                "natoms": self.get_natoms,
            }
        )

    # Helper to fetch current sim state
    def _state(self, ctx: SimContext):
        return ctx.state.get("sim_state")

    def get_epot(self, ctx: SimContext) -> float:
        st = self._state(ctx)
        if st is not None and hasattr(st, "energy"):
            return float(st.energy.detach().cpu().view(-1)[0])
        if ctx.atoms is not None:
            return super().get_epot(ctx)
        return 0.0

    def get_ekin(self, ctx: SimContext) -> float:
        st = self._state(ctx)
        if st is not None and hasattr(st, "momenta"):
            # kinetic = sum(p^2 / (2m))
            p = st.momenta
            m = st.masses.unsqueeze(-1)
            ke = (p * p / (2 * m)).sum()
            return float(ke.detach().cpu())
        if ctx.atoms is not None:
            return super().get_ekin(ctx)
        return 0.0

    def get_etot(self, ctx: SimContext) -> float:
        return self.get_epot(ctx) + self.get_ekin(ctx)

    def get_temp(self, ctx: SimContext) -> float:
        st = self._state(ctx)
        if st is not None and hasattr(st, "momenta"):
            dof = 3 * st.n_atoms
            if dof == 0:
                return 0.0
            ke = self.get_ekin(ctx)
            from ase import units
            return 2.0 * ke / (dof * units.kB)
        if ctx.atoms is not None:
            return super().get_temp(ctx)
        return 0.0

    def get_stress(self, ctx: SimContext):
        st = self._state(ctx)
        if st is not None and hasattr(st, "stress"):
            s = st.stress.detach().cpu().numpy()
            return s.reshape(-1)[[0, 4, 8, 5, 2, 1]]
        if ctx.atoms is not None:
            return super().get_stress(ctx)
        return "N/A"

    def get_pressure(self, ctx: SimContext) -> float:
        s = self.get_stress(ctx)
        try:
            s = np.asarray(s)
            return -float(s[0] + s[1] + s[2]) / 3.0
        except Exception:
            return 0.0

    def get_volume(self, ctx: SimContext) -> float:
        st = self._state(ctx)
        if st is not None and hasattr(st, "cell"):
            cell = st.cell[0] if st.cell.ndim == 3 else st.cell
            vol = torch.det(cell).detach().cpu().item()
            return vol
        if ctx.atoms is not None:
            return super().get_volume(ctx)
        return 0.0

    def get_density(self, ctx: SimContext) -> float:
        st = self._state(ctx)
        if st is not None and hasattr(st, "masses"):
            mass = st.masses.sum().detach().cpu().item()
            vol = self.get_volume(ctx)
            return mass / vol if vol > 0 else 0.0
        if ctx.atoms is not None:
            return super().get_density(ctx)
        return 0.0

    def get_natoms(self, ctx: SimContext) -> int:
        st = self._state(ctx)
        if st is not None:
            return int(st.n_atoms)
        if ctx.atoms is not None:
            return super().get_natoms(ctx)
        return 0
