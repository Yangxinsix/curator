from __future__ import annotations

import numpy as np
import logging
import os
from typing import Any, Dict, List, Optional, Union, Callable
import torch
import torch_sim as ts
from ase.io import Trajectory

from .thermo_uncertainty import ThermoWithUncertainty
from ..core.context import SimContext
from ...utils import CustomFormatter
from ..core.callbacks import Callback


class TorchSimThermoLogger(ThermoWithUncertainty):
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
        per_system_logs: bool = False,
        log_dir: Optional[str] = None,
        uncertainty_backend: Optional[Callable[[Any], Dict[str, float]]] = None,
        monitor: Optional[str] = None,
        low: Optional[float] = 0.05,
        high: Optional[float] = 0.5,
        save_path: Optional[str] = None,
        uncertain_count: Optional[int] = None,
    ):
        super().__init__(
            variables=variables,
            header=header,
            interval=interval,
            logger=logger,
            custom_functions=custom_functions,
            uncertainty_backend=uncertainty_backend,
            monitor=monitor,
            low=low,
            high=high,
            save_path=save_path,
            uncertain_count=uncertain_count,
        )
        self.per_system_logs = per_system_logs
        self.log_dir = log_dir
        self._per_loggers: List[logging.Logger] = []
        self._batch_header_printed = False
        self._per_header_printed: List[bool] = []
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

    def _get_state_tensor(self, tensor, idx: Optional[int]):
        if tensor is None:
            return None
        if idx is None:
            return tensor
        return tensor[idx]

    def get_epot(self, ctx: SimContext, idx: Optional[int] = None) -> float:
        st = self._state(ctx)
        if st is not None and hasattr(st, "energy"):
            e = self._get_state_tensor(st.energy.detach().cpu().view(-1), idx or 0)
            return float(e)
        if ctx.atoms is not None:
            return super().get_epot(ctx)
        return 0.0

    def get_ekin(self, ctx: SimContext, idx: Optional[int] = None) -> float:
        st = self._state(ctx)
        if st is not None and hasattr(st, "momenta"):
            # kinetic = sum(p^2 / (2m))
            p = self._get_state_tensor(st.momenta, idx)
            m_full = st.masses.unsqueeze(-1)
            m = self._get_state_tensor(m_full, idx)
            ke = (p * p / (2 * m)).sum()
            return float(ke.detach().cpu())
        if ctx.atoms is not None:
            return super().get_ekin(ctx)
        return 0.0

    def get_etot(self, ctx: SimContext, idx: Optional[int] = None) -> float:
        return self.get_epot(ctx, idx) + self.get_ekin(ctx, idx)

    def get_temp(self, ctx: SimContext, idx: Optional[int] = None) -> float:
        st = self._state(ctx)
        if st is not None and hasattr(st, "momenta"):
            natoms = st.n_atoms if idx is None else st.n_atoms_per_system[idx]
            dof = 3 * natoms
            if dof == 0:
                return 0.0
            ke = self.get_ekin(ctx, idx)
            from ase import units
            return 2.0 * ke / (dof * units.kB)
        if ctx.atoms is not None:
            return super().get_temp(ctx)
        return 0.0

    def get_stress(self, ctx: SimContext, idx: Optional[int] = None):
        st = self._state(ctx)
        if st is not None and hasattr(st, "stress"):
            stress = st.stress.detach().cpu()
            s = self._get_state_tensor(stress, idx)
            s = s.numpy()
            return s.reshape(-1)[[0, 4, 8, 5, 2, 1]]
        if ctx.atoms is not None:
            return super().get_stress(ctx)
        return "N/A"

    def get_pressure(self, ctx: SimContext, idx: Optional[int] = None) -> float:
        s = self.get_stress(ctx, idx)
        try:
            s = np.asarray(s)
            return -float(s[0] + s[1] + s[2]) / 3.0
        except Exception:
            return 0.0

    def get_volume(self, ctx: SimContext, idx: Optional[int] = None) -> float:
        st = self._state(ctx)
        if st is not None and hasattr(st, "cell"):
            cell = st.cell
            if cell.ndim == 3 and idx is not None:
                cell = cell[idx]
            elif cell.ndim == 3:
                cell = cell[0]
            vol = torch.det(cell).detach().cpu().item()
            return vol
        if ctx.atoms is not None:
            return super().get_volume(ctx)
        return 0.0

    def get_density(self, ctx: SimContext, idx: Optional[int] = None) -> float:
        st = self._state(ctx)
        if st is not None and hasattr(st, "masses"):
            masses = st.masses
            mass = masses[idx].sum().detach().cpu().item() if idx is not None else masses.sum().detach().cpu().item()
            vol = self.get_volume(ctx, idx)
            return mass / vol if vol > 0 else 0.0
        if ctx.atoms is not None:
            return super().get_density(ctx)
        return 0.0

    def get_natoms(self, ctx: SimContext, idx: Optional[int] = None) -> int:
        st = self._state(ctx)
        if st is not None:
            if idx is None:
                return int(st.n_atoms)
            return int(st.n_atoms_per_system[idx])
        if ctx.atoms is not None:
            return super().get_natoms(ctx)
        return 0

    def _get_unc_value(self, ctx: SimContext, key: str, idx: Optional[int] = None) -> float:
        return super()._get_unc_value(ctx, key, idx)


    def _format_var(self, ctx: SimContext, var: str, idx: Optional[int] = None) -> str:
        """Fetch and format a variable for a given system index."""
        func = self.variable_funcs.get(var)
        if func is None:
            return super()._format_value("N/A")
        try:
            if idx is not None:
                try:
                    val = func(ctx, idx)
                except TypeError:
                    val = func(ctx)
            else:
                val = func(ctx)
        except Exception:
            val = "N/A"
        return super()._format_value(val)

    def on_sim_start(self, ctx: SimContext):
        """
        Suppress the base header for batched runs; custom header is emitted in on_step.
        """
        st = self._state(ctx)
        if (st is not None and getattr(st, "n_systems", 1) > 1) or isinstance(ctx.atoms, list):
            return
        super().on_sim_start(ctx)

    # Override on_step for batched logging
    def on_step(self, ctx: SimContext):
        if (ctx.step % self.interval) != 0:
            return

        # compute uncertainties if backend provided
        if self._unc_backend is not None:
            try:
                atoms_obj = ctx.atoms
                if isinstance(atoms_obj, list):
                    ctx.state["uncertainty"] = [self._unc_backend(a) or {} for a in atoms_obj]
                else:
                    ctx.state["uncertainty"] = self._unc_backend(atoms_obj) or {}
            except Exception as exc:
                if self.log is not None:
                    self.log.exception(f"Uncertainty backend failed at step {ctx.step}: {exc}")
                ctx.state["uncertainty"] = {}

        # apply uncertainty rules
        unc_data = ctx.state.get("uncertainty", {})
        atoms_obj = ctx.atoms
        if isinstance(unc_data, list):
            for i, d in enumerate(unc_data):
                atom_ref = atoms_obj[i] if isinstance(atoms_obj, list) and i < len(atoms_obj) else atoms_obj
                ctx.state["system_index"] = i
                ctx.state["uncertainty"] = d
                self._apply_band_and_stop(ctx)
        else:
            ctx.state["system_index"] = None
            self._apply_band_and_stop(ctx)

        st = self._state(ctx)
        n_sys = int(getattr(st, "n_systems", 1)) if st is not None else 1
        # prepare per-system loggers on first use
        if self.per_system_logs and self.log_dir and not self._per_loggers and n_sys > 0:
            os.makedirs(self.log_dir, exist_ok=True)
            for i in range(n_sys):
                lg = logging.getLogger(f"{self.log.name}.sys{i}")
                lg.setLevel(self.log.level or logging.INFO)
                fh = logging.FileHandler(os.path.join(self.log_dir, f"simulation_{i}.log"), mode="w")
                fh.setFormatter(CustomFormatter())
                lg.addHandler(fh)
                lg.propagate = False
                self._per_loggers.append(lg)
            self._per_header_printed = [False for _ in range(n_sys)]

        # print header with sys column if batched
        if n_sys > 1 and st is not None:
            # Build a block-per-system table: step once, then all vars for sys0, then sys1, etc.
            vars_no_step = [v for v in self.variables if v != "step"]

            if self.header and not self._batch_header_printed:
                # Single header line: step + (sys, properties)*n_sys
                header_line = f"{'step':>15}" + "".join(
                    "".join(f"{col:>15}" for col in (["sys"] + vars_no_step)) for _ in range(n_sys)
                )
                self.log.info(header_line)
                self._batch_header_printed = True

            values: List[str] = []
            # step once
            if "step" in self.variables:
                values.append(self._format_var(ctx, "step", None))
            # system blocks
            for i in range(n_sys):
                values.append(super()._format_value(i))
                for var in vars_no_step:
                    values.append(self._format_var(ctx, var, i))

            line = "".join(values)
            self.log.info(line)
            if self._per_loggers:
                stride = 1 + len(vars_no_step)
                for idx, lg in enumerate(self._per_loggers):
                    # Per-system header (only its own properties)
                    if self.header and idx < len(self._per_header_printed) and not self._per_header_printed[idx]:
                        header_line = f"{'step':>15}{'sys':>15}" + "".join(f"{v:>15}" for v in vars_no_step)
                        lg.info(header_line)
                        self._per_header_printed[idx] = True

                    if "step" in self.variables:
                        step_val = values[0]
                        start = 1 + stride * idx
                        block = [step_val] + values[start:start + stride]
                    else:
                        start = stride * idx
                        block = values[start:start + stride]
                    lg.info("".join(block))
        else:
            super().on_step(ctx)


class TorchSimTrajectoryWriter(Callback):
    """Alias for TrajectoryWriter maintained for backwards compatibility."""

    def __init__(self, *args, **kwargs):
        from .trajectory import TrajectoryWriter
        self._impl = TrajectoryWriter(*args, **kwargs)

    def on_sim_start(self, ctx: SimContext):
        return self._impl.on_sim_start(ctx)

    def on_step(self, ctx: SimContext):
        return self._impl.on_step(ctx)

    def on_sim_end(self, ctx: SimContext):
        return self._impl.on_sim_end(ctx)
