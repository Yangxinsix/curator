from __future__ import annotations

from typing import Any, Optional, Union

import torch
import torch_sim as ts
from torch_sim.integrators import INTEGRATOR_REGISTRY, Integrator
from torch_sim.units import UnitSystem

from ..core.engine import BaseEngine
from ..core.context import SimContext


class TorchSimEngine(BaseEngine):
    """
    TorchSim MD engine with per-step callback support via Simulator.attach.
    - Accepts a ModelInterface (e.g., CuratorTorchSimAdapter)
    - Handles SimState creation and integrates with torch-sim integrators.
    """

    def __init__(
        self,
        model: Any,
        *,
        integrator: Union[Integrator, str] = Integrator.nve,
        temperature: float = 300.0,
        timestep: float = 1e-3,
        unit_system: UnitSystem = UnitSystem.metal,
        integrator_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.model = model
        self.integrator = Integrator[integrator] if isinstance(integrator, str) else integrator
        self.temperature = temperature
        self.timestep = timestep
        self.unit_system = unit_system
        self.integrator_kwargs = integrator_kwargs or {}

        self.state = None
        self.ctx: Optional[SimContext] = None
        self.atoms = None

    def setup(self, ctx: SimContext) -> None:
        self.ctx = ctx
        self.atoms = ctx.atoms
        self.state = ts.initialize_state(ctx.atoms, self.model.device, self.model.dtype)

    def run(
        self,
        steps: int,
        *,
        temperature: Optional[float] = None,
        timestep: Optional[float] = None,
        trajectory_reporter: Any = None,
        pbar: bool = False,
        **kwargs,
    ) -> None:
        if self.state is None or self.ctx is None:
            raise RuntimeError("Call setup(ctx) before run().")

        temp = float(temperature if temperature is not None else self.temperature)
        dt_val = float(timestep if timestep is not None else self.timestep)

        kT = torch.tensor(temp, dtype=self.state.dtype, device=self.state.device) * self.unit_system.temperature
        dt = torch.tensor(dt_val * self.unit_system.time, dtype=self.state.dtype, device=self.state.device)

        init_func, step_func = INTEGRATOR_REGISTRY[self.integrator]
        state = init_func(state=self.state, model=self.model, kT=kT, dt=dt, **self.integrator_kwargs, **kwargs)

        reporter = trajectory_reporter

        for step in range(1, steps + 1):
            state = step_func(state=state, model=self.model, dt=dt, kT=kT, **self.integrator_kwargs, **kwargs)
            if reporter is not None:
                reporter.report(state, step, model=self.model)

            # Expose state to callbacks
            self.ctx.state["sim_state"] = state
            try:
                # Optional: keep ctx.atoms in sync for legacy callbacks
                self.ctx.atoms = ts.io.state_to_atoms(state)[0]
            except Exception:
                pass

            # Trigger attached callbacks (Simulator installs the step proxy)
            for fn, interval in self._attached:
                if step % interval == 0:
                    fn()

        self.state = state

    def _attach_to_backend(self, fn, interval: int) -> None:
        # No backend attach required; we call callbacks explicitly in run.
        return
