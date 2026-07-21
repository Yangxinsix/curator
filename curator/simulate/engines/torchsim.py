from __future__ import annotations

from typing import Any, Callable, Optional, Union
import inspect

import torch
try:
    import torch_sim as ts
    from torch_sim.integrators import INTEGRATOR_REGISTRY, Integrator
    from torch_sim.units import UnitSystem
    _HAS_TORCHSIM = True
except ImportError:  # optional dependency
    ts = None
    INTEGRATOR_REGISTRY = {}
    Integrator = None  # type: ignore
    UnitSystem = None  # type: ignore
    _HAS_TORCHSIM = False

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
        integrator: Union["Integrator", str] = "nve",
        temperature: float = 300.0,
        timestep: float = 1e-3,
        unit_system: "UnitSystem" = None,
        integrator_kwargs: Optional[dict[str, Any]] = None,
        **_: Any,
    ):
        if not _HAS_TORCHSIM:
            raise ImportError("torch-sim is not installed. Install `torch-sim` to use TorchSimEngine.")
        super().__init__()
        self.model = model
        self.integrator = Integrator[integrator] if _HAS_TORCHSIM and isinstance(integrator, str) else integrator
        self.temperature = temperature
        self.timestep = timestep
        self.unit_system = unit_system if unit_system is not None else (UnitSystem.metal if _HAS_TORCHSIM else None)
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
        integrator_kwargs = dict(self.integrator_kwargs)
        integrator_kwargs.update(kwargs)
        pressure_gpa = integrator_kwargs.pop("external_pressure_GPa", None)
        if pressure_gpa is not None and "external_pressure" not in integrator_kwargs:
            integrator_kwargs["external_pressure"] = (
                torch.tensor(float(pressure_gpa), dtype=self.state.dtype, device=self.state.device)
                * 10000.0
                * float(self.unit_system.pressure)
            )

        init_kwargs = _accepted_kwargs(init_func, integrator_kwargs)
        step_kwargs = _accepted_kwargs(step_func, integrator_kwargs)
        state = init_func(state=self.state, model=self.model, kT=kT, dt=dt, **init_kwargs)

        reporter = trajectory_reporter

        for step in range(1, steps + 1):
            state = step_func(state=state, model=self.model, dt=dt, kT=kT, **step_kwargs)
            if reporter is not None:
                reporter.report(state, step, model=self.model)

            # Expose state to callbacks
            self.ctx.state["sim_state"] = state
            try:
                # Optional: keep ctx.atoms in sync for legacy callbacks
                atoms = ts.io.state_to_atoms(state)
                self.ctx.atoms = atoms[0] if len(atoms) == 1 else atoms
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


def _accepted_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(fn)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}
