from __future__ import annotations

from typing import Optional, List, Any
from ase.io import Trajectory
from ..core.callbacks import Callback
from ..core.context import SimContext

try:  # optional dependency
    import torch_sim as ts
except Exception:  # pragma: no cover
    ts = None


class TrajectoryWriter(Callback):
    """
    Write structures to trajectory files every ``interval`` steps.

    - Works with ASE atoms or torch-sim SimState (via ctx.state['sim_state']).
    - Can write a single combined file or per-system files (using `{i}` in path or auto suffix).
    """

    def __init__(self, path: str, interval: int = 1, mode: str = "w", per_system: bool = False):
        self.path = path
        self.interval = max(1, int(interval))
        self.mode = mode
        self.per_system = per_system
        self._trajs: List[Trajectory] = []

    def _ensure_trajs(self, n_sys: int):
        if self._trajs:
            return
        base = self.path
        for i in range(n_sys if self.per_system else 1):
            if self.per_system:
                if "{i}" in base:
                    p = base.format(i=i)
                else:
                    stem, ext = base.rsplit(".", 1) if "." in base else (base, "traj")
                    p = f"{stem}_sys{i}.{ext}"
            else:
                p = base
            self._trajs.append(Trajectory(p, self.mode))

    def _atoms_from_ctx(self, ctx: SimContext) -> List[Any]:
        state = ctx.state.get("sim_state")
        if state is not None and ts is not None:
            try:
                return ts.io.state_to_atoms(state)
            except Exception:
                pass
        if isinstance(ctx.atoms, list):
            return ctx.atoms
        if ctx.atoms is not None:
            return [ctx.atoms]
        return []

    def on_sim_start(self, ctx: SimContext):
        atoms_list = self._atoms_from_ctx(ctx)
        n_sys = len(atoms_list) if atoms_list else 1
        self._ensure_trajs(n_sys)

    def on_step(self, ctx: SimContext):
        if ctx.step % self.interval != 0:
            return
        atoms_list = self._atoms_from_ctx(ctx)
        if not atoms_list:
            return
        self._ensure_trajs(len(atoms_list))
        if self.per_system:
            for i, atoms in enumerate(atoms_list):
                if i < len(self._trajs):
                    self._trajs[i].write(atoms)
        else:
            for atoms in atoms_list:
                self._trajs[0].write(atoms)

    def on_sim_end(self, ctx: SimContext):
        self._close()

    def on_exception(self, ctx: SimContext, exc: BaseException):
        self._close()

    def _close(self):
        for t in self._trajs:
            try:
                t.close()
            except Exception:
                pass
        self._trajs = []
