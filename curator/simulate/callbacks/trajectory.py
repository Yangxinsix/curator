from __future__ import annotations

from typing import Optional
from ase.io import Trajectory
from ..core.callbacks import Callback
from ..core.context import SimContext


class TrajectoryWriter(Callback):
    """Write structures to a trajectory file every ``interval`` steps."""

    def __init__(self, path: str, interval: int = 1, mode: str = "w"):
        self.path = path
        self.interval = max(1, int(interval))
        self.mode = mode
        self._traj: Optional[Trajectory] = None

    def on_sim_start(self, ctx: SimContext):
        if self.path:
            self._traj = Trajectory(self.path, self.mode)

    def on_step(self, ctx: SimContext):
        if self._traj is None:
            return
        if ctx.step % self.interval == 0:
            self._traj.write(ctx.atoms)

    def on_sim_end(self, ctx: SimContext):
        if self._traj is not None:
            self._traj.close()
            self._traj = None

