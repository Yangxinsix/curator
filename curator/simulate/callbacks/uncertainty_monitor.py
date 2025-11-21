from __future__ import annotations
import logging
from typing import Optional, Dict, Callable
from ase.io import Trajectory
from ..core.callbacks import Callback
from ..core.context import SimContext
from curator.data import properties

class UncertaintyMonitor(Callback):
    """
    Minimal uncertainty monitor with exactly the requested behavior.

    Rules:
      - low band  (val <  high and val <  low): print only
      - medium    (low <= val < high): print + save (if save_path)
      - high      (val >= high): print + save (if save_path) + early-stop immediately
      - cumulative early-stop: if val >= low occurs `uncertain_count` times (total), early-stop

    Parameters
    ----------
    backend : Callable[[Atoms], Dict[str, float]]
        Function that returns uncertainty dict for the current Atoms.
    monitor : str
        The uncertainty monitor to check (e.g., "sigma").
    low : float
        Lower threshold.
    high : float
        Upper threshold.
    interval : int
        Evaluate every N steps.
    save_path : str | None
        If set, save medium/high frames to this trajectory.
    uncertain_count : int | None
        If set, early stop when cumulative (val >= low) hits reach this count.
    logger : logging.Logger | None
        Optional logger; defaults to "Simulator".
    """
    def __init__(
        self,
        backend: Callable,
        monitor: str = properties.f_sd,
        low: float = 0.05,
        high: float = 0.5,
        interval: int = 1,
        save_path: Optional[str] = None,
        uncertain_count: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.backend = backend
        self.monitor = monitor
        self.low = float(low)
        self.high = float(high)
        self.interval = max(1, int(interval))
        self.save_path = save_path
        self.uncertain_count = uncertain_count
        self.log = logger or logging.getLogger("Simulator")

        self._traj: Optional[Trajectory] = None
        self._low_hits_total = 0  # cumulative counter of (val >= low)

    def on_sim_start(self, ctx: SimContext):
        if self.save_path:
            self._traj = Trajectory(self.save_path, "w")
        self._low_hits_total = 0

    def on_step(self, ctx: SimContext):
        if ctx.step % self.interval != 0:
            return

        res: Dict[str, float] = self.backend(ctx.atoms) or {}
        ctx.state["uncertainty"] = res

        if self.monitor not in res:
            # Print nothing extra if the monitor is missing.
            return

        val = float(res[self.monitor])

        # Always print the uncertainty value
        pairs = " ".join(f"{k}={float(v):.6f}" for k, v in sorted(res.items()))
        self.log.info(f"[uncertainty] step={ctx.step} {pairs}")

        # Band checks
        if val >= self.high:
            # high: print already done, then save, then early-stop
            if self._traj is not None:
                self._traj.write(ctx.atoms)
            ctx.state["early_stop_reason"] = f"uncertainty {self.monitor}={val:.6f} >= high({self.high})"
            return  # stop ASAP (EarlyStopCallback will raise)

        elif val >= self.low:
            # medium: print already done, then save
            if self._traj is not None:
                self._traj.write(ctx.atoms)

        # Cumulative low-hit early-stop: count whenever val >= low (medium or high)
        if val >= self.low and self.uncertain_count is not None:
            self._low_hits_total += 1
            if self._low_hits_total >= self.uncertain_count:
                ctx.state["early_stop_reason"] = (
                    f"uncertainty {self.monitor} >= low({self.low}) "
                    f"{self._low_hits_total} times (threshold={self.uncertain_count})"
                )

    def on_sim_end(self, ctx: SimContext):
        if self._traj is not None:
            self._traj.close()
            self._traj = None
