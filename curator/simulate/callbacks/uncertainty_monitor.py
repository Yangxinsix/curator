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
        Function that returns uncertainty dict for the current Atoms. Should include keys:
        - monitor value(s)
        - optional "is_warning" and "is_outlier" flags
    monitor : str | None
        The uncertainty monitor to check (e.g., "sigma"). Used when flags are absent.
    low : float | None
        Lower threshold (used only if backend flags and monitor are absent).
    high : float | None
        Upper threshold (used only if backend flags and monitor are absent).
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
        monitor: Optional[str] = None,
        low: Optional[float] = 0.05,
        high: Optional[float] = 0.5,
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
        self.log = logger or logging.getLogger(__name__)
        self._threshold_key = None  # filled on first call

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

        # Capture threshold key if backend provides it
        if self._threshold_key is None:
            self._threshold_key = getattr(self.backend, "threshold_key", None)

        if self.monitor not in res:
            # Print nothing extra if the monitor is missing.
            return

        # Prefer backend flags if provided
        is_warn = bool(res.get("is_warning", False))
        is_out = bool(res.get("is_outlier", False))
        monitor_key = self._threshold_key or self.monitor

        if "is_warning" in res or "is_outlier" in res:
            # Use flags directly; do not depend on monitor thresholds
            pairs = " ".join(f"{k}={v}" for k, v in sorted(res.items()))
            self.log.info(f"[uncertainty] step={ctx.step} {pairs}")

            if is_out:
                if self._traj is not None:
                    self._traj.write(ctx.atoms)
                ctx.state["early_stop_reason"] = (
                    f"uncertainty {monitor_key or 'N/A'} flagged as outlier; "
                    f"value={res.get(monitor_key, 'N/A')}"
                )
                return
            if is_warn:
                if self._traj is not None:
                    self._traj.write(ctx.atoms)
                if self.uncertain_count is not None:
                    self._low_hits_total += 1
                    if self._low_hits_total >= self.uncertain_count:
                        ctx.state["early_stop_reason"] = (
                            f"uncertainty {monitor_key or 'N/A'} flagged warning "
                            f"{self._low_hits_total} times (threshold={self.uncertain_count}); "
                            f"last_value={res.get(monitor_key, 'N/A')}"
                        )
            return

        # Fallback: threshold-based
        if monitor_key is None:
            return
        if monitor_key not in res:
            return
        val = float(res[monitor_key])
        pairs = " ".join(f"{k}={float(v):.6f}" for k, v in sorted(res.items()))
        self.log.info(f"[uncertainty] step={ctx.step} {pairs}")

        # Only apply thresholds if provided
        if self.high is not None and val >= self.high:
            if self._traj is not None:
                self._traj.write(ctx.atoms)
            ctx.state["early_stop_reason"] = f"uncertainty {monitor_key}={val:.6f} >= high({self.high})"
            return

        if self.low is not None and val >= self.low:
            if self._traj is not None:
                self._traj.write(ctx.atoms)
            if self.uncertain_count is not None:
                self._low_hits_total += 1
                if self._low_hits_total >= self.uncertain_count:
                    ctx.state["early_stop_reason"] = (
                        f"uncertainty {monitor_key} >= low({self.low}) "
                        f"{self._low_hits_total} times (threshold={self.uncertain_count})"
                    )

    def on_sim_end(self, ctx: SimContext):
        if self._traj is not None:
            self._traj.close()
            self._traj = None
