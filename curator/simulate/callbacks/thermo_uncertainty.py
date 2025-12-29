from __future__ import annotations
import logging
from typing import Any, Callable, Dict, Optional, Sequence, Union
from ase.io import Trajectory
from curator.data import properties
from .thermo import MDThermoLogger
from ..core.context import SimContext

class ThermoWithUncertainty(MDThermoLogger):
    """
    Minimal extension of MDThermoLogger to:
      - compute uncertainty at the same cadence as thermo printing (interval)
      - add uncertainty columns (exactly `monitor`) to the same line
      - apply simple band rules:
          low     : print only
          medium  : print + save (if save_path)
          high    : print + save (if save_path) + early-stop immediately
      - cumulative early-stop: if value >= low occurs N times in total, early-stop

    Parameters
    ----------
    uncertainty_backend : Callable[[Atoms], Dict[str, float]] | None
        If provided, compute uncertainty right before printing.
        If None, read from ctx.state['uncertainty'] only.
    monitor : Sequence[str] | None
        Uncertainty keys to include as columns (e.g., ("forces_sd", "forces_var")). Used for printing/fallback thresholds.
    low, high : float | None
        Band thresholds (used only if backend flags are absent).
    save_path : str | None
        If set, save medium/high frames to this trajectory.
    uncertain_count : int | None
        If set, early stop when cumulative (val >= low) hits reach this count.

    Note: No max-saves, no consecutive mode, no extra flags.
    """
    def __init__(
        self,
        *,
        variables: Optional[Union[str, Sequence[str]]] = None,
        header: bool = True,
        interval: int = 1,
        logger: Optional[logging.Logger] = None,
        custom_functions: Optional[Dict[str, Callable[[SimContext], Any]]] = None,
        # Uncertainty
        uncertainty_backend: Optional[Callable[[Any], Dict[str, float]]] = None,
        monitor: Optional[str] = None,
        low: Optional[float] = None,
        high: Optional[float] = None,
        save_path: Optional[str] = 'warning_struct.traj',
        uncertain_count: Optional[int] = None,
    ):
        super().__init__(
            variables=variables,
            header=header,
            interval=interval,
            logger=logger,
            custom_functions=custom_functions,
        )
        backend = uncertainty_backend
        # If AutoUncertainty (or similar) has no active backend, disable uncertainty.
        if getattr(backend, "_backend", None) is None:
            backend = None

        self._unc_backend = backend
        self.monitor = monitor if self._unc_backend is not None else None
        # allow logger thresholds to inherit backend thresholds when not specified
        backend_low = getattr(self._unc_backend, "low_threshold", None) if self._unc_backend else None
        backend_high = getattr(self._unc_backend, "high_threshold", None) if self._unc_backend else None
        self.low = float(low) if low is not None else (float(backend_low) if backend_low is not None else None)
        self.high = float(high) if high is not None else (float(backend_high) if backend_high is not None else None)
        self.save_path = save_path
        self.uncertain_count = uncertain_count
        self._logged_uncertainty = False

        # Ensure the uncertainty keys appear as columns
        include_keys: Sequence[str] = ()
        if self._unc_backend is not None:
            include_keys = getattr(self._unc_backend, "uncertainty_keys", ()) or ()
        # Prefer backend-provided keys; fall back to monitor if supplied
        self._include = tuple(include_keys) or ((self.monitor,) if self.monitor else ())

        self._ensure_uncertainty_columns()

        # Optional trajectory writer
        self._traj: Optional[Trajectory] = None
        # Cumulative counter for early-stop on low band
        self._low_hits_total = 0
        self._threshold_key = getattr(self._unc_backend, "threshold_key", None) if self._unc_backend else None

    def on_sim_start(self, ctx: SimContext):
        if self.save_path:
            self._traj = Trajectory(self.save_path, "w")
        self._low_hits_total = 0
        # Re-assert uncertainty columns in case the backend or monitor was swapped post-init
        self._ensure_uncertainty_columns()
        return super().on_sim_start(ctx)

    def on_step(self, ctx: SimContext):
        # Only act on thermo cadence
        if ctx.step % self.interval != 0:
            return

        # Compute uncertainties (if backend is provided)
        if self._unc_backend is not None:
            if not self._logged_uncertainty:
                self.log.info(f"uncertainty backend state: {self._unc_backend}")
            try:
                ctx.state["uncertainty"] = self._unc_backend(ctx.atoms) or {}
            except Exception as exc:
                if self.log is not None:
                    self.log.exception(f"Uncertainty backend failed at step {ctx.step}: {exc}")
                ctx.state["uncertainty"] = {}

        # Apply band actions using the chosen monitor
        self._apply_band_and_stop(ctx)

        if not self._logged_uncertainty:
            self.log.info(f"uncertainty snapshot: {ctx.state.get('uncertainty')}")
            self._logged_uncertainty = True

        # Print the thermo line (now includes all uncertainty columns)
        return super().on_step(ctx)

    def on_sim_end(self, ctx: SimContext):
        if self._traj is not None:
            self._traj.close()
            self._traj = None
        return super().on_sim_end(ctx)

    # --- helpers ---
    def _get_unc_value(self, ctx: SimContext, key: str, idx: Optional[int] = None):
        data = ctx.state.get("uncertainty") or {}
        if isinstance(data, list):
            if idx is None:
                data = data[0] if data else {}
            elif idx < len(data):
                data = data[idx]
            else:
                data = {}
        try:
            return float(data.get(key, float("nan")))
        except Exception:
            return float("nan")

    def _ensure_uncertainty_columns(self):
        for k in self._include:
            if k not in self.variables:
                self.variables.append(k)
            if k not in self.variable_funcs:
                # accept optional idx for batched contexts
                self.variable_funcs[k] = (lambda kk: (lambda ctx, idx=None: self._get_unc_value(ctx, kk, idx)))(k)

    def _resolve_monitor_key(self) -> Optional[str]:
        return self._threshold_key or self.monitor

    def _apply_band_and_stop(self, ctx: SimContext):
        data = ctx.state.get("uncertainty") or {}
        sys_idx = ctx.state.get("system_index")
        prefix = f"[sys{sys_idx}] " if sys_idx is not None else ""

        monitor_key = self._resolve_monitor_key()

        # Prefer backend flags
        is_warn = bool(data.get("is_warning", False))
        is_out = bool(data.get("is_outlier", False))

        if monitor_key is not None and ("is_warning" in data or "is_outlier" in data):
            if is_out:
                if self._traj is not None:
                    self._traj.write(ctx.atoms)
                ctx.state["early_stop_reason"] = (
                    f"{prefix}uncertainty {monitor_key or 'N/A'} flagged as outlier; "
                    f"value={data.get(monitor_key, 'N/A')}"
                )
                return

            if is_warn:
                if self._traj is not None:
                    self._traj.write(ctx.atoms)
                if self.uncertain_count is not None:
                    self._low_hits_total += 1
                    if self._low_hits_total >= self.uncertain_count:
                        ctx.state["early_stop_reason"] = (
                            f"{prefix}uncertainty {monitor_key or 'N/A'} flagged warning "
                            f"{self._low_hits_total} times (threshold={self.uncertain_count}); "
                            f"last_value={data.get(monitor_key, 'N/A')}"
                        )
                return

        # Fallback to thresholds
        if monitor_key is None:
            return
        if monitor_key not in data:
            return
        val = float(data[monitor_key])

        if self.high is not None and val >= self.high:
            if self._traj is not None:
                self._traj.write(ctx.atoms)
            ctx.state["early_stop_reason"] = f"{prefix}uncertainty {monitor_key}={val:.6f} >= high({self.high})"
            return

        if self.low is not None and val >= self.low:
            if self._traj is not None:
                self._traj.write(ctx.atoms)
            if self.uncertain_count is not None:
                self._low_hits_total += 1
                if self._low_hits_total >= self.uncertain_count:
                    ctx.state["early_stop_reason"] = (
                        f"{prefix}uncertainty {monitor_key} >= low({self.low}) "
                        f"{self._low_hits_total} times (threshold={self.uncertain_count})"
                    )
