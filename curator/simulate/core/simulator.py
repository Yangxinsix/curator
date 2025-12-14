from __future__ import annotations
import logging, traceback, time
from collections import Counter
from typing import Any, Dict, List, Optional, Union
from ase import Atoms
from .context import SimContext, load_atoms_or_traj
from .engine import BaseEngine
from .callbacks import Callback
import warnings

class Simulator:
    """
    Generic orchestrator (no per-step attach in shell mode):
      - loads atoms,
      - sets up engine,
      - runs,
      - triggers callbacks on start/setup/end/exception.
    """
    def __init__(self,
                 init_traj: Union[str, Atoms],
                 engine: BaseEngine,
                 callbacks: Optional[List[Callback]] = None,
                 start_index: Optional[int] = None,
                 batch: Optional[bool] = None,
                 logger: Optional[logging.Logger] = None,
                 run_kwargs: Optional[Dict[str, Any]] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.ctx = SimContext(atoms=None, step=0, engine=engine, simulator=self)
        self.engine = engine
        self.callbacks = callbacks or []
        self.init_traj = init_traj
        self.start_index = start_index
        self.batch = batch
        self._step_fn = self._step_proxy()
        self._step_attached = False
        self._run_kwargs = run_kwargs or {}

    def _dispatch(self, hook: str, *args):
        for cb in self.callbacks:
            fn = getattr(cb, hook, None)
            if callable(fn):
                fn(*args)

    def _should_batch(self) -> bool:
        """Infer whether we should run in batched mode (multiple systems)."""
        def _as_list(x):
            try:
                from omegaconf import ListConfig
                list_types = (list, tuple, ListConfig)
            except Exception:
                list_types = (list, tuple)
            if isinstance(x, list_types):
                return list(x)
            return None

        traj_list = _as_list(self.init_traj)
        start_list = _as_list(self.start_index)
        if traj_list is not None and len(traj_list) > 1:
            return True
        if start_list is not None and len(start_list) > 1:
            return True
        return False

    def _step_proxy(self):
        """Bridge engine's per-step attach to our callback system.
        Engines that support stepping should call this per step (or via attach()).
        This function enables per-step features such as logging per-step statistics, uncertainty check and so on.
        """
        def _fn():
            self.ctx.step += 1
            for cb in self.callbacks:
                if hasattr(cb, "on_step"):
                    every = getattr(cb, "interval", 1) or 1
                    if self.ctx.step % every == 0:
                        cb.on_step(self.ctx)
        return _fn

    def _summarize_atoms(self, atoms_obj):
        """Return a human-friendly summary for one Atoms object."""
        try:
            symbols = atoms_obj.get_chemical_symbols()
            counts = Counter(symbols)
            natoms = len(symbols)
            parts = [f"natoms={natoms}"] + [f"{el}:{counts[el]}" for el in sorted(counts)]
            return ", ".join(parts)
        except Exception:
            return "unknown atoms"

    def _log_system_info(self):
        atoms_obj = self.ctx.atoms
        if atoms_obj is None:
            return
        if isinstance(atoms_obj, list):
            self.logger.info("Loaded %d systems.", len(atoms_obj))
            for idx, at in enumerate(atoms_obj):
                self.logger.info("  sys%d: %s", idx, self._summarize_atoms(at))
        else:
            self.logger.info("Loaded single system: %s", self._summarize_atoms(atoms_obj))

    def run(self, *args, **run_kwargs):
        wall_start = time.perf_counter()
        timings = {
            "load": 0.0,
            "cb_start": 0.0,
            "engine_setup": 0.0,
            "attach": 0.0,
            "simulate": 0.0,
            "cb_end": 0.0,
        }
        # Prepare atoms
        t0 = time.perf_counter()
        if self.engine.atoms is not None:
            self.ctx.atoms = self.engine.atoms
            warnings.warn(f"Atoms are directly read from engine.")
        else:
            auto_batch = self._should_batch()
            effective_batch = self.batch if self.batch is not None else auto_batch
            self.ctx.atoms = load_atoms_or_traj(self.init_traj, self.start_index, all_frames=effective_batch)
        timings["load"] = time.perf_counter() - t0
        self._log_system_info()

        # preprocess before simulation
        t0 = time.perf_counter()
        self._dispatch("on_sim_start", self.ctx)
        timings["cb_start"] = time.perf_counter() - t0

        # setup engine
        t0 = time.perf_counter()
        self.engine.setup(self.ctx)
        self._dispatch("on_engine_setup", self.ctx)
        timings["engine_setup"] = time.perf_counter() - t0

        # attach step proxy — shell engines can ignore; step-aware engines should attach it.
        t0 = time.perf_counter()
        try:
            if not self._step_attached:
                self.engine.attach(self._step_fn, interval=1)
                self._step_attached = True
        except Exception:
            # Engines without attach support can simply ignore this
            pass
        timings["attach"] = time.perf_counter() - t0

        effective_kwargs = run_kwargs or self._run_kwargs

        try:
            t0 = time.perf_counter()
            self.engine.run(*args, **effective_kwargs)
            timings["simulate"] = time.perf_counter() - t0
            t1 = time.perf_counter()
            self._dispatch("on_sim_end", self.ctx)
            timings["cb_end"] = time.perf_counter() - t1
        except StopIteration as exc:
            timings["simulate"] = time.perf_counter() - t0
            # Graceful early stop
            self.ctx.state["exception"] = exc
            self.ctx.state["traceback"] = traceback.format_exc()
            self._dispatch("on_exception", self.ctx, exc)
        except BaseException as exc:
            timings["simulate"] = time.perf_counter() - t0
            self.ctx.state["exception"] = exc
            self.ctx.state["traceback"] = traceback.format_exc()
            self._dispatch("on_exception", self.ctx, exc)
            raise
        finally:
            total = time.perf_counter() - wall_start
            self.logger.info(
                "Timing summary (s): load=%.4f, callbacks_start=%.4f, engine_setup=%.4f, attach=%.4f, simulate=%.4f, callbacks_end=%.4f, total=%.4f",
                timings["load"],
                timings["cb_start"],
                timings["engine_setup"],
                timings["attach"],
                timings["simulate"],
                timings["cb_end"],
                total,
            )
