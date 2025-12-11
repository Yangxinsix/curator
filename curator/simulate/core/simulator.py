from __future__ import annotations
import logging, traceback
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
                 logger: Optional[logging.Logger] = None,
                 run_kwargs: Optional[Dict[str, Any]] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.ctx = SimContext(atoms=None, step=0, engine=engine, simulator=self)
        self.engine = engine
        self.callbacks = callbacks or []
        self.init_traj = init_traj
        self.start_index = start_index
        self._step_fn = self._step_proxy()
        self._step_attached = False
        self._run_kwargs = run_kwargs or {}

    def _dispatch(self, hook: str, *args):
        for cb in self.callbacks:
            fn = getattr(cb, hook, None)
            if callable(fn):
                fn(*args)

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

    def run(self, *args, **run_kwargs):
        # Prepare atoms
        if self.engine.atoms is not None:
            self.ctx.atoms = self.engine.atoms
            warnings.warn(f"Atoms are directly read from engine.")
        else:
            self.ctx.atoms = load_atoms_or_traj(self.init_traj, self.start_index)

        # preprocess before simulation
        self._dispatch("on_sim_start", self.ctx)

        # setup engine
        self.engine.setup(self.ctx)
        self._dispatch("on_engine_setup", self.ctx)

        # attach step proxy — shell engines can ignore; step-aware engines should attach it.
        try:
            if not self._step_attached:
                self.engine.attach(self._step_fn, interval=1)
                self._step_attached = True
        except Exception:
            # Engines without attach support can simply ignore this
            pass

        effective_kwargs = run_kwargs or self._run_kwargs

        try:
            self.engine.run(*args, **effective_kwargs)
            self._dispatch("on_sim_end", self.ctx)
        except BaseException as exc:
            self.ctx.state["exception"] = exc
            self.ctx.state["traceback"] = traceback.format_exc()
            self._dispatch("on_exception", self.ctx, exc)
            raise
