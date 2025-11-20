from __future__ import annotations
import warnings
from typing import Any, Union
from ase import Atoms
from ..core.engine import BaseEngine
from ..core.context import SimContext

def _resolve(path_or_obj: Any) -> Any:
    if isinstance(path_or_obj, str):
        import importlib
        mod, _, attr = path_or_obj.rpartition(".")
        if not mod:
            raise ValueError(f"Need full module path to resolve: {path_or_obj}")
        return getattr(importlib.import_module(mod), attr)
    return path_or_obj

class MDEngine(BaseEngine):
    """
    ASE MD engine (VelocityVerlet/Langevin/NPT...).
    - setup(ctx: SimContext)  -> normal framework use
    - setup(atoms: Atoms)     -> standalone use
    No SimContext is created internally.
    """
    def __init__(self, dynamics_cls: Any, **dynamics_kwargs: Any):
        super().__init__()
        self.dynamics_cls = _resolve(dynamics_cls)
        self.kw = dynamics_kwargs
        self.atoms: Atoms | None = None
        self.dyn = None

    def setup(self, ctx_or_atoms: Union[SimContext, Atoms, None] = None) -> None:
        if self.dyn is None:
            if isinstance(self.dynamics_cls, type):
                # accept either SimContext or raw Atoms
                self.atoms = ctx_or_atoms.atoms if isinstance(ctx_or_atoms, SimContext) else ctx_or_atoms
                if not isinstance(self.atoms, Atoms):
                    raise TypeError("MDEngine.setup expects SimContext or Atoms.")
                self.dyn = self.dynamics_cls(self.atoms, **self.kw)
            else:
                # if dynamics_cls is a object but not a class
                self.dyn = self.dynamics_cls
                warnings.warn(f"Dynamics is directly initiated from {self.dynamics_cls.__class__.__name__}. It is only not recommended to run simulation in this way.")

    def _attach_to_backend(self, fn, interval: int) -> None:
        if self.dyn is not None:
            self.dyn.attach(fn, interval=interval)

    def run(self, steps: int, **_) -> None:
        if self.dyn is None:
            raise RuntimeError("Call setup() before run().")
        self.dyn.run(steps)