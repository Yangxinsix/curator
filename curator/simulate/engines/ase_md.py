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
            if isinstance(self.dynamics_cls, type) or callable(self.dynamics_cls):
                # accept either SimContext or raw Atoms
                self.atoms = ctx_or_atoms.atoms if isinstance(ctx_or_atoms, SimContext) else ctx_or_atoms
                if not isinstance(self.atoms, Atoms):
                    raise TypeError("MDEngine.setup expects SimContext or Atoms.")
                dyn_kwargs = dict(self.kw)
                timestep = dyn_kwargs.pop("timestep", None)
                dt = dyn_kwargs.pop("dt", None)
                friction = dyn_kwargs.get("friction", None)
                temp_k = dyn_kwargs.get("temperature_K", dyn_kwargs.get("temperature", None))
                try:
                    self.dyn = self.dynamics_cls(self.atoms, **dyn_kwargs)
                except TypeError:
                    # Retry with explicit timestep/temperature/friction if provided
                    if timestep is not None or dt is not None:
                        tval = timestep if timestep is not None else dt
                        extra = {}
                        if temp_k is not None:
                            extra["temperature_K"] = temp_k
                        if friction is not None:
                            extra["friction"] = friction
                        self.dyn = self.dynamics_cls(self.atoms, tval, **extra)
                    else:
                        # support functools.partial or callables expecting atoms only
                        self.dyn = self.dynamics_cls(self.atoms)
            else:
                # if dynamics_cls is an object but not a class or callable
                self.dyn = self.dynamics_cls
                warnings.warn(
                    f"Dynamics is directly initiated from {self.dynamics_cls.__class__.__name__}. It is only not recommended to run simulation in this way."
                )

    def _attach_to_backend(self, fn, interval: int) -> None:
        if self.dyn is not None:
            self.dyn.attach(fn, interval=interval)

    def run(self, steps: int, **_) -> None:
        if self.dyn is None:
            raise RuntimeError("Call setup() before run().")
        try:
            self.dyn.run(steps)
        except StopIteration:
            # Early stop requested by callbacks; swallow to allow graceful exit
            return
        except RuntimeError as exc:
            # ASE wraps StopIteration into RuntimeError("generator raised StopIteration")
            if "StopIteration" in str(exc):
                return
            raise
