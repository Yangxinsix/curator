from __future__ import annotations
from typing import Any, Optional
from ..core.engine import BaseEngine
from ..core.context import SimContext

class OptimizerEngine(BaseEngine):
    """Adapter for ASE optimizers (BFGS, FIRE, LBFGS...). Supports per-step attach via optimizer.attach."""
    def __init__(self, optimizer_cls: Any, **optimizer_kwargs: Any):
        super().__init__()
        self.optimizer_cls = optimizer_cls
        self.kw = optimizer_kwargs
        self.atoms = None
        self.opt = None

    def setup(self, ctx: SimContext) -> None:
        self.atoms = ctx.atoms
        self.opt = self.optimizer_cls(ctx.atoms, **self.kw)

    def _attach_to_backend(self, fn, interval: int) -> None:
        if self.opt is not None:
            self.opt.attach(fn, interval=interval)

    def run(self, fmax: float = 0.02, steps: Optional[int] = None, **_) -> None:
        if self.opt is None:
            raise RuntimeError("OptimizerEngine is not set up. Call setup() first.")
        self.opt.run(fmax=fmax, steps=steps)
