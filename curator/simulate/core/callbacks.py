from __future__ import annotations
import logging
from abc import ABC
from typing import Optional
from .context import SimContext

class Callback(ABC):
    def on_sim_start(self, ctx: SimContext): ...
    def on_engine_setup(self, ctx: SimContext): ...
    def on_sim_end(self, ctx: SimContext): ...
    def on_exception(self, ctx: SimContext, exc: BaseException): ...

class LoggerCallback(Callback):
    def __init__(self, logger: Optional[logging.Logger] = None, level: int = logging.INFO):
        self.log = logger or logging.getLogger(__name__)
        self.level = level
    def on_sim_start(self, ctx): self.log.log(self.level, "Simulation started.")
    def on_engine_setup(self, ctx): self.log.log(self.level, f"Engine ready: {type(ctx.engine).__name__}")
    def on_sim_end(self, ctx): self.log.log(self.level, "Simulation finished.")
    def on_exception(self, ctx, exc): self.log.exception("Simulation exception", exc_info=exc)