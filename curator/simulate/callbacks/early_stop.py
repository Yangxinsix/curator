from __future__ import annotations
import logging
from typing import Optional
from ..core.callbacks import Callback
from ..core.context import SimContext


class EarlyStop(Callback):
    """
    Raises StopIteration when ctx.state['early_stop_reason'] is set by prior callbacks.
    Attach this after uncertainty/monitor callbacks to enforce early termination.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.log = logger or logging.getLogger(__name__)

    def on_step(self, ctx: SimContext):
        reason = ctx.state.get("early_stop_reason")
        if reason:
            self.log.info(f"Early stop requested: {reason}")
            raise StopIteration(reason)

    def on_sim_end(self, ctx: SimContext):
        # Clear any lingering reason
        ctx.state.pop("early_stop_reason", None)
