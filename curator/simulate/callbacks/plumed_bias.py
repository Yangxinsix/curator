from __future__ import annotations

import logging
from typing import Optional

from ..core.callbacks import Callback
from ..core.context import SimContext
from curator.interface.plumed import Plumed


class PlumedBias(Callback):
    """
    Inject a Plumed instance into calculators/adapters during simulation.
    """

    def __init__(self, plumed: Plumed, logger: Optional[logging.Logger] = None, apply_to_neb_images: bool = False):
        self.plumed = plumed
        self.log = logger or logging.getLogger(__name__)
        self.apply_to_neb_images = apply_to_neb_images

    def _attach_to_atoms_calc(self, atoms) -> None:
        if atoms is None:
            return
        if isinstance(atoms, list):
            for at in atoms:
                self._attach_to_atoms_calc(at)
            return
        if getattr(atoms, "calc", None) is None:
            return
        setattr(atoms.calc, "plumed_bias", self.plumed)

    def on_sim_start(self, ctx: SimContext):
        self._attach_to_atoms_calc(ctx.atoms)
        self._attach_to_engine(ctx)

    def on_engine_setup(self, ctx: SimContext):
        if self.apply_to_neb_images and "neb_images" in ctx.state:
            for img in ctx.state["neb_images"]:
                self._attach_to_atoms_calc(img)

    def _attach_to_engine(self, ctx: SimContext) -> None:
        engine = getattr(ctx, "engine", None)
        if engine is None:
            return
        model = getattr(engine, "model", None)
        if model is None:
            return
        if hasattr(model, "plumed_bias"):
            model.plumed_bias = self.plumed
