# simlite/callbacks/calculator.py
from __future__ import annotations
import logging
from typing import Optional, Callable, Union, List, Any
from ase.calculators.calculator import Calculator
from ..core.callbacks import Callback
from ..core.context import SimContext
from ..core.calculator import MLCalculator
from curator.utils import load_models
import torch.nn as nn

class CalculatorAssign(Callback):
    """
    Assign an ASE Calculator to ctx.atoms (and optionally NEB images) at sim start.

    Parameters
    ----------
    calculator : Calculator | callable | str | list[str] | list[nn.Module]
        Ready-made calculator, factory, model path(s), or model instance(s). Paths are loaded via load_models -> MLCalculator.
    warmup : bool
        If True, evaluate energy (and optionally forces) once to warm up the calculator.
    require_forces : bool
        If True (with warmup), also call get_forces().
    apply_to_neb_images : bool
        If True, also assign to each image in ctx.state['neb_images'] (for NEBEngine).
    logger : logging.Logger | None
        Optional logger; defaults to "Simulator".
    """
    def __init__(
        self,
        calculator: Union[
            Calculator,
            Callable[[], Calculator],
            str,
            List[str],
            nn.Module,
            List[nn.Module],
        ],
        warmup: bool = True,
        require_forces: bool = False,
        apply_to_neb_images: bool = False,
        logger: Optional[logging.Logger] = None,
        device: Optional[str] = None,
    ):
        self._calculator = calculator
        self.warmup = warmup
        self.require_forces = require_forces
        self.apply_to_neb_images = apply_to_neb_images
        self.log = logger or logging.getLogger(__name__)
        self.device = device

    def _make_calc(self) -> Calculator:
        # Already a calculator
        if isinstance(self._calculator, Calculator):
            return self._calculator

        # Factory returning a calculator
        if callable(self._calculator) and not isinstance(self._calculator, (str, bytes)):
            return self._calculator()

        # Model-like: path(s) or module(s)
        model_like: Any = self._calculator
        model_list = load_models(model_like, device=self.device)
        self._calculator = MLCalculator(model=model_list, device=self.device)
        return self._calculator

    def _assign_and_warmup(self, atoms) -> None:
        atoms.calc = self._make_calc()
        if self.warmup:
            try:
                _ = atoms.get_potential_energy()
                if self.require_forces:
                    _ = atoms.get_forces()
            except Exception as e:
                self.log.warning(f"Calculator warmup failed: {e}")

    def on_sim_start(self, ctx: SimContext):
        if ctx.atoms is not None:
            self._assign_and_warmup(ctx.atoms)
            self.log.debug("Calcator assigned to atoms.")

    def on_engine_setup(self, ctx: SimContext):
        if self.apply_to_neb_images and "neb_images" in ctx.state:
            for img in ctx.state["neb_images"]:
                self._assign_and_warmup(img)
