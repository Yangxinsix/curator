from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from ase import Atoms
from ase.io import read as ase_read
from ..core.engine import BaseEngine
from ..core.context import SimContext

class NEBEngine(BaseEngine):
    """Adapter for ASE NEB.
    - images: list[Atoms] or path to multi-frame file
    - optimizer_cls: ASE optimizer for NEB (e.g., FIRE)
    - interpolator: optional function to interpolate images in-place
    """
    def __init__(
        self,
        images: Union[List[Atoms], str],
        optimizer_cls: Any,
        interpolator: Optional[Any] = None,
        neb_kwargs: Optional[Dict[str, Any]] = None,
        opt_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        from ase.neb import NEB
        self.NEB = NEB
        self.images_src = images
        self.optimizer_cls = optimizer_cls
        self.interpolator = interpolator
        self.neb_kwargs = neb_kwargs or {}
        self.opt_kwargs = opt_kwargs or {}
        self.neb = None
        self.opt = None

    def setup(self, ctx: SimContext) -> None:
        if isinstance(self.images_src, str):
            imgs = ase_read(self.images_src, index=":")
        else:
            imgs = [img.copy() for img in self.images_src]
        if self.interpolator is not None:
            self.interpolator.interpolate(images=imgs)
        self.neb = self.NEB(imgs, **self.neb_kwargs)
        self.opt = self.optimizer_cls(self.neb, **self.opt_kwargs)
        ctx.state["neb_images"] = imgs

    def _attach_to_backend(self, fn, interval: int) -> None:
        if self.opt is not None:
            self.opt.attach(fn, interval=interval)

    def run(self, fmax: float = 0.05, steps: Optional[int] = None, **_) -> None:
        if self.opt is None:
            raise RuntimeError("NEBEngine is not set up. Call setup() first.")
        self.opt.run(fmax=fmax, steps=steps)