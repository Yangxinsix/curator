from ..registry import (
    ExternalModelSpec,
    is_external_model_spec,
    load_external_model,
    parse_external_model_spec,
    register_adapter_loader,
)
from . import mace as _mace_loader
from . import nequip as _nequip_loader
from .matgl import MatGLAdapter
from .allegro import AllegroAdapter
from .esen import ESENAdapter

__all__ = [
    "ExternalModelSpec",
    "MatGLAdapter",
    "AllegroAdapter",
    "ESENAdapter",
    "register_adapter_loader",
    "parse_external_model_spec",
    "is_external_model_spec",
    "load_external_model",
]
