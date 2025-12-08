# curator/model/compat.py
from __future__ import annotations

import importlib
import sys
from typing import Any, Dict, Tuple

_ALIAS_TABLE: Dict[Tuple[str, str], Any] = {}


def register_class_alias(module_path: str, old_name: str, new_obj: Any) -> None:
    key = (module_path, old_name)
    _ALIAS_TABLE[key] = new_obj
    module = sys.modules.get(module_path)
    if module is None:
        module = importlib.import_module(module_path)
    setattr(module, old_name, new_obj)


# PainnModel -> Painn
from .painn import Painn

register_class_alias(
    module_path="curator.model.painn",
    old_name="PainnModel",
    new_obj=Painn,
)

# NequipModel -> Nequip
from .nequip import Nequip

register_class_alias(
    module_path="curator.model.nequip",
    old_name="NequipModel",
    new_obj=Nequip,
)