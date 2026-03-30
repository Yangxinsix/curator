"""Compatibility shim for legacy curator.cli imports.

Real command entrypoints now live under ``curator.commands``.
"""

from importlib import import_module

_EXPORTS = {
    "train": ("curator.commands.train", "train"),
    "_resolve_checkpoint_mode": ("curator.commands.train", "_resolve_checkpoint_mode"),
    "tmp_train": ("curator.commands.train", "tmp_train"),
    "tmptrain": ("curator.commands.train", "tmp_train"),
    "simulate": ("curator.commands.simulate", "simulate"),
    "select": ("curator.commands.select", "select"),
    "label": ("curator.commands.label", "label"),
    "evaluate": ("curator.commands.evaluate", "evaluate"),
    "evaluate_main": ("curator.commands.evaluate", "evaluate_main"),
    "deploy": ("curator.commands.deploy", "deploy"),
    "deploy_main": ("curator.commands.deploy", "deploy_main"),
    "convert_main": ("curator.commands.convert", "convert_main"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
