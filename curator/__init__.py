"""Top-level CURATOR package.

Keep package import lightweight so CLI startup does not eagerly import the
entire training/simulation stack.
"""

from importlib import import_module

__all__ = ["data", "layer", "model", "select", "simulate", "label"]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
