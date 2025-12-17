import torch
from packaging import version

# parsing torch version for no weight loading
v = version.parse(torch.__version__)
major, minor = v.major, v.minor

if (major > 2) or (major == 2 and minor > 4):
    import os
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from . import data, layer, model, select, simulate, label

__all__ = ["data", "layer", "model", "select", "simulate", "label"]