import torch

# parsing torch version for no weight loading
ver = torch.__version__.split("+")[0]
major, minor, *_ = map(int, ver.split("."))
if (major > 2) or (major == 2 and minor > 4):
    import os
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from . import data, evaluate, label, layer, model, select, simulate

__all__ = ["data", "evaluate", "label", "layer", "model", "select", "simulate"]
