import contextlib
import io
import importlib.abc
import importlib.machinery
import sys
import warnings

import torch

# Suppress noisy third-party warnings (e3nn + nvml deprecation) on import.
warnings.filterwarnings(
    "ignore",
    message=r"The pynvml package is deprecated\..*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*",
    category=UserWarning,
)

def _patch_cueq_cache_manager():
    import logging

    logging.getLogger("cuequivariance_ops.triton.cache_manager").setLevel(logging.ERROR)
    module = sys.modules.get("cuequivariance_ops.triton.cache_manager")
    if module is None or not hasattr(module, "get_gpu_information"):
        return

    orig = module.get_gpu_information

    def _quiet_get_gpu_information():
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return orig()

    module.get_gpu_information = _quiet_get_gpu_information


class _PostImportHookFinder(importlib.abc.MetaPathFinder):
    def __init__(self, target, callback):
        self._target = target
        self._callback = callback

    def find_spec(self, fullname, path, target=None):
        if fullname != self._target:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec

        original_loader = spec.loader

        class _Loader(importlib.abc.Loader):
            def create_module(self, spec):
                if hasattr(original_loader, "create_module"):
                    return original_loader.create_module(spec)
                return None

            def exec_module(self, module):
                # Silence stdout/stderr during cueq cache_manager import,
                # which prints GPU info before we can patch it.
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    original_loader.exec_module(module)
                self._callback()

        spec.loader = _Loader()
        spec.loader._callback = self._callback
        return spec


if "cuequivariance_ops.triton.cache_manager" in sys.modules:
    _patch_cueq_cache_manager()
else:
    sys.meta_path.insert(
        0,
        _PostImportHookFinder("cuequivariance_ops.triton.cache_manager", _patch_cueq_cache_manager),
    )

# parsing torch version for no weight loading
ver = torch.__version__.split("+")[0]
major, minor, *_ = map(int, ver.split("."))
if (major > 2) or (major == 2 and minor > 4):
    import os
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

__all__ = ["data", "layer", "model", "select", "simulate", "label"]


def __getattr__(name):
    if name in __all__:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
