try:
    from .torchsim import CuratorTorchSimAdapter, build_torchsim_callable
    _HAS_TORCHSIM = True
except ImportError:
    CuratorTorchSimAdapter = None
    build_torchsim_callable = None
    _HAS_TORCHSIM = False

__all__ = ["CuratorTorchSimAdapter", "build_torchsim_callable"]
__all__ = [name for name in __all__ if globals().get(name) is not None]
