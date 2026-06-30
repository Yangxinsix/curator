from __future__ import annotations

import threading

_safe_globals_registered = False
_safe_globals_lock = threading.Lock()


def ensure_torch_safe_globals() -> None:
    """Register safe globals needed by third-party torch.load callsites.

    PyTorch 2.6+ defaults ``torch.load(..., weights_only=True)`` when the
    callsite does not pass ``weights_only`` explicitly. e3nn currently loads
    its packaged constants via a bare ``torch.load(...)`` and requires
    ``slice`` to be allow-listed. Registering it here keeps the fix local to
    CURATOR without forcing global environment variables.
    """

    global _safe_globals_registered
    if _safe_globals_registered:
        return
    try:
        import torch
    except Exception:
        return

    add_safe_globals = getattr(getattr(torch, "serialization", None), "add_safe_globals", None)
    if add_safe_globals is None:
        _safe_globals_registered = True
        return

    with _safe_globals_lock:
        if _safe_globals_registered:
            return
        add_safe_globals([slice])
        _safe_globals_registered = True
