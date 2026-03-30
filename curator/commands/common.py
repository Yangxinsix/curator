import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union

try:
    import argcomplete
except ImportError:  # pragma: no cover
    argcomplete = None


CONFIGS_PATH = str(Path(__file__).resolve().parent.parent / "configs")

log = logging.getLogger("curator")
log.setLevel(logging.DEBUG)


class _ConsoleProgressFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "progress", False)


def prepare_run_path(run_path: Optional[Union[str, os.PathLike]]) -> None:
    os.makedirs(os.fspath(run_path or "."), exist_ok=True)


def prepare_cli_environment() -> None:
    os.environ.pop("SLURM_NTASKS", None)
    os.environ.pop("SLURM_JOB_NAME", None)


def configure_cli_logger(
    logger: logging.Logger,
    log_path: str,
    formatter: logging.Formatter,
    stream: bool = True,
) -> None:
    log_path = os.path.abspath(log_path)
    for handler in list(logger.handlers):
        if isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) in (sys.stdout, sys.stderr):
            logger.removeHandler(handler)
    if not any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == log_path
        for h in logger.handlers
    ):
        fh = logging.FileHandler(log_path, mode="w")
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    if stream:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        sh.addFilter(_ConsoleProgressFilter())
        logger.addHandler(sh)
    logger.propagate = False


def ensure_cli_stream_logger(
    logger: logging.Logger,
    level: int = logging.INFO,
    formatter: Optional[logging.Formatter] = None,
) -> None:
    if formatter is None:
        formatter = logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s")
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) in (sys.stdout, sys.stderr):
            return
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    stream_handler.addFilter(_ConsoleProgressFilter())
    logger.addHandler(stream_handler)
    logger.propagate = False


_resolvers_registered = False
_LOGO_LOGGED = False
_torch_safe_globals_registered = False


def ensure_resolvers() -> None:
    global _resolvers_registered
    if _resolvers_registered:
        return
    from ..utils import register_resolvers

    register_resolvers()
    _resolvers_registered = True


def ensure_torch_safe_globals() -> None:
    global _torch_safe_globals_registered
    if _torch_safe_globals_registered:
        return
    try:
        import torch
    except Exception:
        return
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is not None:
        add_safe_globals([slice])
    _torch_safe_globals_registered = True


def log_logo(logger: Optional[logging.Logger] = None) -> None:
    global _LOGO_LOGGED
    if _LOGO_LOGGED:
        return
    _LOGO_LOGGED = True
    active_logger = logger or logging.getLogger("curator")
    logo = [
        """
            █████████  █████  █████ ███████████     █████████   ███████████    ███████    ███████████  
           ███░░░░░███░░███  ░░███ ░░███░░░░░███   ███░░░░░███ ░█░░░███░░░█  ███░░░░░███ ░░███░░░░░███ 
          ███     ░░░  ░███   ░███  ░███    ░███  ░███    ░███ ░   ░███  ░  ███     ░░███ ░███    ░███ 
         ░███          ░███   ░███  ░██████████   ░███████████     ░███    ░███      ░███ ░██████████  
         ░███          ░███   ░███  ░███░░░░░███  ░███░░░░░███     ░███    ░███      ░███ ░███░░░░░███ 
         ░░███     ███ ░███   ░███  ░███    ░███  ░███    ░███     ░███    ░░███     ███  ░███    ░███ 
          ░░█████████  ░░████████   █████   █████ █████   █████    █████    ░░░███████░   █████   █████
           ░░░░░░░░░    ░░░░░░░░   ░░░░░   ░░░░░ ░░░░░   ░░░░░    ░░░░░       ░░░░░░░    ░░░░░   ░░░░░

                           Active learning for machine learning interatomic potentials
        """,
    ]
    display_lines = [line.replace("\\\\", "\\") for line in logo]
    width = max(max(len(line) for line in display_lines), 80)
    for line in display_lines:
        active_logger.info(line.center(width))


def torch_load_compat(torch_module, path, **kwargs):
    try:
        return torch_module.load(path, **kwargs)
    except TypeError:
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("weights_only", None)
        return torch_module.load(path, **fallback_kwargs)
