from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

from omegaconf import DictConfig, ListConfig


def _target_name(cfg: Any) -> str:
    if isinstance(cfg, DictConfig):
        return cfg.get("_target_", "unknown")
    return str(cfg)


def _as_list(x: Any):
    if isinstance(x, (list, tuple, ListConfig)):
        return list(x)
    return None


def describe_init(init_traj, start_index, batch: bool) -> Tuple[Any, str]:
    """
    Best-effort description of how many systems will be simulated and what they are.
    """
    traj_list = _as_list(init_traj)
    start_list = _as_list(start_index)

    if traj_list is not None:
        labels = []
        for src in traj_list:
            if isinstance(src, str):
                labels.append(Path(src).name)
            else:
                labels.append(type(src).__name__)
        return len(traj_list), f"{labels}"

    if start_list is not None:
        return len(start_list), f"frames {start_list}"

    if batch:
        return "unknown", "all frames (batch=True)"

    return 1, "single system"


def log_simulation_summary(logger: logging.Logger, config: DictConfig) -> None:
    """
    Log a brief simulation setup summary (engine, model, init, systems, device, callbacks, params).
    """
    sim_cfg = config.simulator
    eng_cfg = getattr(sim_cfg, "engine", None)
    callbacks_cfg = getattr(sim_cfg, "callbacks", []) or []
    init_traj = getattr(sim_cfg, "init_traj", None)
    start_index = getattr(sim_cfg, "start_index", None)
    batch_mode = bool(getattr(sim_cfg, "batch", False))

    cb_names = ", ".join(_target_name(cb) for cb in callbacks_cfg) if callbacks_cfg else "none"
    n_sys, sys_desc = describe_init(init_traj, start_index, batch_mode)

    summary = [
        ("Engine", _target_name(eng_cfg)),
        ("Model", getattr(config, "model_path", "unknown")),
        ("Init traj", init_traj if init_traj is not None else "unknown"),
        ("Systems", f"{n_sys} | {sys_desc}"),
        ("Device", getattr(config, "device", "unknown")),
    ]
    logger.debug("Simulation setup:")
    for k, v in summary:
        logger.info("  %-10s: %s", k, v)

    if eng_cfg is not None:
        extra = []
        for key in ("integrator", "temperature", "timestep"):
            if key in eng_cfg:
                extra.append(f"{key}={eng_cfg.get(key)}")
        if extra:
            logger.info("  Engine params: %s", ", ".join(extra))
    logger.info("  Callbacks   : %s", cb_names)
