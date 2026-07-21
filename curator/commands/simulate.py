from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Union

from omegaconf import DictConfig, OmegaConf

from .common import (
    configure_cli_logger,
    ensure_resolvers,
    log,
    log_logo,
    prepare_cli_environment,
    prepare_run_path,
)


def run_simulation_config(config: DictConfig, *, stream: bool = True) -> None:
    prepare_cli_environment()
    ensure_resolvers()
    from hydra.utils import instantiate
    from pytorch_lightning import seed_everything

    from ..config_utils import normalize_config_sequences, prune_config_targets, read_user_config
    from ..simulate.sim_logging import log_simulation_summary
    from ..utils import CustomFormatter

    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="simulate")
    else:
        normalize_config_sequences(config)
        prune_config_targets(config, logger=log)
    prepare_run_path(config.run_path)

    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=False)
    configure_cli_logger(
        log,
        os.path.join(config.run_path, "simulation.log"),
        CustomFormatter(),
        stream=stream,
    )
    log_logo(log)
    log.debug("Running on host: " + str(socket.gethostname()))
    log_simulation_summary(log, config)
    if "seed" in config:
        log.debug(f"Seed with <{config.seed}>")
        seed_everything(config.seed, workers=True)
    else:
        log.debug("Seed randomly...")

    simulator = instantiate(config.simulator)
    simulator.run()


def run_simulation_config_file(config_path: Union[str, os.PathLike], *, stream: bool = True) -> None:
    run_simulation_config(OmegaConf.load(Path(config_path).expanduser()), stream=stream)


def run_simulate_config(config: DictConfig):
    return run_simulation_config(config, stream=True)


simulate = run_simulate_config
