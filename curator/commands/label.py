import json
import logging
import os
import socket
from shutil import copy

import hydra
from omegaconf import DictConfig, OmegaConf

from .common import CONFIGS_PATH, configure_cli_logger, ensure_resolvers, log, log_logo, prepare_cli_environment, prepare_run_path


@hydra.main(config_path=CONFIGS_PATH, config_name="label", version_base=None)
def label(config: DictConfig):
    prepare_cli_environment()
    ensure_resolvers()
    from ase.db import connect
    from ase.io import Trajectory
    from hydra.utils import instantiate

    from ..data import read_trajectory
    from ..utils import read_user_config

    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="label")
    prepare_run_path(config.run_path)

    configure_cli_logger(
        log,
        os.path.join(config.run_path, "labelling.log"),
        logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"),
        stream=True,
    )
    log_logo(log)
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=False)
    log.debug("Running on host: " + str(socket.gethostname()))

    if config.pool_set:
        images = read_trajectory(config.pool_set)
        indices = config.indices
        if config.al_info:
            with open(config.al_info) as handle:
                indices = json.load(handle)["selected"]
                log.debug(f"Labelling {len(indices)} active learning selected structures: {config.al_info}")
        elif indices is not None:
            log.debug(f"Labelling {len(indices)} selected structures: {config.indices}")
        images = [images[i] for i in indices] if indices is not None else [atoms for atoms in images]
    else:
        raise RuntimeError("Valid configarations for DFT calculation should be provided!")

    if config.split_jobs or config.imgs_per_job:
        from ..utils import split_list

        if config.split_jobs:
            images = split_list(images, config.split_jobs)
        if config.imgs_per_job:
            images = split_list(images, config.imgs_per_job, by_chunk_size=True)
        images = images[config.job_order]
        log.debug(f"Rank {config.job_order}. Total structures: {len(images)}")

    db = connect(config.run_path + "/dft_structures.db")
    db.metadata = {"path": config.run_path + "/dft_structures.db"}
    annotator = instantiate(config.annotator)

    all_converged = []
    for i, atoms in enumerate(images):
        log.debug(f"Labeling structure {i}.")
        try:
            existing_converged = db[i + 1].get("converged")
            if not existing_converged:
                converged = annotator.annotate(atoms)
                db.update(id=i + 1, atoms=atoms, converged=converged)
                log.debug(f"Recomputing structure {i} converged: {converged}")
            else:
                converged = existing_converged
                log.debug(f"Structure {i} converged. Skipping...")
            all_converged.append(converged)
        except KeyError:
            converged = annotator.annotate(atoms)
            db.write(atoms, converged=converged)
            all_converged.append(converged)

        if os.path.exists("OSZICAR") and (not os.path.exists(f"OSZICAR_{i}") or not converged):
            copy("OSZICAR", f"OSZICAR_{i}")
        if os.path.exists("vasp.out") and (not os.path.exists(f"vasp.out_{i}") or not converged):
            copy("vasp.out", f"vasp.out_{i}")

    if config.datapath is not None:
        log.debug(f"Write atoms to {config.datapath}.")
        total_dataset = Trajectory(config.datapath, "a")
        for row in db.select(converged=True):
            if row.get("stored"):
                log.debug(f"Structure {row.id - 1} is already stored in <{config.datapath}>. Skipping...")
            else:
                db.update(id=row.id, stored=True)
                log.debug(f"Write structure {row.id - 1} to <{config.datapath}>")
                total_dataset.write(row.toatoms())

    if not all(all_converged):
        raise RuntimeError(f"Structures {[row.id - 1 for row in db.select(converged=False)]} are not converged!")
    annotator.sweep()
