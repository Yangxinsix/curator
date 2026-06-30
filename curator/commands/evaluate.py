import argparse
import logging
import os
import socket
import sys
from pathlib import Path
from typing import List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from .common import (
    CONFIGS_PATH,
    argcomplete,
    configure_cli_logger,
    ensure_resolvers,
    log,
    prepare_cli_environment,
    prepare_run_path,
)
from .deploy import deploy


@hydra.main(config_path=CONFIGS_PATH, config_name="evaluate", version_base=None)
def evaluate(config: DictConfig):
    prepare_cli_environment()
    ensure_resolvers()
    import torch
    from hydra.utils import instantiate

    from ..config_utils import prune_config_targets, read_user_config
    from ..model import EnsembleModel
    from ..utils import load_models

    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="evaluate")
    prune_config_targets(config, logger=log)
    if config.model_path is None or config.datapath is None:
        raise RuntimeError("Both model_path and datapath are required for evaluation.")
    prepare_run_path(config.run_path)

    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=False)
    configure_cli_logger(
        log,
        os.path.join(config.run_path, "predict.log"),
        logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"),
        stream=True,
    )
    log.debug("Running on host: " + str(socket.gethostname()))
    log.info("Evaluating datapath=%s", config.datapath)
    log.info("Evaluating model_path=%s", config.model_path)
    if isinstance(config.device, str) and config.device.startswith("cuda"):
        try:
            cuda_ok = torch.cuda.is_available()
        except Exception:
            log.warning("CUDA check failed; falling back to CPU.")
            config.device = "cpu"
        else:
            if not cuda_ok:
                log.warning("CUDA not available; falling back to CPU.")
                config.device = "cpu"

    log.debug("Using model from <{}>".format(config.model_path))
    if config.deploy is not None:
        model = deploy(
            config.model_path,
            load_weights_only=config.deploy.load_weights_only,
            uncertainty_spec=OmegaConf.select(config, "deploy.uncertainty", default=None),
            return_model=True,
        )
    else:
        model = load_models(config.model_path, config.device)
        model = EnsembleModel(model) if len(model) > 1 else model[0]

    evaluator = instantiate(config.evaluator, model=model)
    evaluator.evaluate(config.datapath)


def evaluate_main(argv: Optional[List[str]] = None):
    prepare_cli_environment()

    def _is_hydra_args(args: List[str]) -> bool:
        for arg in args:
            if not arg:
                continue
            if arg.startswith(("+", "~")):
                return True
            if "=" in arg and not arg.startswith("-"):
                return True
        return False

    def _normalize_device(device: Optional[str]) -> Optional[str]:
        if device is None or not isinstance(device, str) or not device.startswith("cuda"):
            return device
        import torch

        try:
            cuda_ok = torch.cuda.is_available()
        except Exception:
            log.warning("CUDA check failed; falling back to CPU.")
            return "cpu"
        if not cuda_ok:
            log.warning("CUDA not available; falling back to CPU.")
            return "cpu"
        return device

    def _parse_args(args: Optional[List[str]] = None):
        parser = argparse.ArgumentParser(
            description="Evaluate a Curator model on a dataset",
            fromfile_prefix_chars="+",
        )
        parser.add_argument("dataset", nargs="?", help="Dataset path (extxyz/xyz/traj); optional if --data is set")
        parser.add_argument("model", nargs="?", help="Model checkpoint/run directory; optional if --model is set")
        parser.add_argument("-d", "--data", dest="datapath", nargs="+", help="Dataset path(s)")
        parser.add_argument("-m", "--model", dest="model_path", nargs="+", help="Model checkpoint or run directory path(s)")
        parser.add_argument("--device", type=str, default="cuda", help="Device for evaluation (default: cuda)")
        parser.add_argument("--out", type=str, default="evaluate", help="Base output directory (default: ./evaluate)")
        parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
        parser.add_argument("--save-data", action="store_true", help="Save raw predictions/targets to results.npz")
        parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation (default: 8)")
        parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (default: 0)")
        parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pin_memory")
        if argcomplete:
            argcomplete.autocomplete(parser)
        return parser.parse_args(args)

    if argv is None:
        argv = sys.argv[1:]
    if _is_hydra_args(argv):
        return evaluate()
    args = _parse_args(argv)

    ensure_resolvers()
    from ..model import EnsembleModel
    from ..evaluate import Evaluator
    from ..utils import load_models

    datapath = args.datapath or args.dataset
    model_path = args.model_path or args.model
    if datapath is None or model_path is None:
        raise RuntimeError("Both dataset and model are required for evaluation.")

    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)
    configure_cli_logger(
        log,
        os.path.join(str(out_base), "predict.log"),
        logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"),
        stream=True,
    )
    log.info("Evaluating datapath=%s", datapath)
    log.info("Evaluating model_path=%s", model_path)
    device = _normalize_device(args.device) or args.device

    models = load_models(model_path, device=device)
    model = EnsembleModel(models) if len(models) > 1 else models[0]
    evaluator = Evaluator(
        model=model,
        save_data=args.save_data,
        plot_figure=not args.no_plot,
        output_dir=args.out,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    evaluator.evaluate(datapath)
