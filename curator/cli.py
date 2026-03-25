import hydra
from omegaconf import DictConfig, OmegaConf
import sys, os, json
from pathlib import Path
import argparse
try:
    import argcomplete
except ImportError:  # pragma: no cover
    argcomplete = None

import logging
import socket
import contextlib
from typing import Optional, Union, Dict, List
import re

# very ugly solution for solving pytorch lighting and myqueue conflictions
if "SLURM_NTASKS" in os.environ:
    del os.environ["SLURM_NTASKS"]
if "SLURM_JOB_NAME" in os.environ:
    del os.environ["SLURM_JOB_NAME"]

# Set up logger for the different tasks 
log = logging.getLogger('curator')
log.setLevel(logging.DEBUG)

class _ConsoleProgressFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "progress", False)


def _prepare_run_path(run_path: Optional[Union[str, os.PathLike]]) -> None:
    os.makedirs(os.fspath(run_path or "."), exist_ok=True)


def _configure_cli_logger(
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

_resolvers_registered = False


def _ensure_resolvers():
    global _resolvers_registered
    if _resolvers_registered:
        return
    from .utils import register_resolvers

    register_resolvers()
    _resolvers_registered = True


def _resolve_checkpoint_mode(task_cfg: DictConfig) -> str:
    mode = str(task_cfg.checkpoint_mode).strip().lower()
    if mode not in {"weights", "model", "resume"}:
        raise ValueError(
            f"Unknown task.checkpoint_mode={mode!r}; expected one of "
            f"['weights', 'model', 'resume']."
        )
    return mode

# Trainining with Pytorch Lightning (only with weights and biasses)
@hydra.main(config_path="configs", config_name="train", version_base=None)
def train(config: DictConfig) -> None:
    """ Train the model with pytorch lightning.
    
    Args:
        config (DictConfig): The configuration file.
    Returns:
        None

    """
    _ensure_resolvers()
    from hydra.utils import instantiate
    import torch
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning import seed_everything
    from curator.model import LitNNP, NeuralNetworkPotential
    from .utils import (
        read_user_config,
        CustomFormatter,
        find_best_model,
        normalize_config_sequences,
        prune_config_targets,
        update_config_from_datamodule,
        log_logo,
        update_model,
        update_model_domains,
    )

    # Load the arguments 
    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="train")

    normalize_config_sequences(config)
    prune_config_targets(config, logger=log)
    _prepare_run_path(config.run_path)
    _configure_cli_logger(
        log,
        os.path.join(config.run_path, "training.log"),
        CustomFormatter(),
        stream=True,
    )
    log_logo(log)

    # Save yaml file in run_path
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=False)
    log.debug("Running on host: " + str(socket.gethostname()))
    
    # Set up seed
    if hasattr(config, "trainer") and getattr(config.trainer, "accelerator", None) == "cpu":
        try:
            torch.cuda.is_available = lambda: False  # avoid CUDA init on CPU-only runs
            torch.cuda.device_count = lambda: 0
        except Exception:
            pass
    if "seed" in config:
        log.debug(f"Seed with <{config.seed}>")
        seed_everything(config.seed, workers=True)
    else:
        log.debug("Seed randomly...")
    
    # Initiate the datamodule
    log.debug(f"Instantiating datamodule <{config.data._target_}> from dataset {config.data.datapath or config.data.train_path}")
    datamodule = instantiate(config.data)
    datamodule.setup()
    # something must be inferred from data before instantiating the model
    update_config_from_datamodule(config, datamodule, logger=log)

    # Extend or replace domains
    domain_mode = getattr(config.task, "domain_mode", None)
    new_domains = getattr(config.task, "new_domains", None)
    if domain_mode is None and config.model_path is not None:
        if hasattr(datamodule, "domain_modules") and len(datamodule.domain_modules) > 1:
            domain_mode = "extend"
            config.task.domain_mode = domain_mode
            log.debug("Auto-set domain_mode=extend for multi-domain fine-tune.")
    if domain_mode in ("extend", "replace") and new_domains is None:
        if hasattr(datamodule, "domain_modules") and hasattr(datamodule, "domain_to_id"):
            inferred = []
            for name in datamodule.domain_modules.keys():
                if str(name).lower().startswith("replay"):
                    continue
                dom_id = datamodule.domain_to_id.get(name)
                if dom_id is not None:
                    inferred.append(str(dom_id))
            if inferred:
                new_domains = inferred
                config.task.new_domains = inferred
                log.debug("Inferred new_domains from datapath (excluding replay): %s", inferred)

    resume_ckpt = None
    checkpoint_outputs = None

    if config.model_path is not None:
        config.model_path = find_best_model(config.model_path)[0]
        checkpoint_mode = _resolve_checkpoint_mode(config.task)
        log.debug(f"Loading trained model from {config.model_path}")
        log.debug("Checkpoint loading mode: %s", checkpoint_mode)
        if checkpoint_mode == "model":
            state_dict = torch.load(config.model_path)
            if isinstance(state_dict, torch.nn.Module):
                model = state_dict
                checkpoint_outputs = getattr(state_dict, "outputs", None)
            else:
                model = state_dict['model']
                checkpoint_outputs = state_dict.get('outputs')
            if not isinstance(model, NeuralNetworkPotential):
                raise TypeError(f"Expected NeuralNetworkPotential, got {type(model)}")
        else:
            if checkpoint_mode == "weights":
                from collections import OrderedDict
                state_dict = torch.load(config.model_path)
                raw_state_dict = None
                if isinstance(state_dict, torch.nn.Module):
                    # Keep source architecture from checkpoint, then expand domains later.
                    model = state_dict
                    raw_state_dict = state_dict.state_dict()
                else:
                    checkpoint_model = state_dict.get("model")
                    checkpoint_model_cfg = state_dict.get("model_params")
                    if domain_mode in ("extend", "replace"):
                        if isinstance(checkpoint_model, torch.nn.Module):
                            model = checkpoint_model
                        elif checkpoint_model_cfg is not None:
                            model = instantiate(checkpoint_model_cfg, _convert_="all")
                        else:
                            raise ValueError(
                                "Weights loading for multi-domain fine-tuning requires the checkpoint "
                                "to store either a full model or model_params so the single-domain "
                                "source architecture can be recovered."
                            )
                        log.debug(
                            "Weights mode with domain_mode=%s: initialized source model from checkpoint metadata before domain expansion.",
                            domain_mode,
                        )
                    else:
                        model = instantiate(config.model)

                    raw_state_dict = state_dict.get("state_dict")
                    if raw_state_dict is None and isinstance(checkpoint_model, torch.nn.Module):
                        raw_state_dict = checkpoint_model.state_dict()
                if raw_state_dict is None:
                    raise ValueError("Checkpoint is missing a state_dict for weights loading.")
                new_state_dict = OrderedDict((key.replace('model.', ''), value) for key, value in raw_state_dict.items())
                model.load_state_dict(new_state_dict, strict=False)
            else:
                model = instantiate(config.model)
                resume_ckpt = config.model_path
                log.debug(
                    "Resuming training from checkpoint; optimizer and scheduler states will be restored from %s",
                    resume_ckpt,
                )
    else:
        model = instantiate(config.model)

    init_from = getattr(config.task, "init_new_domains_from", None)
    if domain_mode in ("extend", "replace") and new_domains is not None:
        if not any(hasattr(module, "domain_modules") for module in model.modules()):
            try:
                model = update_model(model)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to upgrade a single-domain checkpoint to a domain-aware model "
                    f"for domain_mode={domain_mode!r}. Use a newer checkpoint or checkpoint_mode=model."
                ) from exc
            log.debug(
                "Upgraded single-domain model structure before applying domain_mode=%s.",
                domain_mode,
            )
        init_strategy = "copy" if init_from is not None else "random"
        updated = update_model_domains(
            model,
            new_domains,
            mode=domain_mode,
            template_domain=init_from or "0",
            init_strategy=init_strategy,
            logger=log,
        )
        log.debug("Updated model domains: mode=%s new_domains=%s updated=%s", domain_mode, new_domains, updated)

    # Casting model dtype to data dtype
    target_dtype = None
    if hasattr(datamodule, "default_dtype"):
        target_dtype = datamodule.default_dtype
    elif hasattr(datamodule, "domain_modules") and datamodule.domain_modules:
        first_dm = next(iter(datamodule.domain_modules.values()))
        target_dtype = getattr(first_dm, "default_dtype", None)
    if target_dtype is not None:
        model = model.to(dtype=target_dtype)
        log.debug("Casting model dtype to data dtype %s", target_dtype)

    if config.compile:
        log.debug("Compiling model with torch.compile")
        model = torch.compile(model)

    log.debug(f"Instantiating task <{config.task._target_}>")
    task: LitNNP = instantiate(config.task, model=model)
    if checkpoint_outputs is not None:
        if not isinstance(checkpoint_outputs, torch.nn.ModuleList):
            checkpoint_outputs = torch.nn.ModuleList(checkpoint_outputs)
        task.outputs = checkpoint_outputs

    log.debug(f"Instantiating model {type(model)} with GNN representation {type(model.representation)}")

    # Save extra arguments in checkpoint
    task.save_configuration(config)
    
    # Initiate the training
    log.debug(f"Instantiating trainer <{config.trainer._target_}>")
    trainer = instantiate(config.trainer)
    # log.debug(f"Trainer callbacks: {str(callback for callback in trainer.callbacks)}")
    
    # wandb bug!!
    if isinstance(trainer.logger, WandbLogger):
        os.makedirs(trainer.logger.save_dir + '/wandb', exist_ok=True)

    # Train the model
    trainer.fit(model=task, datamodule=datamodule, ckpt_path=resume_ckpt)
    
    # Deploy model to a compiled model
    if config.deploy_model:
        # Load the model
        model_path, val_loss = find_best_model(run_path=config.run_path + '/model_path')
        
        # Compile the model
        log.debug(f"Deploy trained model from {model_path} with validation loss of {val_loss:.3f}")
        deploy(model_path, f"{config.run_path}/compiled_model.pt")
        log.debug(f"Deploying compiled model at <{config.run_path}/compiled_model.pt>")

# Training without Pytorch Lightning
@hydra.main(config_path="configs", config_name="train", version_base=None)
def tmp_train(config: DictConfig):
    """
    Train the model without pytorch lightning.

    Args:
        config (DictConfig): The configuration file.
    Returns:
        None
    
    """
    _ensure_resolvers()
    import torch
    from hydra.utils import instantiate
    from e3nn.util.jit import script
    from torch_ema import ExponentialMovingAverage
    from pytorch_lightning import seed_everything
    from .utils import EarlyStopping
    from .utils import read_user_config, normalize_config_sequences, CustomFormatter
    from curator.train import train

    # Load the arguments
    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="train")

    normalize_config_sequences(config)
    _prepare_run_path(config.run_path)

    # Save yaml file in run_path
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=False)
    log.debug("Running on host: " + str(socket.gethostname()))
    
    # Set up the seed
    if "seed" in config:
        log.debug(f"Seed with <{config.seed}>")
        seed_everything(config.seed, workers=True)
    else:
        log.debug("Seed randomly...")
    
    _configure_cli_logger(
        log,
        os.path.join(config.run_path, "training.log"),
        CustomFormatter(),
        stream=True,
    )
    
    # Set up datamodule and load training and validation set
    # Initiate the datamodule
    log.debug(f"Instantiating datamodule <{config.data._target_}> from dataset {config.data.datapath or config.data.train_path}")
    datamodule = hydra.utils.instantiate(config.data)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    # Set up the model, the optimizer and  the scheduler
    model = instantiate(config.model)
    model.initialize_modules(datamodule)
    outputs = instantiate(config.task.outputs)
    optimizer = instantiate(config.task.optimizer)(model.parameters())
    scheduler = instantiate(config.task.scheduler)(optimizer=optimizer)

    model = train(
        model=model, 
        outputs=outputs,
        optimizer=optimizer, 
        scheduler=scheduler, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader,
        device=config.device,
        num_epochs=config.trainer.max_epochs,
        log_frequency=config.trainer.log_every_n_steps,
    )
    
    # Deploy model
    if config.deploy_model:
        # Load the model
        model_path = [str(f) for f in Path(f"{config.run_path}").rglob("best_model.pth*")]
        if len(model_path) > 1:
            log.warning("Multiple best models found, using the last one.")
        model_path = model_path[-1]
        
        # Compile the model
        model = torch.load(model_path, map_location=torch.device(config.device))
        model_compiled = script(model)
        metadata = {"cutoff": str(model_compiled.representation.cutoff).encode("ascii")}
        model_compiled.save(f"{config.run_path}/compiled_model.pt", _extra_files=metadata)
        log.debug(f"Deploying compiled model at <{config.run_path}/compiled_model.pt>")

def _deploy_parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Script for deploy curator models",
        fromfile_prefix_chars="+",
    )
    parser.add_argument(
        "model_path",
        metavar="INPUT_FILE",
        type=str,
        nargs="+",
        help="Path(s) to model to be compiled",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default="compiled_model.pt",
        help="Path to save compiled model",
    )
    parser.add_argument(
        "--load_weights_only",
        action="store_true",
        help="Load trained weights while initializing the model",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        help="Configuration file that defines model parameters (optional)",
    )
    parser.add_argument(
        "--lammps",
        action="store_true",
        help="Export a LAMMPS MLIAP-ready model instead of torchscript",
    )
    parser.add_argument(
        "--element-types",
        type=str,
        nargs="+",
        default=None,
        help="Element symbols ordered as in the LAMMPS pair_style; required when --lammps-mliap is set",
    )
    if argcomplete:
        argcomplete.autocomplete(parser)
    return parser.parse_args(argv)

def deploy_main(argv: Optional[List[str]] = None):
    args = _deploy_parse_args(argv)
    return deploy(
        model_path=args.model_path,
        target_path=args.target_path,
        load_weights_only=args.load_weights_only,
        cfg_path=args.cfg_path,
        return_model=False,
        lammps_mliap=args.lammps,
        element_types=args.element_types,
    )

def _convert_parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Convert MACE or official NequIP checkpoints to Curator format, or upgrade Curator checkpoints",
        fromfile_prefix_chars="+",
    )
    parser.add_argument(
        "ckpt_path",
        metavar="INPUT_FILE",
        type=str,
        help="Path to a MACE or Curator checkpoint to convert",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output path for the converted checkpoint (default: alongside input with _converted suffix)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for loading the checkpoint; defaults to CPU so conversion works without GPUs",
    )
    parser.add_argument(
        "-u",
        "--update",
        action="store_true",
        help="Update an old CURATOR checkpoint by rebuilding the stored model.",
    )
    parser.add_argument(
        "--e3nn-to-cueq",
        action="store_true",
        help="Convert a Curator e3nn model checkpoint to cuequivariance backend.",
    )
    parser.add_argument(
        "--cueq-to-e3nn",
        action="store_true",
        help="Convert a Curator cuequivariance model checkpoint back to e3nn backend.",
    )
    parser.add_argument(
        "--mace-to-curator",
        action="store_true",
        help="Convert an original MACE checkpoint to a Curator MACE checkpoint.",
    )
    parser.add_argument(
        "--curator-to-mace",
        action="store_true",
        help="Convert a Curator MACE checkpoint back to an original MACE checkpoint.",
    )
    parser.add_argument(
        "--nequip-to-curator",
        action="store_true",
        help="Convert an official NequIP package/checkpoint (including nequip.net model refs) to a Curator checkpoint.",
    )
    if argcomplete:
        argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def _default_nequip_output_path(raw_input: str) -> Path:
    if raw_input.endswith(".nequip.zip"):
        input_path = Path(raw_input)
        base_name = input_path.name[:-len(".nequip.zip")]
        return input_path.with_name(f"{base_name}_curator.pth")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_input).strip("._")
    if not safe_name:
        safe_name = "nequip_model"
    return Path.cwd() / f"{safe_name}_curator.pth"

def convert_main(argv: Optional[List[str]] = None):
    args = _convert_parse_args(argv)
    from .utils import upgrade_checkpoint, convert_mace_to_curator

    device = args.device
    target = None

    if args.update:
        target = upgrade_checkpoint(
            ckpt_path=args.ckpt_path,
            output_path=args.output,
            device=device,
        )
    elif args.e3nn_to_cueq or args.cueq_to_e3nn:
        import torch
        from curator.utils import load_models, convert_e3nn_to_cueq, convert_cueq_to_e3nn
        from curator.layer._cuequivariance_wrapper import IS_CUET_AVAILABLE

        if args.cueq_to_e3nn and (not torch.cuda.is_available() or not IS_CUET_AVAILABLE):
            raise RuntimeError(
                "Converting from cueq to e3nn requires cuequivariance with CUDA. "
                "Please run on a CUDA-enabled environment with cuequivariance installed."
            )

        try:
            models = load_models(args.ckpt_path, device=torch.device(device), load_compiled=False)
        except PermissionError as e:
            raise RuntimeError(
                "Failed to load cueq checkpoint due to permission/CUDA issues. "
                "cuequivariance typically requires CUDA; please run on a CUDA-enabled setup "
                "with cuequivariance installed."
            ) from e
        except Exception as e:
            if args.cueq_to_e3nn:
                raise RuntimeError(
                    "Failed to load cueq checkpoint. Ensure cuequivariance is installed and CUDA is available."
                ) from e
            raise
        if len(models) != 1:
            raise ValueError("Cueq/e3nn conversion supports single-model checkpoints only.")
        model = models[0]

        if args.e3nn_to_cueq and args.cueq_to_e3nn:
            raise ValueError("Choose only one of --e3nn-to-cueq or --cueq-to-e3nn.")
        if args.e3nn_to_cueq:
            converted = convert_e3nn_to_cueq(model)
            suffix = "_cueq"
        else:
            converted = convert_cueq_to_e3nn(model)
            suffix = "_e3nn"

        ckpt_path = Path(args.ckpt_path)
        output_path = args.output
        if output_path is None:
            output_path = ckpt_path.with_name(f"{ckpt_path.stem}{suffix}{ckpt_path.suffix}")
        torch.save(converted, output_path)
        target = output_path
    elif args.nequip_to_curator:
        import torch
        from curator.utils import convert_nequip_to_curator

        output_path = Path(args.output) if args.output is not None else _default_nequip_output_path(args.ckpt_path)
        target = convert_nequip_to_curator(
            nequip_path=args.ckpt_path,
            output_path=output_path,
            device=torch.device(device),
        )
    elif args.mace_to_curator or args.curator_to_mace:
        import torch
        from curator.utils import convert_mace_to_curator, convert_curator_to_mace

        ckpt_path = Path(args.ckpt_path)
        output_path = args.output
        if output_path is None:
            suffix = "_mace" if args.curator_to_mace else "_converted"
            output_path = ckpt_path.with_name(f"{ckpt_path.stem}{suffix}{ckpt_path.suffix}")
        if args.curator_to_mace:
            target = convert_curator_to_mace(
                curator_path=ckpt_path,
                output_path=output_path,
                device=torch.device(device),
            )
        else:
            target = convert_mace_to_curator(
                mace_path=ckpt_path,
                output_path=output_path,
                device=torch.device(device),
            )
    else:
        import torch

        ckpt_path = Path(args.ckpt_path)
        output_path = args.output
        if output_path is None:
            output_path = ckpt_path.with_name(f"{ckpt_path.stem}_converted{ckpt_path.suffix}")
        target = convert_mace_to_curator(
            mace_path=ckpt_path,
            output_path=output_path,
            device=torch.device(device),
        )

    print(f"Converted checkpoint saved to {target}")

    return target

# Deploy the model and save a compiled model
def deploy(
        model_path: Union[str, list],
        target_path: str = 'compiled_model.pt',
        load_weights_only: bool=False,
        cfg_path: Optional[str] = None,
        return_model: bool = False,
        lammps_mliap: bool = False,
        element_types: Optional[List[str]] = None,
    ):
    """ Deploy the model and save a compiled model.

    Args:
        config (DictConfig): The configuration file.
    Returns:
        None
    
    """
    _ensure_resolvers()
    import torch
    from e3nn.util.jit import script
    from curator.model import EnsembleModel
    from curator.layer.utils import find_layer_by_name_recursive
    from .utils import read_user_config, normalize_config_sequences, load_models

    cfg = None
    if cfg_path is not None:
        cfg = read_user_config(cfg_path, config_path="configs", config_name="train")
        normalize_config_sequences(cfg)

    # Load model(s)
    models = load_models(
        model_path,
        device=None,
        load_compiled=False,
        load_weights_only=load_weights_only,
        cfg=cfg,
    )
    if len(models) > 1:
        model = EnsembleModel(models)
    else:
        model = models[0]

    if lammps_mliap:
        if not element_types:
            raise ValueError("element_types must be provided when exporting LAMMPS MLIAP models.")
        from curator.simulate.lammps_mliap_interface import LAMMPS_MLIAP
        lmp_model = LAMMPS_MLIAP(model, element_types)
        if target_path == 'compiled_model.pt':
            target_path = 'lmp_model.pt'
        torch.save(lmp_model, target_path)
        return lmp_model

    # Compile the model
    model_compiled = script(model)
    metadata = {"cutoff": str(find_layer_by_name_recursive(model_compiled, 'cutoff')).encode("ascii")}
    model_compiled.save(target_path, _extra_files=metadata)
    log.debug(f"Deploying compiled model at <{target_path}> from <{model_path}>")
    if return_model:
        return model_compiled

@hydra.main(config_path="configs", config_name="evaluate", version_base=None)
def evaluate(config: DictConfig):
    _ensure_resolvers()
    from hydra.utils import instantiate
    from .utils import read_user_config, prune_config_targets, load_models
    from curator.model import EnsembleModel
    from curator.simulate import MLCalculator
    import torch

    # Load the arguments
    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="evaluate")
    prune_config_targets(config, logger=log)
    if config.model_path is None or config.datapath is None:
        raise RuntimeError("Both model_path and datapath are required for evaluation.")

    _prepare_run_path(config.run_path)

    # Save yaml file in run_path
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=False)

    _configure_cli_logger(
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

    # Load model. Uses a compiled model, if any, otherwise a uncompiled model
    log.debug("Using model from <{}>".format(config.model_path))
    if config.deploy is not None:
        model = deploy(
            config.model_path, 
            load_weights_only=config.deploy.load_weights_only,
            return_model=True,
        )
    else:
        model = load_models(config.model_path, config.device)
        model = EnsembleModel(model) if len(model) > 1 else model[0]
    
    evaluator = instantiate(config.evaluator, model=model)
    evaluator.evaluate(config.datapath)


def evaluate_main(argv: Optional[List[str]] = None):
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
        if device is None:
            return None
        if not isinstance(device, str):
            return device
        if not device.startswith("cuda"):
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
        parser.add_argument(
            "dataset",
            nargs="?",
            help="Dataset path (extxyz/xyz/traj); optional if --data is set",
        )
        parser.add_argument(
            "model",
            nargs="?",
            help="Model checkpoint/run directory; optional if --model is set",
        )
        parser.add_argument(
            "-d",
            "--data",
            dest="datapath",
            nargs="+",
            help="Dataset path(s)",
        )
        parser.add_argument(
            "-m",
            "--model",
            dest="model_path",
            nargs="+",
            help="Model checkpoint or run directory path(s)",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            help="Device for evaluation (default: cuda)",
        )
        parser.add_argument(
            "--out",
            type=str,
            default="evaluate",
            help="Base output directory (default: ./evaluate)",
        )
        parser.add_argument(
            "--no-plot",
            action="store_true",
            help="Disable plotting",
        )
        parser.add_argument(
            "--save-data",
            action="store_true",
            help="Save raw predictions/targets to results.npz",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=8,
            help="Batch size for evaluation (default: 8)",
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            default=0,
            help="DataLoader workers (default: 0)",
        )
        parser.add_argument(
            "--pin-memory",
            action="store_true",
            help="Enable DataLoader pin_memory",
        )
        if argcomplete:
            argcomplete.autocomplete(parser)
        return parser.parse_args(args)

    if argv is None:
        argv = sys.argv[1:]
    if _is_hydra_args(argv):
        return evaluate()
    args = _parse_args(argv)

    _ensure_resolvers()
    from curator.simulate.evaluator import Evaluator
    from curator.model import EnsembleModel
    from .utils import load_models

    datapath = args.datapath or args.dataset
    model_path = args.model_path or args.model

    if datapath is None or model_path is None:
        raise RuntimeError("Both dataset and model are required for evaluation.")

    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)
    _configure_cli_logger(
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

# Simulate with the model
@hydra.main(config_path="configs", config_name="simulate", version_base=None)
def simulate(config: DictConfig):
    """ Simulate with the model.

    Args:
        config (DictConfig): The configuration file.
    Returns:
        None
    """
    _ensure_resolvers()
    from hydra.utils import instantiate
    from pytorch_lightning import seed_everything
    from .utils import read_user_config, normalize_config_sequences, prune_config_targets, log_logo, CustomFormatter
    from curator.simulate.sim_logging import log_simulation_summary
    from curator.model import EnsembleModel
    from curator.simulate import MLCalculator

    # Load the arguments
    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="simulate")
    else:
        normalize_config_sequences(config)
        prune_config_targets(config, logger=log)

    _prepare_run_path(config.run_path)
    
    # Save yaml file in run_path
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=False)
    
    _configure_cli_logger(
        log,
        os.path.join(config.run_path, "simulation.log"),
        CustomFormatter(),
        stream=True,
    )
    log_logo(log)

    # Brief simulation summary for easier debugging
    log.debug("Running on host: " + str(socket.gethostname()))
    log_simulation_summary(log, config)

    # Set up the seed
    if "seed" in config:
        log.debug(f"Seed with <{config.seed}>")
        seed_everything(config.seed, workers=True)
    else:
        log.debug("Seed randomly...")

    # Setup simulator
    simulator = instantiate(config.simulator)
    simulator.run()
    
@hydra.main(config_path="configs", config_name="select", version_base=None)   
def select(config: DictConfig):
    """ Select structures with active learning.

    Args:
        config (DictConfig): The configuration file.
    Returns:
        None
    """
    _ensure_resolvers()
    from hydra.utils import instantiate
    from pytorch_lightning import seed_everything
    from omegaconf import OmegaConf
    from curator.select import GeneralActiveLearning
    from .utils import read_user_config, load_models, log_logo

    # Load the arguments
    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="select")

    _prepare_run_path(config.run_path)

    _configure_cli_logger(
        log,
        os.path.join(config.run_path, "selection.log"),
        logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"),
        stream=True,
    )
    log_logo(log)

    # Save yaml file in run_path
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=False)
    log.debug("Running on host: " + str(socket.gethostname()))

    legacy_fields = []
    for field in ("kernel", "export_kernels", "n_random_features"):
        if field in config:
            legacy_fields.append(field)
    if legacy_fields:
        raise RuntimeError(
            "The projector/kernel selection interface has been removed. "
            f"Replace {legacy_fields} with 'feature_specs' and optional 'selection_feature'."
        )

    # Set up the seed
    if "seed" in config:
        log.debug(f"Seed with <{config.seed}>")
        seed_everything(config.seed, workers=True)
    else:
        log.debug("Seed randomly...")
    
    # Load model and datasets
    log.debug("Using model from <{}>".format(config.model_path))
    models = load_models(config.model_path, config.device, load_compiled=False)
    cutoff = models[0].representation.cutoff

    transforms = instantiate(config.transforms) if config.transforms else []
    pool_source = None
    if config.data_url is not None:
        if isinstance(config.data_url, str):
            pool_source = config.data_url
        else:
            data_url = dict(config.data_url)
            if "url" not in data_url:
                raise RuntimeError("data_url must include a 'url' field.")
            pool_source = data_url["url"]
            if len(data_url) > 1:
                log.warning("data_url options are ignored; use pool_set with a URL string instead.")
    elif config.pool_set:
        pool_source = config.pool_set
    if pool_source is None:
        raise RuntimeError("pool_set or data_url is required for selection.")

    feature_specs = OmegaConf.select(config, "feature_specs")
    if not feature_specs:
        raise RuntimeError("feature_specs must be provided for selection.")
    feature_specs = OmegaConf.to_container(feature_specs, resolve=True)
    if not isinstance(feature_specs, list):
        raise RuntimeError("feature_specs must be a list of feature spec mappings.")
    selection_feature = OmegaConf.select(config, "selection_feature")

    select_batch_size = OmegaConf.select(config, "select_batch_size") or OmegaConf.select(config, "batch_size") or 100
    data_batch_size = OmegaConf.select(config, "data_batch_size") or select_batch_size
    save_features = None
    if config.save_features:
        if isinstance(config.save_features, str):
            save_features = config.save_features
        else:
            save_features = os.path.join(config.run_path, "features.h5")
    save_selected_features = None
    if getattr(config, "save_selected_features", None):
        if isinstance(config.save_selected_features, str):
            save_selected_features = config.save_selected_features
        else:
            save_selected_features = os.path.join(config.run_path, "selected_features.h5")
    save_images = None
    if config.save_images:
        if isinstance(config.save_images, str):
            save_images = config.save_images
        else:
            save_images = os.path.join(config.run_path, "selected.traj")

    # Select structures based on the active learning method
    al = GeneralActiveLearning(
        models=models,
        feature_specs=feature_specs,
        selection_feature=selection_feature,
        selection=config.method,
        target_layer=OmegaConf.select(config, "target_layer", default="readout_mlp"),
        batch_size=data_batch_size,
        device=config.device,
        dataset_cutoff=cutoff,
        transforms=transforms,
        save_features=save_features,
        target_domain=OmegaConf.select(config, "target_domain"),
    )
    save_json = os.path.join(config.run_path, "selected.json")
    indices = al.select(
        pool_set=pool_source,
        train_set=config.train_set,
        select_batch_size=select_batch_size,
        save_json=save_json,
        save_images=save_images,
        save_selected_features=save_selected_features,
        normalize_features=OmegaConf.select(config, "export_normalized_features", default=True),
        compute_features_only=bool(OmegaConf.select(config, "compute_features_only", default=False)),
    )

    log.debug(
        "Active learning selection completed! Check %s for %d selected structures!",
        os.path.abspath(save_json),
        len(indices),
    )

# Label the dataset selected by active learning
@hydra.main(config_path="configs", config_name="label", version_base=None)   
def label(config: DictConfig):
    """ Label the dataset selected by active learning.

    Args:
        config (DictConfig): The configuration file.
    Returns:
        None
    """
    _ensure_resolvers()
    from hydra.utils import instantiate
    from .utils import read_user_config, log_logo
    from curator.data import read_trajectory
    from ase.db import connect
    from ase.io import Trajectory
    import json
    import numpy as np
    from curator.label import AtomsAnnotator
    from shutil import copy

    # Load the arguments
    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="label")

    _prepare_run_path(config.run_path)

    _configure_cli_logger(
        log,
        os.path.join(config.run_path, "labelling.log"),
        logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"),
        stream=True,
    )
    log_logo(log)

    # Save yaml file in run_path
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=False)
    log.debug("Running on host: " + str(socket.gethostname()))

    # get images and set parameters
    if config.pool_set:
        images = read_trajectory(config.pool_set)
        # Use active learning indices if provided
        indices = config.indices
        if config.al_info:
            with open(config.al_info) as f:
                indices = json.load(f)["selected"]
                log.debug(f"Labelling {len(indices)} active learning selected structures: {config.al_info}")
        elif indices is not None:
            log.debug(f"Labelling {len(indices)} selected structures: {config.indices}")
        
        images = [images[i] for i in indices] if indices is not None else [atoms for atoms in images]
    else:
        raise RuntimeError('Valid configarations for DFT calculation should be provided!')
    
    # split jobs if needed to accelerate labelling if you have a lot of resources
    if config.split_jobs or config.imgs_per_job:
        from .utils import split_list
        if config.split_jobs:
            images = split_list(images, config.split_jobs)
        if config.imgs_per_job:
            images = split_list(images, config.imgs_per_job, by_chunk_size=True)
        images = images[config.job_order]          # specify which parts of the images to label
        log.debug(f"Rank {config.job_order}. Total structures: {len(images)}")

    # create or read existing ase database
    db = connect(config.run_path+'/dft_structures.db')
    db.metadata ={
        'path': config.run_path+'/dft_structures.db',
    }

    # Set up calculator
    annotator = instantiate(config.annotator)
    
    # Label the structures
    all_converged = []
    for i, atoms in enumerate(images):
        # Label the structure with the choosen method
        log.debug(f"Labeling structure {i}.")
        try:
            existing_converged = db[i+1].get('converged')
            if not existing_converged:
                converged = annotator.annotate(atoms)
                db.update(id=i+1, atoms=atoms, converged=converged)
                log.debug(f"Recomputing structure {i} converged: {converged}")
            else:
                converged = existing_converged
                log.debug(f"Structure {i} converged. Skipping...")
            all_converged.append(converged)
        except KeyError:
            converged = annotator.annotate(atoms)
            db.write(atoms, converged=converged)
            all_converged.append(converged)
        
        # TODO: add this feature into annotator
        # copy files
        if os.path.exists('OSZICAR') and (not os.path.exists(f'OSZICAR_{i}') or not converged):
            copy('OSZICAR', f'OSZICAR_{i}')
        if os.path.exists('vasp.out') and (not os.path.exists(f'vasp.out_{i}') or not converged):
            copy('vasp.out', f'vasp.out_{i}')
    
    # write to datapath
    if config.datapath is not None:
        log.debug(f"Write atoms to {config.datapath}.") 
        total_dataset = Trajectory(config.datapath, 'a')
        for row in db.select(converged=True):
            if row.get('stored'):
                log.debug(f"Structure {row.id - 1} is already stored in <{config.datapath}>. Skipping...")
            else:
                db.update(id=row.id, stored=True)
                log.debug(f"Write structure {row.id - 1} to <{config.datapath}>")
                total_dataset.write(row.toatoms())
    
    if not all(all_converged):
        raise RuntimeError(f'Structures {[row.id -1 for row in db.select(converged=False)]} are not converged!')
    else:
        # sweep all unnessary files after labeling
        annotator.sweep()
