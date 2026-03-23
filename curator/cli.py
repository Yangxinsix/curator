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
from typing import Optional, Union, Dict, List, Any, Mapping

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


def _plain_uncertainty_spec(spec: Optional[Any]) -> Optional[dict]:
    if spec is None:
        return None
    if isinstance(spec, DictConfig):
        spec = OmegaConf.to_container(spec, resolve=False)
    if not isinstance(spec, Mapping):
        raise TypeError(f"deploy.uncertainty must be a mapping, got {type(spec)}")
    return dict(spec)


def _default_uncertainty_spec(method: str, *, lammps_mliap: bool) -> dict:
    method = str(method).strip().lower()
    if method in ("", "none", "null"):
        return {"method": "none"}
    if method == "ensemble":
        return {"method": "ensemble", "output_keys": None}
    if method == "mahalanobis":
        # TorchScript pair_curator needs a scriptable kernel by default.
        default_kernel = "local-full-g" if lammps_mliap else "local-gnn"
        return {
            "method": "mahalanobis",
            "dataset": None,
            "output_keys": None,
            "maha": {
                "kernel": default_kernel,
                "max_structures": None,
                "regularization": 1e-6,
                "streaming": False,
            },
        }
    raise ValueError(f"Unknown uncertainty preset '{method}'.")


def _merge_uncertainty_specs(base: Optional[Any], override: Optional[Any]) -> Optional[dict]:
    base_plain = _plain_uncertainty_spec(base) or {}
    override_plain = _plain_uncertainty_spec(override) or {}
    if not base_plain and not override_plain:
        return None
    if str(override_plain.get("method", "")).strip().lower() in ("none", "null"):
        return {"method": "none"}

    merged = dict(base_plain)
    for key, value in override_plain.items():
        if value is None:
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            nested = dict(merged[key])
            for nested_key, nested_value in value.items():
                if nested_value is not None:
                    nested[nested_key] = nested_value
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _build_deploy_uncertainty_spec(
    *,
    method: Optional[str] = None,
    dataset: Optional[str] = None,
    lammps_mliap: bool = False,
    allow_partial: bool = False,
) -> Optional[dict]:
    if method is None:
        if dataset is None:
            return None
        if not allow_partial:
            raise ValueError(
                "--dataset requires --uncertainty mahalanobis "
                "or deploy.uncertainty.method=mahalanobis in cfg."
            )
        return {"dataset": dataset}

    cli_method = str(method).strip().lower()
    if cli_method in ("", "none", "null"):
        return {"method": "none"}

    spec = _default_uncertainty_spec(cli_method, lammps_mliap=lammps_mliap)

    spec["method"] = str(spec.get("method", "none")).strip().lower()

    if dataset is not None:
        spec["dataset"] = dataset

    if spec["method"] == "mahalanobis" and spec.get("dataset") in (None, "", "none", "null"):
        raise ValueError(
            "Mahalanobis deploy needs a reference dataset. "
            "Pass --dataset or set deploy.uncertainty.dataset in cfg."
        )

    return spec


def _resolve_default_deploy_target_path(
    target_path: str,
    *,
    lammps_mliap: bool,
) -> str:
    if lammps_mliap and target_path == "compiled_model.pt":
        return "mliap_model.pt"
    return target_path

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
        log.debug(f"Loading trained model from {config.model_path}")
        if config.task.load_entire_model:
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
            model = instantiate(config.model)
            if config.task.load_weights_only:
                from collections import OrderedDict
                state_dict = torch.load(config.model_path)
                if isinstance(state_dict, torch.nn.Module):
                    state_dict = {"state_dict": state_dict.state_dict()}
                new_state_dict = OrderedDict((key.replace('model.', ''), value) for key, value in state_dict['state_dict'].items())
                model.load_state_dict(new_state_dict, strict=False)
            else:
                resume_ckpt = config.model_path
                log.debug(
                    "Resuming training from checkpoint; optimizer and scheduler states will be restored from %s",
                    resume_ckpt,
                )
    else:
        model = instantiate(config.model)

    init_from = getattr(config.task, "init_new_domains_from", None)
    if domain_mode in ("extend", "replace") and new_domains is not None:
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
        best_model = find_best_model(run_path=config.run_path + '/model_path')
        if best_model is None:
            if getattr(trainer, "fast_dev_run", False):
                log.warning("Skipping deploy because fast_dev_run does not write checkpoints.")
            else:
                log.warning("Skipping deploy because no checkpoint was written to <%s/model_path>.", config.run_path)
        else:
            model_path, val_loss = best_model
            if val_loss is None:
                log.debug(f"Deploy trained model from {model_path}")
            else:
                log.debug(f"Deploy trained model from {model_path} with validation loss of {val_loss:.3f}")
            deploy(
                model_path,
                f"{config.run_path}/compiled_model.pt",
                uncertainty_spec=OmegaConf.select(config, "deploy.uncertainty", default=None),
            )
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
        if not model_path:
            log.warning("Skipping deploy because no best_model.pth checkpoint was written to <%s>.", config.run_path)
            return
        if len(model_path) > 1:
            log.warning("Multiple best models found, using the last one.")
        model_path = model_path[-1]

        deploy(
            model_path,
            f"{config.run_path}/compiled_model.pt",
            uncertainty_spec=OmegaConf.select(config, "deploy.uncertainty", default=None),
        )
        log.debug(f"Deploying compiled model at <{config.run_path}/compiled_model.pt>")

def _deploy_parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Deploy CURATOR checkpoint(s) to either a TorchScript model for "
            "pair_style curator or a Python-backed model for pair_style mliap unified."
        ),
        epilog=(
            "Examples:\n"
            "  pair_style curator:\n"
            "    python curator/deploy.py model.ckpt --target_path compiled_model.pt\n"
            "\n"
            "  mliap:\n"
            "    python curator/deploy.py model.ckpt --mliap \\\n"
            "      --element-types Fe Li O P --target_path mliap_model.pt\n"
            "\n"
            "  mliap + mahalanobis:\n"
            "    python curator/deploy.py model.ckpt --mliap \\\n"
            "      --element-types Fe Li O P --uncertainty mahalanobis \\\n"
            "      --dataset reference.traj --target_path mliap_model.pt\n"
            "\n"
            "  ensemble:\n"
            "    python curator/deploy.py ckpt1.ckpt ckpt2.ckpt ckpt3.ckpt \\\n"
            "      --uncertainty ensemble --target_path compiled_ensemble.pt\n"
            "\n"
            "Notes:\n"
            "  - passing multiple INPUT_FILE values creates an EnsembleModel\n"
            "  - --mliap requires --element-types\n"
            "  - --dataset is only needed for Mahalanobis\n"
            "  - use --cfg_path for advanced deploy.uncertainty settings"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars="+",
    )
    parser.add_argument(
        "model_path",
        metavar="INPUT_FILE",
        type=str,
        nargs="+",
        help="One or more checkpoint/model paths to export",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default="compiled_model.pt",
        help="Output path for the exported model; if left unchanged, --mliap rewrites compiled_model.pt to mliap_model.pt",
    )
    parser.add_argument(
        "--load_weights_only",
        action="store_true",
        help="Rebuild the model from config/checkpoint metadata and load weights only",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        help="Optional config file; use deploy.uncertainty there for detailed deploy tuning",
    )
    parser.add_argument(
        "--uncertainty",
        type=str,
        choices=["none", "ensemble", "mahalanobis"],
        default=None,
        help="Convenience uncertainty preset; keep detailed tuning in cfg_path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Reference dataset for Mahalanobis fitting; not needed for ensemble",
    )
    parser.add_argument(
        "--mliap",
        action="store_true",
        help="Export an mliap unified model instead of TorchScript pair_style curator output",
    )
    parser.add_argument(
        "--element-types",
        type=str,
        nargs="+",
        default=None,
        help="Element symbols in LAMMPS type order; required when --mliap is set",
    )
    if argcomplete:
        argcomplete.autocomplete(parser)
    return parser.parse_args(argv)

def deploy_main(argv: Optional[List[str]] = None):
    args = _deploy_parse_args(argv)
    target_path = _resolve_default_deploy_target_path(
        args.target_path,
        lammps_mliap=args.mliap,
    )
    uncertainty_spec = _build_deploy_uncertainty_spec(
        method=args.uncertainty,
        dataset=args.dataset,
        lammps_mliap=args.mliap,
        allow_partial=bool(args.cfg_path),
    )
    model = deploy(
        model_path=args.model_path,
        target_path=target_path,
        load_weights_only=args.load_weights_only,
        cfg_path=args.cfg_path,
        return_model=False,
        lammps_mliap=args.mliap,
        element_types=args.element_types,
        uncertainty_spec=uncertainty_spec,
    )
    export_kind = "mliap" if args.mliap else "torchscript"
    print(f"Deploy succeeded: type={export_kind} output={target_path}")
    return model

def _convert_parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Convert original MACE checkpoints to Curator format or upgrade Curator checkpoints",
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
    if argcomplete:
        argcomplete.autocomplete(parser)

    return parser.parse_args(argv)

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
        uncertainty_spec: Optional[dict] = None,
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
    from curator.layer import PairwiseDistance
    from curator.simulate.uncertainty._deploy import prepare_deploy_uncertainty
    from curator.layer.utils import find_layer_by_name_recursive
    from .utils import read_user_config, normalize_config_sequences, load_models

    def disable_internal_neighborlist(model_obj) -> int:
        disabled = 0
        for module in model_obj.modules():
            if isinstance(module, PairwiseDistance):
                module.compute_neighbor_list = False
                module.batch_nl = None
                module.compute_distance_from_R = False
                module.compute_forces = True
                disabled += 1
        return disabled

    target_path = _resolve_default_deploy_target_path(
        target_path,
        lammps_mliap=lammps_mliap,
    )

    cfg = None
    if cfg_path is not None:
        cfg = read_user_config(cfg_path, config_path="configs", config_name="train")
        normalize_config_sequences(cfg)
        cfg_uncertainty_spec = OmegaConf.select(cfg, "deploy.uncertainty", default=None)
        uncertainty_spec = _merge_uncertainty_specs(cfg_uncertainty_spec, uncertainty_spec)

    # Load model(s)
    models = load_models(
        model_path,
        device=None,
        load_compiled=False,
        load_weights_only=load_weights_only,
        cfg=cfg,
    )
    uncertainty_method = "none"
    if uncertainty_spec is not None and hasattr(uncertainty_spec, "get"):
        uncertainty_method = str(uncertainty_spec.get("method", "none")).strip().lower()
    if len(models) > 1:
        model = EnsembleModel(
            models,
            per_atom_uncertainty=bool(lammps_mliap or uncertainty_method == "ensemble"),
        )
    else:
        model = models[0]
    if uncertainty_spec is not None and str(uncertainty_spec.get("method", "none")).strip().lower() not in ("none", "", "null"):
        log.info("Preparing deploy uncertainty via unified registry: %s", uncertainty_spec.get("method"))

    disabled_neighborlist_modules = disable_internal_neighborlist(model)

    if lammps_mliap:
        prepare_deploy_uncertainty(
            model,
            uncertainty_spec,
            lammps_mliap=True,
        )
        if not element_types:
            raise ValueError("element_types must be provided when exporting LAMMPS MLIAP models.")
        from curator.simulate.lammps_mliap_interface import LAMMPS_MLIAP
        lmp_model = LAMMPS_MLIAP(model, element_types)
        torch.save(lmp_model, target_path)
        if disabled_neighborlist_modules:
            log.info(
                "Disabled internal PairwiseDistance neighbor-list construction in %d module(s) before LAMMPS MLIAP export.",
                disabled_neighborlist_modules,
            )
        return lmp_model

    # TorchScript struggles with dynamic ModuleDict logic in MultiDomain readout
    # when there is only one domain; collapse to the single AtomwiseNN.
    readout = getattr(getattr(model, "representation", None), "readout", None)
    if hasattr(readout, "domain_modules"):
        domain_modules = list(readout.domain_modules.values())
        if len(domain_modules) == 1:
            model.representation.readout = domain_modules[0]

    # Collapse single-domain MultiDomainRescaleShift into GlobalRescaleShift.
    if hasattr(model, "output_modules"):
        for i, module in enumerate(model.output_modules):
            if hasattr(module, "domain_modules"):
                domain_modules = list(module.domain_modules.values())
                if len(domain_modules) == 1:
                    model.output_modules[i] = domain_modules[0]

    if disabled_neighborlist_modules:
        log.info(
            "Disabled internal PairwiseDistance neighbor-list construction in %d module(s) before deploy.",
            disabled_neighborlist_modules,
        )

    prepare_deploy_uncertainty(
        model,
        uncertainty_spec,
        lammps_mliap=False,
    )

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
            uncertainty_spec=OmegaConf.select(config, "deploy.uncertainty", default=None),
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
        kernel=config.kernel, 
        kernels=OmegaConf.select(config, "export_kernels"),
        selection=config.method, 
        n_random_features=config.n_random_features,
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
