# General modules for all tasks
from hydra.utils import instantiate
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf, open_dict
import sys, os, json
from pathlib import Path
import argparse
try:
    import argcomplete
except ImportError:  # pragma: no cover
    argcomplete = None

import pytorch_lightning.callbacks
import pytorch_lightning.loggers
from .utils import (
    read_user_config,
    CustomFormatter,
    register_resolvers,
    find_best_model,
    load_models,
    upgrade_checkpoint,
    convert_mace_to_curator,
    normalize_config_sequences,
    prune_config_targets,
)
import logging
import socket
import contextlib
from typing import Optional, Union, Dict, List
from pytorch_lightning import seed_everything
from curator.simulate.sim_logging import log_simulation_summary

# very ugly solution for solving pytorch lighting and myqueue conflictions
if "SLURM_NTASKS" in os.environ:
    del os.environ["SLURM_NTASKS"]
if "SLURM_JOB_NAME" in os.environ:
    del os.environ["SLURM_JOB_NAME"]

# Set up logger for the different tasks 
log = logging.getLogger('curator')
log.setLevel(logging.DEBUG)

# register omegaconf resolvers
register_resolvers()


# Trainining with Pytorch Lightning (only with weights and biasses)
@hydra.main(config_path="configs", config_name="train", version_base=None)
def train(config: DictConfig) -> None:
    """ Train the model with pytorch lightning.
    
    Args:
        config (DictConfig): The configuration file.
    Returns:
        None

    """
    import torch
    import pytorch_lightning
    from pytorch_lightning import (
    LightningDataModule, 
    Trainer,
    )
    from curator.model import LitNNP
    from e3nn.util.jit import script

    # set up logger
    local_rank = os.environ.get("LOCAL_RANK")
    log_name = "training.log" if local_rank in (None, "0") else f"training_rank{local_rank}.log"
    fh = logging.FileHandler(os.path.join(config.run_path, log_name), mode="w")
    fh.setFormatter(CustomFormatter())
    log.addHandler(fh)
    
    # Load the arguments 
    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="train")

    normalize_config_sequences(config)
    prune_config_targets(config, logger=log)
    prune_config_targets(config, logger=log)

    # Save yaml file in run_path
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=False)
    log.debug("Running on host: " + str(socket.gethostname()))
    
    # Set up seed
    if "seed" in config:
        log.debug(f"Seed with <{config.seed}>")
        seed_everything(config.seed, workers=True)
    else:
        log.debug("Seed randomly...")
    
    # Initiate the datamodule
    log.debug(f"Instantiating datamodule <{config.data._target_}> from dataset {config.data.datapath or config.data.train_path}")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.data)
    datamodule.setup()
    # something must be inferred from data before instantiating the model
    if datamodule.species == 'auto':
        config.data.species = datamodule._get_species()

    model = hydra.utils.instantiate(config.model)
    resume_ckpt = None
    checkpoint_outputs = None

    if config.model_path is not None:
        config.model_path = find_best_model(config.model_path)[0]
        log.debug(f"Loading trained model from {config.model_path}")
        if config.task.load_entire_model:
            state_dict = torch.load(config.model_path)
            model = state_dict['model']
            checkpoint_outputs = state_dict.get('outputs')
        elif config.task.load_weights_only:
            from collections import OrderedDict
            state_dict = torch.load(config.model_path)
            new_state_dict = OrderedDict((key.replace('model.', ''), value) for key, value in state_dict['state_dict'].items())
            model.load_state_dict(new_state_dict, strict=False)
        else:
            resume_ckpt = config.model_path
            log.debug(
                "Resuming training from checkpoint; optimizer and scheduler states will be restored from %s",
                resume_ckpt,
            )

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
    trainer: Trainer = hydra.utils.instantiate(config.trainer)
    # log.debug(f"Trainer callbacks: {str(callback for callback in trainer.callbacks)}")
    
    # wandb bug!!
    if isinstance(trainer.logger, pytorch_lightning.loggers.WandbLogger):
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
    import torch
    from e3nn.util.jit import script
    from torch_ema import ExponentialMovingAverage
    from .utils import EarlyStopping
    from curator.train import train

    # Load the arguments
    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="train")

    normalize_config_sequences(config)

    # Save yaml file in run_path
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=False)
    log.debug("Running on host: " + str(socket.gethostname()))
    
    # Set up the seed
    if "seed" in config:
        log.debug(f"Seed with <{config.seed}>")
        seed_everything(config.seed, workers=True)
    else:
        log.debug("Seed randomly...")
    
    # Setup the logger
    # set up logger
    fh = logging.FileHandler(os.path.join(config.run_path, "training.log"), mode="w")
    fh.setFormatter(CustomFormatter())
    log.addHandler(fh)
    
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
    ):
    """ Deploy the model and save a compiled model.

    Args:
        config (DictConfig): The configuration file.
    Returns:
        None
    
    """
    import torch
    from e3nn.util.jit import script
    from curator.model import EnsembleModel
    from curator.layer.utils import find_layer_by_name_recursive

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
    from curator.model import EnsembleModel
    from curator.simulate import MLCalculator

    # Load the arguments
    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="evaluate")
    prune_config_targets(config, logger=log)

    # Save yaml file in run_path
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=False)

    # set logger
    fh = logging.FileHandler(os.path.join(config.run_path, "predict.log"), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)
    log.debug("Running on host: " + str(socket.gethostname()))

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

# Simulate with the model
@hydra.main(config_path="configs", config_name="simulate", version_base=None)
def simulate(config: DictConfig):
    """ Simulate with the model.

    Args:
        config (DictConfig): The configuration file.
    Returns:
        None
    """
    from curator.model import EnsembleModel
    from curator.simulate import MLCalculator

    # Load the arguments
    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="simulate")
    else:
        normalize_config_sequences(config)
        prune_config_targets(config, logger=log)
    
    # Ensure run_path exists before writing any artifacts
    Path(config.run_path).mkdir(parents=True, exist_ok=True)

    # Save yaml file in run_path
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=False)
    
    # set logger
    # set up logger
    fh = logging.FileHandler(os.path.join(config.run_path, "simulation.log"), mode="w")
    fh.setFormatter(CustomFormatter())
    log.addHandler(fh)
    # mirror to stdout for live feedback
    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout for h in log.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(CustomFormatter())
        log.addHandler(sh)

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
    from curator.data import read_trajectory
    import torch
    from ase.io import read, Trajectory
    from omegaconf import OmegaConf
    from curator.select import GeneralActiveLearning
    import json
    from curator.data import AseDataset

    # Load the arguments
    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="select")
    
    prune_config_targets(config, logger=log)

    # set up logger
    # set logger
    fh = logging.FileHandler(os.path.join(config.run_path, "selection.log"), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)

    # Save yaml file in run_path
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=False)
    log.debug("Running on host: " + str(socket.gethostname()))

    # Set up the seed
    if "seed" in config:
        log.debug(f"Seed with <{config.seed}>")
        seed_everything(config.seed, workers=True)
    else:
        log.debug("Seed randomly...")
    
    # Set up datamodule and load training and validation set
    # The active learning only works for uncompiled model at the moment
    log.debug("Using model from <{}>".format(config.model_path))
    models = load_models(config.model_path, config.device, load_compiled=False)
    cutoff = models[0].representation.cutoff

    # Load the pool data set and training data set
    if config.dataset and config.split_file:
        dataset = AseDataset(read_trajectory(config.dataset), cutoff=cutoff, transforms=instantiate(config.transforms))
        with open(config.split_file) as f:
            split = json.load(f)
        data_dict = {}
        for k in split:
            data_dict[k] = torch.utils.data.Subset(dataset, split[k])
    elif config.pool_set:
        data_dict = {'pool': AseDataset(read_trajectory(config.pool_set), cutoff=cutoff, transforms=instantiate(config.transforms))}
        if config.train_set:
            data_dict["train"] = AseDataset(read_trajectory(config.train_set), cutoff=cutoff, transforms=instantiate(config.transforms))
    else:
        raise RuntimeError("Please give valid pool data set for selection!")


    # Check the size of pool data set
    if len(data_dict['pool']) < config.batch_size * 10: 
            log.warning(f"The pool data set ({len(data_dict['pool'])}) is not large enough for selection! " 
                + f"It should be larger than 10 times batch size ({config.batch_size*10}). "
                + "Check your simulation!")
    elif len(data_dict['pool']) < config.batch_size:
        raise RuntimeError(f"""The pool data set ({len(data_dict['pool'])}) is not large enough for selection! Add more data or change batch size {config.batch_size}.""")

    # Select structures based on the active learning method
    al = GeneralActiveLearning(
        kernel=config.kernel, 
        selection=config.method, 
        n_random_features=config.n_random_features,
        save_features=config.save_features,
        use_faiss=getattr(config, 'use_faiss', False),
    )
    indices = al.select(models, data_dict, al_batch_size=config.batch_size, debug=config.debug)

    # Save the selected indices
    datapath = config.dataset if config.dataset and config.split_file else config.pool_set
    datapath = datapath if isinstance(datapath, str) else list(datapath)
    al_info = {
        'kernel': config.kernel,
        'selection': config.method,
        'dataset': datapath,
        'selected': indices,
    }
    with open(config.run_path+'/selected.json', 'w') as f:
        json.dump(al_info, f)
    
    log.debug(f"Active learning selection completed! Check {os.path.abspath(config.run_path+'/selected.json')} for selected structures!")
    if config.save_images:
        pool_set = read_trajectory(config.pool_set)
        selected_images = [pool_set[i] for i in indices]
        save_path = config.save_images if isinstance(config.save_images, str) else os.path.join(config.run_path, 'selected.traj')
        with Trajectory(config.save_images if isinstance(config.save_images, str) else 'selected.traj', 'w') as traj:
            for atoms in selected_images:
                traj.write(atoms)
        log.debug(f"Saving selected images into {save_path}.")

# Label the dataset selected by active learning
@hydra.main(config_path="configs", config_name="label", version_base=None)   
def label(config: DictConfig):
    """ Label the dataset selected by active learning.

    Args:
        config (DictConfig): The configuration file.
    Returns:
        None
    """
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

    # set up logger
    # set logger
    fh = logging.FileHandler(os.path.join(config.run_path, "labelling.log"), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)

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
