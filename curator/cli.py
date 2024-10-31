# General modules for all tasks
from hydra.utils import instantiate
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf, open_dict
import sys, os, json
from pathlib import Path

import pytorch_lightning.callbacks
import pytorch_lightning.loggers
from .utils import (
    read_user_config, 
    CustomFormatter, 
    register_resolvers,
    find_best_model,
)
import logging
import socket
import contextlib
from typing import Optional
from pytorch_lightning import seed_everything

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
    fh = logging.FileHandler(os.path.join(config.run_path, "training.log"), mode="w")
    fh.setFormatter(CustomFormatter())
    log.addHandler(fh)
    
    # Load the arguments 
    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="train")

    # Save yaml file in run_path
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=True)
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

    # TODO: enable loading existing optimizers and schedulers
    if config.model_path is not None:
        # When using CSVlogger and save_hyperparameters() together, the code will report pickle error.
        # So we choose to save the entire model and outputs in LitNNP and then reload it
        config.model_path = find_best_model(config.model_path)[0]
        log.debug(f"Loading trained model from {config.model_path}")
        # load model or model state dict
        if config.task.load_entire_model:
            state_dict = torch.load(config.model_path)
            model = state_dict['model']
            outputs = state_dict.get('outputs', instantiate(config.task.outputs))
        else:
            from collections import OrderedDict
            state_dict = torch.load(config.model_path)
            new_state_dict = OrderedDict((key.replace('model.', ''), value) for key, value in state_dict['state_dict'].items())
            model = instantiate(config.model)
            model.load_state_dict(new_state_dict)
            outputs = instantiate(config.task.outputs)
            
        log.debug(f"Instantiating task <{config.task._target_}>")
        # load optimizers and schedulers or not
        if config.task.load_weights_only:
            task = instantiate(config.task, model=model)
        else:
            task = LitNNP.load_from_checkpoint(
                checkpoint_path=config.model_path, 
                model=model, 
                outputs=outputs,
            )
    else:
        # Initiate the model
        model = hydra.utils.instantiate(config.model)
        # Initiate the task and load old model, if any
        log.debug(f"Instantiating task <{config.task._target_}>")
        task: LitNNP = hydra.utils.instantiate(config.task, model=model)

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
    trainer.fit(model=task, datamodule=datamodule)
    
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

    # Save yaml file in run_path
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=True)
    log.info("Running on host: " + str(socket.gethostname()))
    
    # Set up the seed
    if "seed" in config:
        log.info(f"Seed with <{config.seed}>")
        seed_everything(config.seed, workers=True)
    else:
        log.info("Seed randomly...")
    
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
        log.info(f"Deploying compiled model at <{config.run_path}/compiled_model.pt>")

# Deploy the model and save a compiled model
def deploy(
        model_path: str, 
        compiled_model_path: str = 'compiled_model.pt', 
        cfg_path: Optional[str] = None
    ):
    """ Deploy the model and save a compiled model.

    Args:
        config (DictConfig): The configuration file.
    Returns:
        None
    
    """
    import torch
    from curator.model import LitNNP
    from e3nn.util.jit import script

    # Load model
    loaded_model = torch.load(model_path, map_location="cpu" if not torch.cuda.is_available() else "cuda")

    if 'model' in loaded_model and cfg_path is None:
        model = loaded_model['model']
    else:
        # Load the arguments
        config = read_user_config(cfg_path, config_path="configs", config_name="train")

        # Set up datamodule and load training and validation set
        if not os.path.isfile(config.data.datapath):
            raise RuntimeError("Please provide valid data path!")
        datamodule = instantiate(config.data)
        datamodule.setup()
        
        # Set up model, optimizer and scheduler
        model = instantiate(config.model)
        model.initialize_modules(datamodule)
        model.load_state_dict(loaded_model["state_dict"])

    # Compile the model
    model_compiled = script(model)
    metadata = {"cutoff": str(model_compiled.representation.cutoff).encode("ascii")}
    model_compiled.save(compiled_model_path, _extra_files=metadata)
    log.info(f"Deploying compiled model at <{compiled_model_path}>")

# Simulate with the model
@hydra.main(config_path="configs", config_name="simulate", version_base=None)
def simulate(config: DictConfig):
    """ Simulate with the model.

    Args:
        config (DictConfig): The configuration file.
    Returns:
        None
    """
    from .utils import load_models
    from curator.model import EnsembleModel
    from curator.simulate import MLCalculator

    # Load the arguments
    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="simulate")
    
    # Save yaml file in run_path
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=True)
    
    # set logger
    fh = logging.FileHandler(os.path.join(config.run_path, "simulation.log"), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)
    log.info("Running on host: " + str(socket.gethostname()))

    # Set up the seed
    if "seed" in config:
        log.info(f"Seed with <{config.seed}>")
        seed_everything(config.seed, workers=True)
    else:
        log.info("Seed randomly...")
    
    # Load model. Uses a compiled model, if any, otherwise a uncompiled model
    log.info("Using model from <{}>".format(config.model_path))
    model = load_models(config.model_path, config.device)
    
    # Set up calculator
    model = EnsembleModel(model) if len(model) > 1 else model[0]
    calculator = instantiate(config.calculator, model=model)

    # Setup simulator
    simulator = instantiate(config.simulator, calculator=calculator)
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
    from .utils import load_models

    # Load the arguments
    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="select")
    
    # set up logger
    # set logger
    fh = logging.FileHandler(os.path.join(config.run_path, "selection.log"), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)

    # Save yaml file in run_path
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=True)
    log.info("Running on host: " + str(socket.gethostname()))

    # Set up the seed
    if "seed" in config:
        log.info(f"Seed with <{config.seed}>")
        seed_everything(config.seed, workers=True)
    else:
        log.info("Seed randomly...")
    
    # Set up datamodule and load training and validation set
    # The active learning only works for uncompiled model at the moment
    log.info("Using model from <{}>".format(config.model_path))
    models = load_models(config.model_path, config.device, load_compiled=False)
    cutoff = models[0].representation.cutoff

    # Load the pool data set and training data set
    if config.dataset and config.split_file:
        dataset = AseDataset(read_trajectory(config.dataset), cutoff=cutoff)
        with open(config.split_file) as f:
            split = json.load(f)
        data_dict = {}
        for k in split:
            data_dict[k] = torch.utils.data.Subset(dataset, split[k])
    elif config.pool_set:
        data_dict = {'pool': AseDataset(read_trajectory(config.pool_set), cutoff=cutoff)}
        if config.train_set:
            data_dict["train"] = AseDataset(read_trajectory(config.train_set), cutoff=cutoff)
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
    
    log.info(f"Active learning selection completed! Check {os.path.abspath(config.run_path+'/selected.json')} for selected structures!")
    if config.save_images:
        selected_images = read_trajectory(config.pool_set)
        save_path = config.save_images if isinstance(config.save_images, str) else os.path.join(config.run_path, 'selected.traj')
        with Trajectory(config.save_images if isinstance(config.save_images, str) else 'selected.traj', 'w') as traj:
            for atoms in selected_images:
                traj.write(atoms)
        log.info(f"Saving selected images into {save_path}.")

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
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=True)
    log.info("Running on host: " + str(socket.gethostname()))

    # get images and set parameters
    if config.pool_set:
        images = read_trajectory(config.pool_set)
        # Use active learning indices if provided
        if config.al_info:
            with open(config.al_info) as f:
                indices = json.load(f)["selected"]
                log.info(f"Labelling {len(indices)} active learning selected structures: {config.al_info}")
        elif config.indices:
            indices = config.indices
            log.info(f"Labelling {len(indices)} selected structures: {config.indices}")
        
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
        log.info(f"Rank {config.job_order}. Total structures: {len(images)}")

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
        log.info(f"Labeling structure {i}.")
        try:
            converged = db[i+1].get('converged')
            if not converged:
                converged = annotator.annotate(atoms)
                db.update(id=i+1, atoms=atoms, converged=converged)
                log.info(f"Recomputing structure {i} converged: {converged}")
            else:
                log.info(f"Structure {i} converged. Skipping...")
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
        log.info(f"Write atoms to {config.datapath}.") 
        total_dataset = Trajectory(config.datapath, 'a')
        for row in db.select(converged=True):
            if row.get('stored'):
                log.info(f"Structure {row.id - 1} is already stored in <{config.datapath}>. Skipping...")
            else:
                db.update(id=row.id, stored=True)
                log.info(f"Write structure {row.id - 1} to <{config.datapath}>")
                total_dataset.write(row.toatoms())
    
    if not all(all_converged):
        raise RuntimeError(f'Structures {[row.id -1 for row in db.select(converged=False)]} are not converged!')
    else:
        # sweep all unnessary files after labeling
        annotator.sweep()