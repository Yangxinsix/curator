# General modules for all tasks
from hydra.utils import instantiate
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf, open_dict
import sys, os, json
from pathlib import Path
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
    else:
        log.debug("Seed randomly...")
    
    # Initiate the datamodule
    log.debug(f"Instantiating datamodule <{config.data._target_}> from dataset {config.data.datapath or config.data.train_path}")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.data)
    
    # Initiate the model
    log.debug(f"Instantiating model <{config.model._target_}> with GNN representation <{config.model.representation._target_}>")
    model = hydra.utils.instantiate(config.model)
    log.debug(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,d}")
    
    # Initiate the task and load old model, if any
    log.debug(f"Instantiating task <{config.task._target_}>")
    task: LitNNP = hydra.utils.instantiate(config.task, model=model)
    if config.model_path is not None:
        # When using CSVlogger and save_hyperparameters() together, the code will report pickle error.
        # So we choose to save the entire model and outputs in LitNNP and then reload it
        config.model_path = find_best_model(config.model_path)[0]
        log.debug(f"Loading trained model from {config.model_path}")
        if config.task.load_entire_model:
            state_dict = torch.load(config.model_path)
            model = state_dict['model']
            outputs = state_dict['outputs']
        else:
            outputs = instantiate(config.task.outputs)
        task = LitNNP.load_from_checkpoint(checkpoint_path=config.model_path, model=model, outputs=outputs)
    # Save extra arguments in checkpoint
    task.save_configuration(config)
    
    # Initiate the training
    log.debug(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer)
    
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


    # Load the arguments
    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="train")

    # Save yaml file in run_path
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=True)
    log.info("Running on host: " + str(socket.gethostname()))
    
    # Set up the seed
    if "seed" in config:
        log.info(f"Seed with <{config.seed}>")
    else:
        log.info("Seed randomly...")
    
    # Setup the logger
    runHandler = logging.FileHandler(os.path.join(config.run_path, "printlog.txt"), mode="w")
    runHandler.setLevel(logging.DEBUG)
    runHandler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(runHandler)
    log.addHandler(logging.StreamHandler())    
    
    # Set up datamodule and load training and validation set
    print(config.data.datapath)
    if not os.path.isfile(config.data.datapath):
        raise RuntimeError("Please provide valid data path!")
    datamodule = instantiate(config.data)
    datamodule.setup()
    train_loader = datamodule.train_dataloader
    val_loader = datamodule.val_dataloader

    # Set up the model, the optimizer and  the scheduler
    model = instantiate(config.model)
    model.initialize_modules(datamodule)
    outputs = instantiate(config.task.outputs)
    model.to(config.device)
    for output in outputs:
        output.to(config.device)
    optimizer = instantiate(config.task.optimizer)(model.parameters())
    scheduler = instantiate(config.task.scheduler)(optimizer=optimizer)

    # If you use exponentialmovingaverage, you need to update the model parameters
    if config.task.use_ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=config.task.ema_decay)

    # Setup early stopping and inital training param  
    early_stop = EarlyStopping(patience=config.patience) 
    epoch = config.trainer.max_epochs
    best_val_loss = torch.inf
    prev_loss = None
    rescale_layers = []
    for layer in model.output_modules:
        if hasattr(layer, "unscale"):
            rescale_layers.append(layer)
    # Start training
    for e in range(epoch):
        # train
        model.train()
        for i, batch in enumerate(train_loader):
            # Initialize the batch, targets and loss
            batch = {k: v.to(config.device) for k, v in batch.items()}
            targets = {
                output.target_property: batch[output.target_property]
                for output in outputs
            }
            atoms_dict = {k: v for k, v in batch.items() if k not in targets}
            optimizer.zero_grad()
            unscaled_targets = targets.copy()
            unscaled_targets.update(atoms_dict)
            for layer in rescale_layers:
                unscaled_targets = layer.unscale(unscaled_targets, force_process=True)
            pred = model(batch)
            loss = 0.0
            
            # Calculate the loss and metrics
            metrics = {}
            for output in outputs:
                tmp_loss, _ = output.calculate_loss(unscaled_targets, pred)
                metrics[f"{output.target_property}_loss"] = tmp_loss.detach()
                loss += tmp_loss
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            if config.task.use_ema:
                ema.update()
            
            # Log the training metrics
            scaled_pred = pred.copy()
            scaled_pred.update(atoms_dict)
            for layer in rescale_layers:
                scaled_pred = layer.scale(scaled_pred, force_process=True)
            if i % config.trainer.log_every_n_steps == 0:
                for output in outputs:
                    for k, v in output.metrics['train'].items():
                        metrics[f"{output.name}_{k}"] = v(scaled_pred[output.name], targets[output.name]).detach()
                log_outputs = ",".join([f"{k}: {v:8.3f} " for k, v in metrics.items()])
                log.info(f"Training step: {i} {log_outputs}")
        
        # validation for each epoch
        model.eval()
        metrics = {}
        a_counts = 0
        s_counts = 0
        if config.task.use_ema:
            cm = ema.average_parameters()
        else:
            cm = contextlib.nullcontext()
        with cm:
            for i, batch in enumerate(val_loader):
                # Initialize the batch, targets and loss
                batch = {k: v.to(config.device) for k, v in batch.items()}
                targets = {
                    output.target_property: batch[output.target_property]
                    for output in outputs
                }
                atoms_dict = {k: v for k, v in batch.items() if k not in targets}
                    
                a = batch["forces"].shape[0]
                s = batch["energy"].shape[0]
                a_counts += a
                s_counts += s
                pred = model(batch)
                unscaled_targets, unscaled_pred = targets.copy(), pred.copy()
                unscaled_pred.update(atoms_dict)
                unscaled_targets.update(atoms_dict)
                for layer in rescale_layers:
                    unscaled_targets = layer.unscale(unscaled_targets, force_process=True)
                    unscaled_pred = layer.unscale(unscaled_pred, force_process=True)
                    
                # calculate loss
                for output in outputs:
                    tmp_loss = output.calculate_loss(unscaled_targets, unscaled_pred, return_num_obs=False).detach()
                    # metrics
                    if i == 0:
                        metrics[f"{output.target_property}_loss"] = tmp_loss
                    else:
                        metrics[f"{output.target_property}_loss"] += tmp_loss
                        
                    for k, v in output.metrics['train'].items():
                        m = v(pred[output.name], targets[output.name]).detach()
                        if "rmse" in k:
                            m = m ** 2
                        if i == 0:
                            metrics[f"{output.name}_{k}"] = m
                        else:
                            metrics[f"{output.name}_{k}"] += m
            
        # postprocess validation metrics    
        for k in metrics:
            metrics[k] /= i+1
            if "rmse" in k:
                metrics[k] = torch.sqrt(metrics[k])
        
        val_loss = metrics["energy_loss"] + metrics["forces_loss"]
        smooth_loss = val_loss if prev_loss is None else 0.9 * val_loss + 0.1 * prev_loss
        prev_loss = val_loss        
        
        # Save the model if validation loss is improving
        if not early_stop(smooth_loss, best_val_loss):
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {   "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_val_loss": best_val_loss,
                        "model_params": config.model,
                        "data_params": config.data,
                        "device": config.device,
                    },
                    os.path.join(config.run_path, "best_model.pth"),
                )
        # Stop training if validation loss is not improving for a while
        else:
            log.info(f"Validation epoch: {e}, {log_outputs}, patience: {early_stop.counter}")
            log.info(f"Validation loss is not improving for {early_stop.counter} epoches, exiting")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "model_params": config.model,
                    "data_params": config.data,
                    "device": config.device,
                },
                os.path.join(config.run_path, "exit_model.pth"),
            )
            break # stops the training loop
        # Log the validation metrics
        log_outputs = ",".join([f"{k}: {v:<8.3f} " for k, v in metrics.items()])
        log.info(f"Validation epoch: {e}, {log_outputs}, patience: {early_stop.counter}")
    
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
    else:
        log.info("Seed randomly...")
    
    # Load model. Uses a compiled model, if any, otherwise a uncompiled model
    log.info("Using model from <{}>".format(config.model_path))
    model = load_models(config.model_path, config.device)
    
    # Set up calculator
    model = EnsembleModel(model) if len(model) > 1 else model[0]
    MLcalc = MLCalculator(model)

    # Setup simulator
    simulator = instantiate(config.simulator, calculator=MLcalc)
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
    from ase.io import read
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
    log.info("Running on host: " + str(socket.gethostname()))

    # Save yaml file in run_path
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=True)
    log.info("Running on host: " + str(socket.gethostname()))

    # Set up the seed
    if "seed" in config:
        log.info(f"Seed with <{config.seed}>")
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
    )
    indices = al.select(models, data_dict, al_batch_size=config.batch_size, debug=config.debug)

    # Save the selected indices
    al_info = {
        'kernel': config.kernel,
        'selection': config.method,
        'dataset': list(config.dataset) if config.dataset and config.split_file else list(config.pool_set),
        'selected': indices,
    }
    with open(config.run_path+'/selected.json', 'w') as f:
        json.dump(al_info, f)
    
    log.info(f"Active learning selection completed! Check {os.path.abspath(config.run_path+'/selected.json')} for selected structures!")

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

    # Set up dataframe and load possible converged data id's
    db = connect(config.run_path+'/dft_structures.db')
    db_al_ind = [row.al_ind for row in db.select([('converged','=','True')])] #
    
    # get images and set parameters
    if config.label_set:
        images = read_trajectory(config.label_set)
        log.info(f"Labeling the structures in {config.label_set}")
    elif config.pool_set:
        pool_traj = read_trajectory(config.pool_set)
        # Use active learning indices if provided
        if config.al_info:
            with open(config.al_info) as f:
                indices = json.load(f)["selected"]
        elif config.indices:
            indices = config.indices
        else:
            raise RuntimeError('Valid index for labeling set should be provided!')
        # Remove converged structures from the label set
        if db_al_ind:
            _,rm,_ = np.intersect1d(indices, db_al_ind,return_indices=True)
            indices = np.delete(indices,rm)
        images = [pool_traj[i] for i in indices]
        log.info(f"Labeling {len(images)} structures in pool set: {config.pool_set}")    
    else:
        raise RuntimeError('Valid configarations for DFT calculation should be provided!')
    
    # split jobs if needed to accelerate labelling if you have a lot of resources
    if config.split_jobs or config.imgs_per_job:
        def split_list(lst, chunk_or_num, by_chunk_size=False):
            if by_chunk_size:
                num_chunks, remainder = divmod(len(lst), chunk_or_num)
            else:
                chunk_or_num, remainder = divmod(len(lst), chunk_or_num)
            if by_chunk_size:
                return [
                    lst[i * chunk_or_num + min(i, remainder):(i + 1) * chunk_or_num + min(i + 1, remainder)]
                    for i in range(num_chunks)
                ]
            else:
                return [
                    lst[i * (chunk_or_num + (1 if i < remainder else 0)):(i + 1) * (chunk_or_num + (1 if i < remainder else 0))]
                    for i in range(chunk_or_num)
                ]

        if config.split_jobs:
            images = split_list(images, config.split_jobs)        
        if config.imgs_per_job:
            images = split_list(images, config.imgs_per_job, by_chunk_size=True)
        images = images[config.job_order]          # specify which parts of the images to label

    # Set up calculator
    annotator = instantiate(config.annotator)
    
    # Label the structures
    for i, atoms in enumerate(images):
        # Label the structure with the choosen method
        converged = annotator.annotate(atoms)
        # Save the labeled structure with the index it comes from.
        al_ind = indices[i]
        db.write(atoms, al_ind=al_ind, converged=converged)
    
    # write to datapath
    if config.datapath is not None:
        log.info(f"Write atoms to {config.datapath}") 
        total_dataset = Trajectory(config.datapath, 'a')
        for row in db.select([('converged','=','True')]):
            total_dataset.write(row.toatoms())