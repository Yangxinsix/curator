# General modules for all tasks
from hydra.utils import instantiate
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf, open_dict
import sys, os, json
from pathlib import Path
from .utils import read_user_config, CustomFormatter
import logging
import socket
import contextlib

# very ugly solution for solving pytorch lighting and myqueue conflictions
if "SLURM_NTASKS" in os.environ:
    del os.environ["SLURM_NTASKS"]
if "SLURM_JOB_NAME" in os.environ:
    del os.environ["SLURM_JOB_NAME"]

# Set up logger for the different tasks 
log = logging.getLogger('curator')
log.setLevel(logging.DEBUG)

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
    log.debug(f"Instantiating datamodule <{config.data._target_}> from dataset {config.data.datapath or config.data.train_datapath}")
    if not os.path.isfile(config.data.datapath or config.data.train_datapath):
        raise RuntimeError("Please provide valid data path!")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.data)
    
    # Initiate the model
    log.debug(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)
    log.debug(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,d}")
    
    # Initiate the task and load old model, if any
    log.debug(f"Instantiating task <{config.task._target_}>")
    task: LitNNP = hydra.utils.instantiate(config.task, model=model)
    if config.model_path is not None:
        log.debug(f"Loading trained model from {config.model_path}")
        # When using CSVlogger and save_hyperparameters() together, the code will report pickle error.
        # So we choose to save the entire model and outputs in LitNNP and then reload it
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
        model_path = [str(f) for f in Path(f"{config.run_path}").rglob("best_model_epoch*")]
        val_loss = float('inf')
        index = 0
        for i, p in enumerate(model_path):
            loss = float(p.split('=')[-1].rstrip('.ckpt'))
            if loss < val_loss:
                val_loss = loss
                index = i
        model_path = model_path[index]
          
        # Compile the model
        outputs = torch.load(model_path)['outputs']
        log.debug(f"Deploy trained model from {model_path} with validation loss of {val_loss:.3f}")
        task = LitNNP.load_from_checkpoint(checkpoint_path=f"{model_path}", model=model, outputs=outputs)
        model_compiled = script(task.model)
        metadata = {"cutoff": str(model_compiled.representation.cutoff).encode("ascii")}
        model_compiled.save(f"{config.run_path}/compiled_model.pt", _extra_files=metadata)
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
@hydra.main(config_path="configs", config_name="train", version_base=None)   
def deploy(config: DictConfig):
    """ Deploy the model and save a compiled model.

    Args:
        config (DictConfig): The configuration file.
    Returns:
        None
    
    """
    import torch
    from curator.model import LitNNP
    from e3nn.util.jit import script


    # Load the arguments
    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="train")

    # Set up datamodule and load training and validation set
    if not os.path.isfile(config.data.datapath):
        raise RuntimeError("Please provide valid data path!")
    datamodule = instantiate(config.data)
    datamodule.setup()
    
    # Set up model, optimizer and scheduler
    model = instantiate(config.model)
    model.initialize_modules(datamodule)
   
    # Load model
    model_path = [str(f) for f in Path(f"{config.run_path}").rglob("best_model.pth*")]
    val_loss = float('inf')
    index = 0
    for i, p in enumerate(model_path):
        loss = float(p.split('=')[-1].rstrip('.ckpt'))
        if loss < val_loss:
            val_loss = loss
            index = i
    model_path = model_path[index]
    loaded_model = torch.load(model_path, map_location=torch.device(config.device))
    model.load_state_dict(loaded_model["model"])

    # Compile the model
    model_compiled = script(model)
    metadata = {"cutoff": str(model_compiled.representation.cutoff).encode("ascii")}
    model_compiled.save(f"{config.run_path}/compiled_model.pt", _extra_files=metadata)
    log.info(f"Deploying compiled model at <{config.run_path}/compiled_model.pt>")

# Simulate with the model
@hydra.main(config_path="configs", config_name="simulate", version_base=None)
def simulate(config: DictConfig):
    """ Simulate with the model.

    Args:
        config (DictConfig): The configuration file.
    Returns:
        None
    """
    import torch
    from curator.simulator.calculator import MLCalculator, EnsembleCalculator
    from curator.simulator import PrintEnergy

    # Load the arguments
    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="simulate")
    
    # Save yaml file in run_path
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=True)
    
    # set logger
    log.setLevel(logging.DEBUG)
    log.info("Running on host: " + str(socket.gethostname()))

    # Set up the seed
    if "seed" in config:
        log.info(f"Seed with <{config.seed}>")
    else:
        log.info("Seed randomly...")
    
    # Load model. Uses a compiled model, if any, otherwise a not compiled model
    model_pt = Path(config.model_path).rglob('*compiled_model.pt')
    models = [torch.jit.load(each, map_location=torch.device(config.device)) for each in model_pt]
    
    if len(models)==0:
        model_pt = Path(config.model_path).rglob('*best_model.pth')
        #models = [torch.load(each,map_location=torch.device(config.device)) for each in model_pt]
        models = []
        for each in model_pt:
            model_dict = torch.load(each, map_location=torch.device(config.device))
            datamodule = instantiate(model_dict['data_params'])
            datamodule.setup()
            model = instantiate(model_dict['model_params'])
            model.initialize_modules(datamodule)
            model.load_state_dict(model_dict['model'])
            models.append(model)
        
        if len(models)==0:
            raise RuntimeError("No compiled or not complied models found!")
        else:
            log.info("Uses not compiled model!")
            #cutoff = float(models[0]['model']['cutoff'])
            cutoff = float(model_dict['data_params']['cutoff'])
    else:
        log.info("Uses compiled model!")
        cutoff = models[0].representation.cutoff
    
    
    # Set up calculator
    if len(models) >1:
        log.info("Ensemble calculator")
        MLcalc = EnsembleCalculator(models,cutoff=cutoff)
    elif len(models) == 1:
        log.info("Single calculator")
        MLcalc = MLCalculator(models[0],cutoff=cutoff)
    else:
        raise RuntimeError("No models found!")
    # Setup the universal logger
    PE = PrintEnergy(config.uncertainty,log)

    # Initiate simulators
    if config.simulator.method == "md":
        from curator.simulator.md import MD
        simulator = MD(config.simulator,MLcalc,PE)
    
    elif config.simulator.method == "neb":
        from curator.simulator.neb import NEB
        simulator = NEB(config.simulator,MLcalc,PE)
    
    elif config.simulator.method == "md_meta":
        from curator.simulator.md_meta import MD_meta
        simualte = MD_meta(config.simulator,MLcalc,PE)
    
    else:
        raise NotImplementedError(f"Simulator <{config.simulator.method}> not implemented yet!")
    
@hydra.main(config_path="configs", config_name="select", version_base=None)   
def select(config: DictConfig):
    """ Select structures with active learning.

    Args:
        config (DictConfig): The configuration file.
    Returns:
        None
    """
    import torch
    from ase.io import read
    from curator.selection import GeneralActiveLearning
    import json
    from curator.data import AseDataset

    # Load the arguments
    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="select")
    
    # Save yaml file in run_path
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=True)
    log.info("Running on host: " + str(socket.gethostname()))

    # Set up the seed
    if "seed" in config:
        log.info(f"Seed with <{config.seed}>")
    else:
        log.info("Seed randomly...")

    # Load the arguments
    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="select")
    
    # Set up datamodule and load training and validation set
    # The active learning only works for uncompiled model at the moment
    model_pt = list(Path(config.model_path).rglob('*best_model.pth'))
    models = []
    for each in model_pt:
        model_dict = torch.load(each, map_location=torch.device(config.device))
        datamodule = instantiate(model_dict['data_params'])
        datamodule.setup()
        model = instantiate(model_dict['model_params'])
        model.initialize_modules(datamodule)
        model.load_state_dict(model_dict['model'])
        models.append(model)

    cutoff = models[0].representation.cutoff

    # Load the pool data set and training data set
    if config.pool_set and config.train_set:
        if isinstance(config.pool_set, list):
            pool_set = []
            for traj in config.pool_set:
                if Path(traj).stat().st_size > 0:
                    pool_set += read(traj, index=':') 
        else:
            pool_set = read(config.pool_set, index=':')
    else:
        raise RuntimeError("Please give valid pool data set for selection!")
    
    data_dict = {
            'pool': AseDataset(pool_set, cutoff=cutoff),
            'train': AseDataset(config.train_set, cutoff=cutoff),
        }
    
    # Check the size of pool data set
    if len(data_dict['pool']) < config.batch_size * 10: 
            log.warning(f"""The pool data set ({len(data_dict['pool'])}) is not large enough for selection!
            It should be larger than 10 times batch size ({config.batch_size*10}).
            Check your MD simulation!""")
    elif len(data_dict['pool']) < config.batch_size:
        raise RuntimeError(f"""The pool data set ({len(data_dict['pool'])}) is not large enough for selection! Add more data or change batch size {config.selection.batch_size}.""")

    # Select structures based on the active learning method
    al = GeneralActiveLearning(
        kernel=config.kernel, 
        selection=config.method, 
        n_random_features=config.n_random_features,
    )
    indices = al.select(models, data_dict, al_batch_size=config.batch_size)

    # Save the selected indices
    al_info = {
        'kernel': config.kernel,
        'selection': config.method,
        'dataset': config.pool_set,
        'selected': indices,
    }
    with open(config.run_path+'/selected.json', 'w') as f:
        json.dump(al_info, f)

# Label the dataset selected by active learning
@hydra.main(config_path="configs", config_name="label", version_base=None)   
def label(config: DictConfig):
    """ Label the dataset selected by active learning.

    Args:
        config (DictConfig): The configuration file.
    Returns:
        None
    """
    from ase.db import connect
    from ase.io import read, Trajectory
    import json
    import numpy as np

    # Load the arguments
    if config.cfg is not None:
        config = read_user_config(config.cfg, config_path="configs", config_name="label")

    # Save yaml file in run_path
    OmegaConf.save(config, f"{config.run_path}/config.yaml", resolve=True)
    log.info("Running on host: " + str(socket.gethostname()))

    # Set up dataframe and load possible converged data id's
    db = connect(config.run_path+'/dft_structures.db')
    db_al_ind = [row.al_ind for row in db.select([('converged','=','True')])] #
    
    # get images and set parameters
    if config.label_set:
        images = read(config.label_set, index = ':')
    elif config.pool_set:
        if isinstance(config.pool_set, list):
            pool_traj = []
            for pool_path  in config.pool_set:
                if Path(pool_path).stat().st_size > 0:
                    pool_traj += read(pool_path, ':')
        else:
            pool_traj = Trajectory(config.pool_set)
    
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
    else:
        raise RuntimeError('Valid configarations for DFT calculation should be provided!')
    
    # Set up calculator
    if config.labeling_method.method == 'vasp':
        from curator.labeling.vasp import VASP
        label = VASP(config.labeling_method.parameters)
    elif config.labeling_method.method== 'gpaw':
        from curator.labeling.gpaw import GPAW
        label = GPAW(config.labeling_method.parameters)
    elif config.user_method:
        raise NotImplementedError('User defined method is not implemented yet!')
    
    # Label the structures
    for i, atoms in enumerate(images):
        # Label the structure with the choosen method
        check_result = label.label(atoms)
        # Save the labeled structure with the index it comes from.
        al_ind = indices[i]
        db.write(atoms, al_ind=al_ind, converged=check_result)
    
    if config.datapath is not None:
        total_dataset = Trajectory(config.datapath, 'a')
        atoms = []
        for row in db.select([('converged','=','True')]):
            total_dataset.write(row.toatoms())