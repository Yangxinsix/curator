import torch
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

def loss_fn(outputs, pred, batch, subset):
    """
    Computes the loss for the given predictions and batch data.

    Args:
        outputs: List of output modules.
        pred: Predicted values from the model.
        batch: Batch of data.
        subset: Indicates the subset ('train', 'val', etc.).

    Returns:
        A dictionary with loss values and the number of observations.
    """
    loss_dict = OrderedDict()
    loss_dict[subset + '_total_loss'] = 0.0
    num_obs_dict = OrderedDict()
    num_obs_dict[subset + '_total_loss'] = 1  # Assuming at least one observation

    for output in outputs:
        key = subset + '_' + output.name + '_loss'
        loss, num_obs = output.calculate_loss(pred, batch, True)
        loss_dict[key] = loss
        num_obs_dict[key] = num_obs
        loss_dict[subset + '_total_loss'] += loss

    return loss_dict, num_obs_dict

def run_step(model, batch, outputs, rescale_layers, device, optimizer=None, stage='train'):
    """
    Performs a single training step: forward pass, loss calculation, backpropagation.
    """
    # Move batch to device
    batch = {key: val.to(device) for key, val in batch.items()}

    # Zero gradients
    if stage == 'train':
        optimizer.zero_grad()

    # Forward pass
    pred = model(batch)
    pred.update({k: v for k, v in batch.items() if k not in pred.keys()})

    if stage == 'train':
        # Unscale batch for loss calculation
        unscaled_batch = batch
        for layer in rescale_layers:
            unscaled_batch = layer.unscale(unscaled_batch, force_process=True)

        # Compute loss
        loss_dict, _ = loss_fn(outputs, pred, unscaled_batch, stage)

        scaled_pred = pred
        for layer in rescale_layers[::-1]:
            scaled_pred = layer.scale(scaled_pred, force_process=True)

        # Calculate metrics
        all_metrics = {}
        for output in outputs:
            metrics = output.calculate_metrics(scaled_pred, batch, 'train')
            all_metrics.update(metrics)

        # Backpropagation
        loss_dict[stage + '_total_loss'].backward()
        optimizer.step()
    else:
        # Unscale batch and predictions for loss calculation
        unscaled_batch = batch
        unscaled_pred = pred
        for layer in rescale_layers:
            unscaled_batch = layer.unscale(unscaled_batch, force_process=True)
            unscaled_pred = layer.unscale(unscaled_pred, force_process=True)

        # Compute loss
        loss_dict, _ = loss_fn(outputs, unscaled_pred, unscaled_batch, 'val')
        # Calculate metrics
        all_metrics = {}
        for output in outputs:
            metrics = output.calculate_metrics(pred, batch, 'train')
            all_metrics.update(metrics)

    return loss_dict, all_metrics

def run_epoch(
        epoch_idx,
        model, 
        outputs, 
        data_loader, 
        device, 
        rescale_layers, 
        optimizer,
        stage='train',
        log_frequency=10,
    ):
    """
    Performs one full epoch over the data loader (training or validation).
    
    Args:
        model: The model to use for prediction.
        outputs: List of output modules with calculate_loss and calculate_metrics.
        data_loader: DataLoader to iterate over batches.
        device: Device to run the computation on (CPU or GPU).
        rescale_layers: Layers used to unscale or scale batch and prediction.
        optimizer: The optimizer to use for updating weights (for training only).
        is_train: Flag to indicate if this is a training epoch.
        log_every_n_steps: Interval of steps to log progress.

    Returns:
        Averages of losses and metrics for the entire epoch.
    """
    # Set model mode
    if stage == 'train':
        model.train()
    else:
        model.eval()
    
    for batch_idx, batch in enumerate(data_loader):
        loss_dict, batch_metrics = run_step(model, batch, outputs, rescale_layers, device, optimizer, stage)

        if batch_idx == 0:
            logger.info("\n")
            logger.debug(stage.capitalize())
            metric_names = [k.replace(stage + '_', '') for k in loss_dict.keys()]
            metric_names += [k.replace(stage + '_', '') for k in batch_metrics.keys()]
            metric_names = "".join([f'{m:>16s}' for m in metric_names])
            metric_names = f'# epoch      batch' + metric_names        
            logger.info(metric_names)
    
        if batch_idx % log_frequency == 0:
            msgs = [f'{epoch_idx:>7d}', f'{batch_idx:>11d}']
            vals = [f'{v or 0.0: >16.3g}' for v in loss_dict.values()]
            vals += [f'{v or 0.0: >16.3g}' for v in batch_metrics.values()]
            logger.info("".join(msgs + vals))

    # Average loss and metrics for the entire epoch
    all_loss = {'Total_loss': 0.0}
    all_metrics = {}
    for output in outputs:
        loss = output.accumulate_loss()
        all_loss['Total_loss'] += loss
        all_loss[output.name + '_loss'] = loss
        metrics = output.accumulate_metrics(stage)
        for k, v in metrics.items():
            all_metrics[output.name + '_' + k] = v
        #reset loss and metrics
        output.reset_loss()
        output.reset_metrics()

    head = [f'{stage:<12s}# epoch'] + [f'{k:>16s}' for k in all_loss] + [f'{k:>16s}' for k in all_metrics]
    info = [f'{epoch_idx:>19d}'] + [f'{v:>16.3g}' for v in all_loss.values()] + [f'{v:>16.3g}' for v in all_metrics.values()]
    logger.info("\n")
    logger.debug("Epoch summary:")
    logger.info("".join(head))
    logger.info("".join(info))

def train(
    model, 
    outputs, 
    optimizer, 
    scheduler=None, 
    train_loader=None, 
    val_loader=None, 
    test_loader=None, 
    device='cuda',
    num_epochs=10, 
    log_frequency=10,
):  
    # initialize model
    model = model.to(device)
    for output in outputs:
        output.to(device)

    # access rescale layers
    rescale_layers = []
    for layer in model.output_modules:
        if hasattr(layer, "unscale"):
            rescale_layers.append(layer)

    logger.info("\n")
    logger.debug("Start training model")

    for epoch_idx in range(num_epochs):
        # train
        run_epoch(epoch_idx, model, outputs, train_loader, device, rescale_layers, optimizer, 'train', log_frequency)
        # validation
        run_epoch(epoch_idx, model, outputs, val_loader, device, rescale_layers, optimizer, 'val', log_frequency)
        # test
        if test_loader is not None:
            run_epoch(epoch_idx, model, outputs, test_loader, device, rescale_layers, optimizer, 'test', log_frequency)

        # Scheduler step
        if scheduler is not None:
            scheduler.step()
    
    logger.debug("Training complete")

    return model