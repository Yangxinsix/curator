import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Type, Any
from curator.data import properties
from pytorch_lightning.utilities.types import STEP_OUTPUT
import warnings
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
import copy
from pytorch_lightning import LightningDataModule
from omegaconf import DictConfig
import logging
from collections import OrderedDict, defaultdict
from curator.model import MACE


class NeuralNetworkPotential(nn.Module):
    """ Base class for neural network potentials."""
    def __init__(
        self,
        representation: nn.Module,
        input_modules: List[nn.Module] = None,
        output_modules: List[nn.Module] = None,    
    ) -> None:
        """ Base class for neural network potentials.
        
        Args:
            representation (nn.Module): Representation module
            input_modules (List[nn.Module], optional): Input modules. Defaults to None.
            output_modules (List[nn.Module], optional): Output modules. Defaults to None.
        """
        super().__init__()
        self.representation = representation
        self.input_modules = nn.ModuleList(input_modules)
        self.output_modules = nn.ModuleList(output_modules)
        self.model_outputs: Optional[List[str]] = None
        self._initialized: bool = False
        self.collect_outputs()

        for module in self.output_modules:
            if hasattr(module, 'update_callback'):
                module.update_callback = self.collect_outputs
        
    def forward(self, data: properties.Type) -> properties.Type:
        data = data.copy()
        for m in self.input_modules:
            data = m(data)
            
        data = self.representation(data)
        
        for m in self.output_modules:
            data = m(data)
        
        results = self.extract_outputs(data)
        return results

    def initialize_modules(self, datamodule: LightningDataModule) -> None:
        for module in self.modules():
            if hasattr(module, "datamodule"):
                module.datamodule(datamodule)
        self._initialized = True
    
    def collect_outputs(self) -> None:
        self.model_outputs = None
        model_outputs = set()
        for m in self.modules():
            if hasattr(m, "model_outputs") and m.model_outputs is not None:
                model_outputs.update(m.model_outputs)
        model_outputs: List[str] = list(model_outputs)
        self.model_outputs = model_outputs
    
    def extract_outputs(self, data: properties.Type) -> properties.Type:
        results = {k: data[k] for k in self.model_outputs}
        return results
    
    # used to update model outputs
    def register_callbacks(self, module: nn.Module) -> None:
        module.update_callback = self.collect_outputs

class EnsembleModel(nn.Module):
    """
    Ensemble model for evaluating uncertainties
    """
    def __init__(self, models: List[NeuralNetworkPotential]) -> None:
        super().__init__()
        self.models = nn.ModuleList([model for model in models])
        self.compute_uncertainty = True if len(models) > 1 else False

    def forward(self, data: properties.Type) -> properties.Type:
        energy = []
        forces = []
        for model in self.models:
            out = model(data)
            energy.append(out[properties.energy].detach())
            forces.append(out[properties.forces].detach())
        
        energy = torch.stack(energy)
        forces = torch.stack(forces)
        f_scatter = torch.zeros(data[properties.n_atoms].shape[0], device=out[properties.energy].device)
        
        result_dict = {
            properties.energy: torch.mean(energy, dim=0),
            properties.forces: torch.mean(forces, dim=0),
        }

        if self.compute_uncertainty:
            uncertainty = {
                properties.e_max: torch.max(energy).unsqueeze(-1),
                properties.e_min: torch.min(energy).unsqueeze(-1),
                properties.e_var: torch.var(energy, dim=0),
                properties.e_sd: torch.std(energy, dim=0),
                properties.f_var: f_scatter.index_add(0, data[properties.image_idx], torch.var(forces, dim=0).mean(dim=1)) / data[properties.n_atoms],
            }
            uncertainty[properties.f_sd] = uncertainty[properties.f_var].sqrt()
            result_dict[properties.uncertainty] = uncertainty
        
        if properties.energy in data:
            # calculate errors
            e_diff = result_dict[properties.energy] - data[properties.energy]
            f_diff = result_dict[properties.forces] - data[properties.forces]
            error = {
                properties.e_ae: torch.abs(e_diff),
                properties.e_se: torch.square(e_diff),
                properties.f_ae: f_scatter.index_add(0, data[properties.image_idx], torch.abs(f_diff).mean(dim=1)) / data[properties.n_atoms],
                properties.f_se: f_scatter.index_add(0, data[properties.image_idx], torch.square(f_diff).mean(dim=1)) / data[properties.n_atoms],
            }

            # TODO: wait for torch_scatter.
            # error[properties.f_maxe] = f_scatter.scatter_reduce(0, data[properties.image_idx].unsqueeze(1).expand_as(f_diff), torch.abs(f_diff), reduce='amax', include_self=False)
            # error[properties.f_mine] = f_scatter.scatter_reduce(0, data[properties.image_idx].unsqueeze(1).expand_as(f_diff), torch.abs(f_diff), reduce='amin', include_self=False)
            result_dict[properties.error] = error

        return result_dict

class DropoutModel(nn.Module):
    pass

class ModelOutput(nn.Module):
    """ Base class for model outputs."""
    def __init__(
        self,
        name: str,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: float = 1.0,
        metrics: Optional[Dict[str, Metric]] = None,
        target_property: Optional[str]=None,
        # per_species_loss: bool=False,
        # per_species_metrics: bool=False,
    ) -> None:
        """ Base class for model outputs. 

        Args:
            name (str): Name of the output
            loss_fn (Optional[nn.Module], optional): Loss function. Defaults to None.
            loss_weight (float, optional): Loss weight. Defaults to 1.0.
            metrics (Optional[Dict[str, Metric]], optional): Metrics. Defaults to None.
            target_property (Optional[str], optional): Target property. Defaults to None.
        """
        super().__init__()
        self.name = name
        self.target_property = target_property or name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        self.train_metrics = nn.ModuleDict(metrics)
        self.val_metrics = nn.ModuleDict({k: copy.copy(v) for k, v in metrics.items()})
        self.test_metrics = nn.ModuleDict({k: copy.copy(v) for k, v in metrics.items()})
        
        # here we found a serious bug that deepcopy is not working in hydra instantiate!!!
        self.metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }
        
        # self.per_species_loss = per_species_loss
        # self.per_species_metrics = per_species_metrics
    
    def calculate_loss(self, pred: Dict, target: Dict, return_num_obs=True) -> torch.Tensor:
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0

        loss = self.loss_weight * self.loss_fn(
            pred[self.name], target[self.target_property]
        )
        num_obs = target[self.target_property].view(-1).shape[0]
        
        if return_num_obs:
            return loss, num_obs
        
        return loss

    def update_metrics(self, pred: Dict, target: Dict, subset: str) -> None:
        for metric in self.metrics[subset].values():
            metric(pred[self.name], target[self.target_property])

    def calculate_metrics(self, pred: Dict, target: Dict, subset: str) -> None:
        batch_val = OrderedDict()
        # if self.per_species_metrics:
        #     pass
        for k in self.metrics[subset]:
            metric = self.metrics[subset][k]
            if isinstance(metric, torchmetrics.Metric):
                metric(pred[self.name].detach(), target[self.target_property].detach())
            else:
                metric = metric(pred[self.name].detach(), target[self.target_property].detach())
            batch_val[f"{subset}_{self.name}_{k}"] = metric
        
        return batch_val
    
    def reset_loss(self, subset: Optional[str]=None) -> None:
        if subset is None:
            for k in self.loss:
                self.loss[k] = 0.0
                self.num_obs[k] = 0
        else:
            self.loss[subset] = 0.0
            self.num_obs[subset] = 0
    
    def reset_metrics(self, subset: Optional[str]=None) -> None:
        if subset is None:
            for k1 in self.metrics:
                for k2 in self.metrics[k1]:
                    self.metrics[k1][k2].reset()
        else:
            for k in self.metrics[subset]:
                self.metrics[subset][k].reset()
    
    # # def register_key(self,)
    
    # def add_metrics(self, name: str, metric: Metric, subset: Optional[str]=None) -> None:
    #     if subset is None:
    #         for k1 in self.metrics:
    #             self.metrics[k1][name] = metric
    #     else:
    #         self.metrics[subset][name] = metric
    
    # def update_metrics(self, metric_dict: Dict[str, Metric], subset: Optional[str]=None) -> None:
    #     if subset is None:
    #         for k1 in self.metrics:
    #             self.metrics[k1].update(metric_dict)
    #     else:
    #         self.metrics[subset].update(metric_dict)


logger = logging.getLogger(__name__)    # console output
class LitNNP(pl.LightningModule):
    """ Base class for neural network potentials using PyTorch Lightning."""
    def __init__(
        self,
        model: NeuralNetworkPotential,
        outputs: List[ModelOutput],
        optimizer: Type[torch.optim.Optimizer],
        scheduler: Optional[Type] = None,
        scheduler_monitor: Optional[str] = None,
        warmup_steps: int = 0,
        save_entire_model: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """ Base class for neural network potentials using PyTorch Lightning.

        Args:
            model (NeuralNetworkPotential): Neural network potential model
            outputs (List[ModelOutput]): List of model outputs
            optimizer (Type[torch.optim.Optimizer]): Optimizer
            scheduler (Optional[Type], optional): Scheduler. Defaults to None.
            scheduler_monitor (Optional[str], optional): Scheduler monitor. Defaults to None.
            warmup_steps (int, optional): Warmup steps. Defaults to 0.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'outputs'])
        self.model = model
        self.outputs = nn.ModuleList(outputs)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_monitor = scheduler_monitor
        self.warmup_steps = warmup_steps
        self.save_entire_model = save_entire_model
        
    def setup(self, stage: Optional[str]=None) -> None:
        if stage == "fit":
            if not self.model._initialized:
                self.model.initialize_modules(self.trainer.datamodule)
            self.rescale_layers = []
            for layer in self.model.output_modules:
                if hasattr(layer, "unscale"):
                    self.rescale_layers.append(layer)
    
    def loss_fn(self, pred: Dict, batch: Dict, subset: str):
        loss_dict = OrderedDict()
        loss_dict[subset + '_total_loss'] = 0.0
        num_obs_dict = OrderedDict()
        num_obs_dict[subset + '_total_loss'] = 1
        for output in self.outputs:
            key = subset + '_' + output.name + '_loss'
            loss_dict[key], num_obs_dict[key] = output.calculate_loss(pred, batch, True)
            loss_dict[subset + '_total_loss'] += loss_dict[key]
            
        return loss_dict, num_obs_dict
    
    # TODO: manually reset metrics

    # def log_metrics(self, pred: Dict, batch: Dict, subset: str) -> None:
    #     for output in self.outputs:
    #         output.update_metrics(pred, batch, subset)
    #         for metric_name, metric in output.metrics[subset].items():
    #             self.log(
    #                 f"{subset}_{output.name}_{metric_name}",
    #                 metric,
    #                 on_step=(subset == "train"),
    #                 on_epoch=(subset != "train"),
    #                 prog_bar=False,
    #             )
                
    # def reset_loss_metrics(self, subset: str) -> None:
    #     for output in self.outputs:
    #         output.loss = 0.0
    #         output.num_obs = 0
    #         for k in output.metrics:
    #             output.metrics[subset][k].reset()
    
    
    def _get_metrics_key(self, subset: str):
        msgs = [f'{"total_loss":>16s}']
        msgs += [output.name + '_loss' for output in self.outputs]
        msgs += [output.name + '_' + metric for output in self.outputs for metric in output.metrics[subset]]
        msgs = [f'{msg:>16s}' for msg in msgs]
        return msgs
    
    def _get_metrics(self):
        forward_cache = [f'{metric._forward_cache:>16.3g}' for metric in self.trainer._results.values()]
        return forward_cache
    
    def on_train_start(self):
        logger.info("\n")
        logger.debug("Start training model")
    
    # def on_validation_start(self):
    #     logger.info("\nStart validating model")
        
    def on_test_start(self):
        logger.info("\n")
        logger.debug("Start testing model")

    def on_train_epoch_start(self):
        logger.info("\n")
        logger.debug("Training")
        head = [f'# epoch      batch']
        logger.info("".join(head + self._get_metrics_key(subset='train')))
        
    def on_validation_epoch_start(self):
        torch.set_grad_enabled(True)
        logger.info("\n")
        logger.debug("Validating")
        head = [f'# epoch      batch']
        logger.info("".join(head + self._get_metrics_key(subset='train')))
    
    def on_test_epoch_start(self):
        torch.set_grad_enabled(True)
        logger.info("\n")
        logger.debug("Testing")
        head = [f'# epoch      batch']
        logger.info("".join(head + self._get_metrics_key(subset='train')))
    
    def training_step(self, batch: Dict, batch_idx: List[int]) -> torch.Tensor:
        pred = self.model(batch)
        pred.update({k: v for k, v in batch.items() if k not in pred.keys()})

        # calculate loss, metrics
        # unscale batch because loss will be calculated with normalized units
        for layer in self.rescale_layers:
            unscaled_batch = layer.unscale(batch, force_process=True)
        loss_dict, num_abs_dict = self.loss_fn(pred, unscaled_batch, 'train')
        for k in loss_dict.keys():
            self.log(k, loss_dict[k], batch_size=num_abs_dict[k], on_step=True, on_epoch=True, prog_bar=True)
        
        # when calculate metrics pred need to be scaled to get real units
        for layer in self.rescale_layers[::-1]:
            scaled_pred = layer.scale(pred, force_process=True)
        for output in self.outputs:
            batch_metrics = output.calculate_metrics(scaled_pred, batch, 'train')
            self.log_dict(batch_metrics, on_step=True, on_epoch=True, prog_bar=False)
        
        # # logging metric keys
        # if self.current_epoch == 0 and not self._train_log_head:
        #     logger.info("".join(self._get_metrics_key(subset='train')))
        #     self._train_log_head = True
        # logging metrics to console
        if batch_idx % self.trainer.log_every_n_steps == 0:
            msgs = [f'{self.current_epoch:>7d}', f'{batch_idx:>11d}']
            forward_cache = [f'{metric._forward_cache:>16.3g}' for metric in self.trainer._results.values()]
            logger.info("".join(msgs + forward_cache))
            
        return loss_dict['train_total_loss']
    
    def validation_step(self, batch: Dict, batch_idx: List[int]) -> torch.Tensor:
        pred = self.model(batch)
        pred.update({k: v for k, v in batch.items() if k not in pred.keys()})
        
        # calculate loss, metrics
        # both batch and pred need to be normalized for calculating loss in validation mode
        for layer in self.rescale_layers:
            unscaled_batch = layer.unscale(batch, force_process=True)
            unscaled_pred = layer.unscale(pred, force_process=True)
        loss_dict, num_abs_dict = self.loss_fn(unscaled_pred, unscaled_batch, 'val')
        for k in loss_dict.keys():
            self.log(k, loss_dict[k], batch_size=num_abs_dict[k], on_step=True, on_epoch=True, prog_bar=True) 
        
        # nothing need to be scaled for calculating metrics        
        for output in self.outputs:
            batch_metrics = output.calculate_metrics(pred, batch, 'val')
            self.log_dict(batch_metrics, on_step=True, on_epoch=True, prog_bar=False)
        
        # logging metrics to console
        if batch_idx % self.trainer.log_every_n_steps == 0:
            msgs = [f'{self.current_epoch:>7d}', f'{batch_idx:>11d}']
            forward_cache = [f'{metric._forward_cache:>16.3g}' for metric in self.trainer._results.values()]
            logger.info("".join(msgs + forward_cache))
        
        return loss_dict['val_total_loss']
    
    def test_step(self, batch: Dict, batch_idx: List[int]) -> torch.Tensor:
        pred = self.model(batch)
        pred.update({k: v for k, v in batch.items() if k not in pred.keys()})
        
        # calculate loss, metrics
        # both targets and pred need to be normalized for calculating loss in validation mode
        for layer in self.rescale_layers:
            unscaled_targets = layer.unscale(batch, force_process=True)
            unscaled_pred = layer.unscale(pred, force_process=True)
        loss_dict, num_abs_dict = self.loss_fn(unscaled_pred, unscaled_targets, 'test')
        for k in loss_dict.keys():
            self.log(k, loss_dict[k], batch_size=num_abs_dict[k], on_step=True, on_epoch=False, prog_bar=True) 
               
        for output in self.outputs:
            batch_metrics = output.calculate_metrics(pred, batch, 'test')
            self.log_dict(batch_metrics, on_step=True, on_epoch=True, prog_bar=False)
        
        # logging metrics to console
        if batch_idx % self.trainer.log_every_n_steps == 0:
            msgs = [f'{self.current_epoch:>7d}', f'{batch_idx:>11d}']
            forward_cache = [f'{metric._forward_cache:>16.3g}' for metric in self.trainer._results.values()]
            logger.info("".join(msgs + forward_cache))
        
        return loss_dict['test_loss']
    
    def on_train_epoch_end(self):
        msgs = ['Train       # epoch'] + self._get_metrics_key(subset='train')
        metrics = [f'{self.current_epoch:>19d}']
        metrics += [f'{metric.compute():>16.3g}' for metric in self.trainer._results.values()]
        logger.info("".join(msgs))
        logger.info("".join(metrics))
    
    def on_validation_epoch_end(self):
        # validation end goes before train epoch
        msgs = ['Validation  # epoch'] + self._get_metrics_key(subset='val')
        metrics = [f'{self.current_epoch:>19d}']
        metrics += [f'{metric.compute():>16.3g}' for metric in self.trainer._results.values()]
        
        logger.info("\n")
        logger.debug("Epoch summary:")
        logger.info("".join(msgs))
        logger.info("".join(metrics))
    
    def save_configuration(self, config: DictConfig):
        self.config = config
        
    def on_save_checkpoint(self, checkpoint):
        checkpoint['data_params'] = self.config.data
        checkpoint['model_params'] = self.config.model
        checkpoint['outputs'] = self.outputs
        if self.save_entire_model:
            checkpoint['model'] = self.model
    
    def configure_optimizers(self) -> Type[torch.optim.Optimizer]:
        if type(self.model.representation) == MACE:
            decay_params = {}
            no_decay_params = {}

            for name, param in self.named_parameters():
                if ("linear_2.weight" in name or "skip_tp_full.weight" in name or "products" in name) and "readouts" not in name:
                    decay_params[name] = param
                else:
                    no_decay_params[name] = param
                    
            param_group = [
                {
                    "name": "decay_params",
                    "params": list(decay_params.values()),
                    "weight_decay": self.optimizer.keywords['weight_decay'],
                },
                {
                    "name": "no_decay_params",
                    "params": list(no_decay_params.values()),
                    "weight_decay": 0.0,
                },
            ]
            optimizer = self.optimizer(params=param_group)
        else:
            optimizer = self.optimizer(params=self.parameters())
        # optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            lr_scheduler = {"scheduler": scheduler}
            if self.scheduler_monitor:
                lr_scheduler["monitor"] = self.scheduler_monitor
            if self.trainer.val_check_interval < 1.0:
                warnings.warn(
                    "Learning rate is scheduled after epoch end. To enable scheduling before epoch end, "
                    "please specify val_check_interval by the number of training epochs after which the "
                    "model is validated."
                )
            # in case model is validated before epoch end (recommended use of val_check_interval)
            if self.trainer.val_check_interval > 1.0:
                lr_scheduler["interval"] = "step"
                lr_scheduler["frequency"] = self.trainer.val_check_interval
                
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
            }
        else:
            return optimizer