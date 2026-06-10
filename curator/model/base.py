import torch
import re
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Type, Any, Union
from curator.data import properties
from curator.train.model_output import ModelOutput
from pytorch_lightning.utilities.types import STEP_OUTPUT
import warnings
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from omegaconf import DictConfig
import logging
import os
from collections import OrderedDict, defaultdict
try:
    from torch_scatter import scatter_add, scatter_mean
except ImportError:
    from curator.utils import scatter_add, scatter_mean

logger = logging.getLogger(__name__)    # console output
class NeuralNetworkPotential(nn.Module):
    """ Base class for neural network potentials."""
    def __init__(
        self,
        representation: nn.Module,
        input_modules: List[nn.Module] = None,
        output_modules: List[nn.Module] = None,
        model_outputs: List[str] = [],
    ) -> None:
        """ Base class for neural network potentials.
        
        Args:
            representation (nn.Module): Representation module
            input_modules (List[nn.Module], optional): Input modules. Defaults to None.
            output_modules (List[nn.Module], optional): Output modules. Defaults to None.
        """
        super().__init__()

        self.representation = representation
        self.model_outputs = model_outputs
        self.input_modules = CallbackModuleList(input_modules, on_register_callback=None)
        self.output_modules = CallbackModuleList(output_modules, on_register_callback=self.register_callbacks)
        self._initialized: bool = False
        self._return_all_outputs: bool = False  # When True, bypass extract_outputs and return all data
        self.collect_outputs()
        self.register_callbacks()
        self.rescale_layers = []
        for layer in self.output_modules:
            if hasattr(layer, "unscale"):
                self.rescale_layers.append(layer)
        
    def forward(self, data: properties.Type) -> properties.Type:
        data = data.copy()
        for m in self.input_modules:
            data = m(data)
            
        data = self.representation(data)
        
        for m in self.output_modules:
            # import pdb; pdb.set_trace()
            if hasattr(m, 'electronegativity_mlp'):
                # import pdb; pdb.set_trace()
                scaled_data = data.copy()
                for layer in self.rescale_layers[::-1]:
                    scaled_data = layer.scale(scaled_data, force_process=True)
                # Copy scaled values back before QEQ.
                # This must include LAMMPS-specific aliases such as atomic_energy and edge_forces,
                # not only the structure-level energy/forces keys.
                data.update(scaled_data)
                # import pdb; pdb.set_trace()
                data = m(data)
                # import pdb; pdb.set_trace()
                for layer in self.rescale_layers:
                    data = layer.unscale(data, force_process=True)
                
            else:
                data = m(data)
        # import pdb; pdb.set_trace()
        return self.extract_outputs(data)

    def initialize_modules(self, datamodule: LightningDataModule) -> None:
        for module in self.modules():
            if hasattr(module, "datamodule"):
                module.datamodule(datamodule)
        self._initialized = True
    
    def collect_outputs(self) -> None:
        model_outputs = set()
        for m in self.modules():
            if hasattr(m, "model_outputs") and m.model_outputs is not None:
                model_outputs.update(m.model_outputs)
        model_outputs: List[str] = list(model_outputs)
        self.model_outputs = list(set(self.model_outputs + model_outputs))
    
    def extract_outputs(self, data: properties.Type) -> properties.Type:
        # Handle models saved before _return_all_outputs was added
        return_all = getattr(self, '_return_all_outputs', False)
        if return_all or 'all' in self.model_outputs:
            return data
        else: 
            return {k: data[k] for k in self.model_outputs if k in data}
    
    # used to update model outputs
    def register_callbacks(self, target_module: Union[nn.Module, List[nn.Module], None]=None) -> None:
        def register_module(module):
            if hasattr(module, 'update_callback'):
                module.update_callback = self.collect_outputs
            if hasattr(module, 'repr_callback'):
                module.register_repr_callback(self)        # activate repr callback for feature extractor and calculator
            if hasattr(module, "model_outputs") and module.model_outputs is not None:
                for model_output in module.model_outputs:
                    if model_output not in self.model_outputs:
                        self.model_outputs.append(model_output)
                        
        if target_module is None:
            for module in self.output_modules:
                register_module(module)
        elif isinstance(target_module, list):
            for module in target_module:
                register_module(module)
        else:
            register_module(target_module)
                
class CallbackModuleList(nn.ModuleList):
    def __init__(self, modules=None, on_register_callback=None):
        super().__init__()
        self.on_register_callback = on_register_callback
        if modules:
            super().extend(modules)

    def append(self, module):
        if self.on_register_callback is not None:
            self.on_register_callback(module)
        super().append(module)

    def extend(self, modules):
        if self.on_register_callback is not None:
            self.on_register_callback()
        super().extend(modules)

    def insert(self, index, module):
        if self.on_register_callback is not None:
            self.on_register_callback(module)
        super().insert(index, module)

    def __setitem__(self, idx, module):
        if self.on_register_callback is not None:
            self.on_register_callback(module)
        super().__setitem__(idx, module)

# ligtning model
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
        finetune_config: Optional[Dict[str, Any]] = None,
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
        self.finetune_config = finetune_config
        self._apply_finetune_config(finetune_config)
        logger.debug(" ".join([f'{output.name.capitalize()} loss weight: {output.loss_weight}'for output in self.outputs]))

        # metrics related things
        self.metric_names_initialized = False          # for first batch
        self.metric_names_logged = False               # for first batch logging
        self.metric_names = None                       # for epoches

    @staticmethod
    def _config_get(config: Any, key: str, default: Any = None) -> Any:
        if config is None:
            return default
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    def _set_module_trainable(
        self,
        module: Optional[nn.Module],
        trainable: bool,
        prefix: str,
        changed: List[str],
    ) -> None:
        if module is None:
            return
        for name, param in module.named_parameters():
            full_name = f"{prefix}.{name}" if name else prefix
            if param.requires_grad != trainable:
                changed.append(full_name)
            param.requires_grad = trainable

    def _set_named_modules_trainable(
        self,
        names: Any,
        trainable: bool,
        changed: List[str],
    ) -> None:
        if names is None:
            return
        if isinstance(names, str):
            names = [names]
        module_dict = dict(self.model.named_modules())
        for module_name in names:
            module = module_dict.get(module_name)
            if module is None:
                logger.warning("Fine-tune module '%s' was requested but not found", module_name)
                continue
            self._set_module_trainable(module, trainable, module_name, changed)

    def _apply_finetune_config(self, finetune_config: Any) -> None:
        """Apply optional parameter freezing for fine-tuning.

        This is intentionally opt-in.  If ``finetune_config`` is omitted, every
        parameter keeps its default ``requires_grad`` value, so old configs and
        old checkpoints continue to train/load as before.
        """
        if finetune_config is None:
            return
        if self._config_get(finetune_config, "enabled", True) is False:
            return

        freeze_all = bool(self._config_get(finetune_config, "freeze_all", True))
        verbose = bool(self._config_get(finetune_config, "verbose", True))
        show_frozen = bool(self._config_get(finetune_config, "show_frozen", False))
        changed: List[str] = []

        if freeze_all:
            total_params = sum(p.numel() for p in self.model.parameters())
            for param in self.model.parameters():
                param.requires_grad = False
            if verbose:
                logger.info("Frozen all %s parameters initially", f"{total_params:,d}")

        # Whole top-level components.
        input_flag = self._config_get(finetune_config, "finetune_input_modules", None)
        if input_flag is not None:
            self._set_module_trainable(self.model.input_modules, bool(input_flag), "input_modules", changed)

        representation_flag = self._config_get(finetune_config, "finetune_representation", None)
        if representation_flag is not None:
            self._set_module_trainable(self.model.representation, bool(representation_flag), "representation", changed)

        # Representation sub-components.  These work for MACE and for other
        # representations that expose similarly named modules.
        representation = self.model.representation
        for flag_name, attr_name in (
            ("finetune_embeddings", "embeddings"),
            ("finetune_interactions", "interactions"),
            ("finetune_products", "products"),
            ("finetune_readout", "readout"),
        ):
            flag = self._config_get(finetune_config, flag_name, None)
            if flag is not None and hasattr(representation, attr_name):
                self._set_module_trainable(getattr(representation, attr_name), bool(flag), f"representation.{attr_name}", changed)

        # Backward-compatible aliases used by train-16.  The current MACE
        # readout shares the final output tensor for energy and charge, so either
        # of these flags enables the whole readout module.
        energy_readout_flag = self._config_get(finetune_config, "finetune_energy_readout", None)
        charge_readout_flag = self._config_get(finetune_config, "finetune_charge_readout", None)
        if (bool(energy_readout_flag) or bool(charge_readout_flag)) and hasattr(representation, "readout"):
            self._set_module_trainable(representation.readout, True, "representation.readout", changed)
        elif not freeze_all and energy_readout_flag is False and charge_readout_flag is False and hasattr(representation, "readout"):
            self._set_module_trainable(representation.readout, False, "representation.readout", changed)

        output_modules_flag = self._config_get(finetune_config, "finetune_output_modules", None)
        if output_modules_flag is not None:
            self._set_module_trainable(self.model.output_modules, bool(output_modules_flag), "output_modules", changed)

        charge_eq_flag = self._config_get(finetune_config, "finetune_charge_equilibration", None)
        rescale_flag = self._config_get(finetune_config, "finetune_rescale", None)
        for idx, module in enumerate(self.model.output_modules):
            module_name = f"output_modules.{idx}"
            class_name = type(module).__name__.lower()
            is_charge_equilibration = (
                hasattr(module, "electronegativity_mlp")
                or hasattr(module, "hardness_mlp")
                or "chargeequilibration" in class_name
            )
            is_rescale = hasattr(module, "scale") and hasattr(module, "unscale")
            if charge_eq_flag is not None and is_charge_equilibration:
                self._set_module_trainable(module, bool(charge_eq_flag), module_name, changed)
            if rescale_flag is not None and is_rescale:
                self._set_module_trainable(module, bool(rescale_flag), module_name, changed)

        self._set_named_modules_trainable(self._config_get(finetune_config, "finetune_modules", None), True, changed)
        self._set_named_modules_trainable(self._config_get(finetune_config, "freeze_modules", None), False, changed)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        frozen_params = total_params - trainable_params
        if verbose:
            unfrozen_names = [name for name, p in self.model.named_parameters() if p.requires_grad]
            frozen_names = [name for name, p in self.model.named_parameters() if not p.requires_grad]
            logger.info("Unfrozen parameters: %s", unfrozen_names)
            if show_frozen:
                logger.info("Frozen parameters: %s", frozen_names)
            logger.info(
                "Fine-tuning setup complete:\n"
                "  Total parameters: %s\n"
                "  Trainable (requires_grad=True): %s\n"
                "  Frozen (requires_grad=False): %s",
                f"{total_params:,d}",
                f"{trainable_params:,d}",
                f"{frozen_params:,d}",
            )
        
    def setup(self, stage: Optional[str]=None) -> None:
        if stage == "fit":
            # if not self.model._initialized:  # need to initialize modules everytime to update scales and atomic energies
            self.model.initialize_modules(self.trainer.datamodule)
            self.rescale_layers = []
            for layer in self.model.output_modules:
                if hasattr(layer, "unscale"):
                    self.rescale_layers.append(layer)

        if getattr(self.trainer, "is_global_zero", True):
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(
                "Model summary: %s(representation=%s, trainable_parameters=%s, total_parameters=%s)",
                type(self.model).__name__,
                type(self.model.representation).__name__,
                f"{trainable_params:,d}",
                f"{total_params:,d}",
            )
            if os.environ.get("CURATOR_LOG_FULL_MODEL", "0") == "1":
                logger.info(self.model)
            logger.debug(f"Model parameters: {trainable_params:,d}")
    
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

    def on_train_start(self):
        logger.info("\n")
        logger.debug("Start training model")
        if getattr(self.trainer, "is_global_zero", True):
            logger.info(f"{self.optimizers()}")
    
    # def on_validation_start(self):
    #     logger.info("\nStart validating model")
        
    def on_test_start(self):
        logger.info("\n")
        logger.debug("Start testing model")

    def on_train_epoch_start(self):
        logger.info("\n")
        logger.debug("Training")
        if self.metric_names is not None:
            head = [f'# epoch      batch']
            logger.info("".join(head + [f'{m:>16s}' for m in self.metric_names]))
        
    def on_validation_epoch_start(self):
        torch.set_grad_enabled(True)
        logger.info("\n")
        logger.debug("Validating")
        if self.metric_names is not None:
            head = [f'# epoch      batch']
            logger.info("".join(head + [f'{m:>16s}' for m in self.metric_names]))
    
    def on_test_epoch_start(self):
        torch.set_grad_enabled(True)
        logger.info("\n")
        logger.debug("Testing")
        head = [f'# epoch      batch']
        logger.info("".join(head + [f'{m:>16s}' for m in self.metric_names]))
    
    def training_step(self, batch: Dict, batch_idx: List[int]) -> torch.Tensor:
        pred = self.model(batch)
        pred.update({k: v for k, v in batch.items() if k not in pred.keys()})

        # calculate loss, metrics
        # unscale batch because loss will be calculated with normalized units
        unscaled_batch = batch
        for layer in self.rescale_layers:
            unscaled_batch = layer.unscale(unscaled_batch, force_process=True)
        loss_dict, num_abs_dict = self.loss_fn(pred, unscaled_batch, 'train')
        for k in loss_dict.keys():
            self.log(k, loss_dict[k].detach().cpu().item(), batch_size=num_abs_dict[k], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # when calculate metrics pred need to be scaled to get real units
        scaled_pred = pred
        for layer in self.rescale_layers[::-1]:
            scaled_pred = layer.scale(scaled_pred, force_process=True)
        
        all_metrics = {}
        for output in self.outputs:
            for k, v in output.calculate_metrics(scaled_pred, batch, 'train').items():
                all_metrics[k] = v
        self.log_dict(all_metrics, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # get metric names for first epoch
        self.log_head(loss_dict, all_metrics, 'train')

        # logging metrics to console
        if batch_idx % self.trainer.log_every_n_steps == 0:
            msgs = [f'{self.current_epoch:>7d}', f'{batch_idx:>11d}']
            forward_cache = [f'{metric._forward_cache or 0.0:>16.3g}' for metric in self.trainer._results.values()]
            logger.info("".join(msgs + forward_cache))
            
        return loss_dict['train_total_loss']

    def validation_step(self, batch: Dict, batch_idx: List[int]) -> torch.Tensor:
        pred = self.model(batch)
        pred.update({k: v for k, v in batch.items() if k not in pred.keys()})
        
        # calculate loss, metrics
        # both batch and pred need to be normalized for calculating loss in validation mode
        unscaled_batch, unscaled_pred = batch, pred
        # import pdb; pdb.set_trace()
        for layer in self.rescale_layers:
            unscaled_batch = layer.unscale(unscaled_batch, force_process=True)
            unscaled_pred = layer.unscale(unscaled_pred, force_process=True)
        loss_dict, num_abs_dict = self.loss_fn(unscaled_pred, unscaled_batch, 'val')
        for k in loss_dict.keys():
            self.log(k, loss_dict[k].detach().cpu().item(), batch_size=num_abs_dict[k], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True) 
        # import pdb; pdb.set_trace()
        # nothing need to be scaled for calculating metrics        
        all_metrics = {}
        for output in self.outputs:
            for k, v in output.calculate_metrics(pred, batch, 'val').items():
                all_metrics[k] = v
        # import pdb; pdb.set_trace()
        self.log_dict(all_metrics, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        # import pdb; pdb.set_trace()
        # get metric names for first epoch
        self.log_head(loss_dict, all_metrics, 'val')
        
        # logging metrics to console
        if batch_idx % self.trainer.log_every_n_steps == 0:
            msgs = [f'{self.current_epoch:>7d}', f'{batch_idx:>11d}']
            forward_cache = [f'{metric._forward_cache or 0.0:>16.3g}' for metric in self.trainer._results.values()]
            logger.info("".join(msgs + forward_cache))
        
        return loss_dict['val_total_loss']
    
    def test_step(self, batch: Dict, batch_idx: List[int]) -> torch.Tensor:
        pred = self.model(batch)
        pred.update({k: v for k, v in batch.items() if k not in pred.keys()})
        
        # calculate loss, metrics
        # both targets and pred need to be normalized for calculating loss in validation mode
        unscaled_batch, unscaled_pred = batch, pred
        for layer in self.rescale_layers:
            unscaled_targets = layer.unscale(unscaled_batch, force_process=True)
            unscaled_pred = layer.unscale(unscaled_pred, force_process=True)
        loss_dict, num_abs_dict = self.loss_fn(unscaled_pred, unscaled_targets, 'test')
        for k in loss_dict.keys():
            self.log(k, loss_dict[k].detach().cpu().item(), batch_size=num_abs_dict[k], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True) 
               
        for output in self.outputs:
            batch_metrics = output.calculate_metrics(pred, batch, 'test')
            self.log_dict(batch_metrics, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # logging metrics to console
        if batch_idx % self.trainer.log_every_n_steps == 0:
            msgs = [f'{self.current_epoch:>7d}', f'{batch_idx:>11d}']
            forward_cache = [f'{metric._forward_cache or 0.0:>16.3g}' for metric in self.trainer._results.values()]
            logger.info("".join(msgs + forward_cache))
        
        return loss_dict['test_loss']
    
    def on_train_epoch_end(self):
        msgs = ['Train       # epoch'] + [f'{m:>16s}' for m in self.metric_names]
        metrics = [f'{self.current_epoch:>19d}']
        metrics += [f'{metric.compute():>16.3g}' for metric in self.trainer._results.values()]
        # reset metrics
        for output in self.outputs:
            output.reset_metrics(subset='train')
        # skip collecting metric names after first epoch
        if not self.metric_names_initialized:
            self.metric_names_initialized = True

        logger.info("".join(msgs))
        logger.info("".join(metrics))
    
    def on_validation_epoch_end(self):
        # validation end goes before train epoch
        msgs = ['Validation  # epoch'] + [f'{m:>16s}' for m in self.metric_names]
        metrics = [f'{self.current_epoch:>19d}']
        metrics += [f'{metric.compute():>16.3g}' for metric in self.trainer._results.values()]
        # reset metrics
        for output in self.outputs:
            output.reset_metrics(subset='val')
        logger.info("\n")
        logger.debug("Epoch summary:")
        logger.info("".join(msgs))
        logger.info("".join(metrics))
    
    def log_head(self, loss_dict, metrics_dict, stage='train'):
        # get metric names for first epoch
        if not self.metric_names_initialized:
            metric_names = [k.replace(stage + '_', '') for k in loss_dict.keys()]
            metric_names += [k.replace(stage + '_', '') for k in metrics_dict.keys()]

            # collect metric names
            if self.metric_names is None:
                self.metric_names = metric_names
            else:
                for name in metric_names:
                    if name not in self.metric_names:
                        self.metric_names.append(name)

            if not self.metric_names_logged:
                metric_names = "".join([f'{m:>16s}' for m in self.metric_names])
                metric_names = f'# epoch      batch' + metric_names
                logger.info(metric_names)
                self.metric_names_logged = True

    def save_configuration(self, config: DictConfig):
        self.config = config
        
    def on_save_checkpoint(self, checkpoint):
        checkpoint['data_params'] = self.config.data
        checkpoint['model_params'] = self.config.model
        checkpoint['outputs'] = self.outputs
        checkpoint['optimizer'] = self.optimizer
        if self.save_entire_model:
            checkpoint['model'] = self.model
    
    def configure_optimizers(self) -> Type[torch.optim.Optimizer]:
        from curator.model import MACE
        def trainable_parameters(parameters):
            return [param for param in parameters if param.requires_grad]

        if type(self.model.representation) == MACE:
            decay_params = {}
            no_decay_params = {}

            for name, param in self.model.representation.interactions.named_parameters():
                if "linear.weight" in name or "skip_tp_full.weight":
                    decay_params[name] = param
                else:
                    no_decay_params[name] = param

            param_group = [
                {
                    "name": "input_modules",
                    "params": trainable_parameters(self.model.input_modules.parameters()),
                    "weight_decay": 0.0,
                },
                {
                    "name": "embeddings",
                    "params": trainable_parameters(self.model.representation.embeddings.parameters()),
                    "weight_decay": self.optimizer.keywords['weight_decay'],
                },
                {
                    "name": "decay_params",
                    "params": trainable_parameters(decay_params.values()),
                    "weight_decay": self.optimizer.keywords['weight_decay'],
                },
                {
                    "name": "no_decay_params",
                    "params": trainable_parameters(no_decay_params.values()),
                    "weight_decay": 0.0,
                },
                {
                    "name": "products",
                    "params": trainable_parameters(self.model.representation.products.parameters()),
                    "weight_decay": self.optimizer.keywords['weight_decay'],
                },
                {
                    "name": "readout",
                    "params": trainable_parameters(self.model.representation.readout.parameters()),
                    "weight_decay": 0.0,
                },
                {
                    "name": "output_modules",
                    "params": trainable_parameters(self.model.output_modules.parameters()),
                    "weight_decay": 0.0,
                },
            ]
            param_group = [group for group in param_group if len(group["params"]) > 0]
            if len(param_group) == 0:
                raise ValueError("No trainable parameters found. Check task.finetune_config.")
            optimizer = self.optimizer(params=param_group)
        else:
            params = trainable_parameters(self.parameters())
            if len(params) == 0:
                raise ValueError("No trainable parameters found. Check task.finetune_config.")
            optimizer = self.optimizer(params=params)
        # optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            lr_scheduler = {"scheduler": scheduler}
            if self.scheduler_monitor:
                lr_scheduler["monitor"] = self.scheduler_monitor
            if self._trainer is not None:
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