import torch
import re
import os
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Type, Any, Union, Callable, Tuple
from functools import partial
from curator.data import properties
from curator.train.model_output import ModelOutput
from pytorch_lightning.utilities.types import STEP_OUTPUT
import warnings
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from omegaconf import DictConfig
import logging
from collections import OrderedDict, defaultdict
import inspect
try:
    from torch_scatter import scatter_add, scatter_mean
except ImportError:
    from curator.utils import scatter_add, scatter_mean

logger = logging.getLogger(__name__)    # console output
class Representation(nn.Module):
    """
    Shared mixin/base to standardize handling of head configs and readout instantiation
    across representations (MACE/Nequip/PAINN).
    """

    def __init__(self, heads: Optional[list] = None) -> None:
        super().__init__()
        self.heads = heads or []

    def _instantiate_readout(
        self,
        readout: Union[nn.Module, Type[nn.Module], Callable],
        heads: Optional[list] = None,
        **kwargs,
    ) -> nn.Module:
        """Instantiate readout, passing heads when supported."""
        if isinstance(readout, nn.Module):
            # assume already configured
            return readout

        call = readout
        if isinstance(readout, partial):
            call = readout.func

        sig = inspect.signature(call)

        def maybe(name, value):
            return {name: value} if name in sig.parameters and value is not None else {}

        init_kwargs = dict(kwargs)
        init_kwargs.update(maybe("heads", heads))

        return readout(**init_kwargs)  # type: ignore[arg-type]

    @staticmethod
    def _enable_cueq(use_cueq: bool):
        """Helper to enable cuequivariance with a single warning path."""
        if not use_cueq:
            return
        from curator.layer._cuequivariance_wrapper import IS_CUET_AVAILABLE, set_use_cueq
        import warnings

        set_use_cueq(use_cueq)
        if use_cueq and not IS_CUET_AVAILABLE:
            warnings.warn(
                "Requested use_cueq=True but cuequivariance is not available; falling back to e3nn kernels.",
                RuntimeWarning,
            )

    @staticmethod
    def _apply_cutoff_mask(data: properties.Type, cutoff: float):
        """Apply edge cutoff mask in-place. Returns original (edge_idx, edge_diff, edge_dist) for optional downstream use."""
        try:
            edge_idx = data[properties.edge_idx]
            edge_diff = data[properties.edge_diff]
            edge_dist = data[properties.edge_dist]
        except KeyError:
            return None
        mask = edge_dist < cutoff
        data[properties.edge_idx] = edge_idx[mask]
        data[properties.edge_diff] = edge_diff[mask]
        data[properties.edge_dist] = edge_dist[mask]
        return (edge_idx, edge_diff, edge_dist)

    @staticmethod
    def _restore_cutoff_mask(data: properties.Type, cache):
        """Restore edges previously masked by _apply_cutoff_mask."""
        if cache is None:
            return
        edge_idx, edge_diff, edge_dist = cache
        data[properties.edge_idx] = edge_idx
        data[properties.edge_diff] = edge_diff
        data[properties.edge_dist] = edge_dist

class NeuralNetworkPotential(nn.Module):
    """ Base class for neural network potentials."""
    def __init__(
        self,
        representation: nn.Module,
        input_modules: List[nn.Module] = None,
        output_modules: List[nn.Module] = None,
        model_outputs: List[str] = [],
        heads: Optional[list] = None,
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
        self.heads = heads
        self._initialized: bool = False
        self.collect_outputs()
        self.register_callbacks()
        
    def forward(self, data: properties.Type, force_domain: Optional[Union[str, int]] = None) -> properties.Type:
        data = data.copy()
        if force_domain is not None:
            dom = torch.tensor([int(force_domain)], dtype=torch.long, device=data[properties.n_atoms].device)
            data[properties.domain] = dom
        for m in self.input_modules:
            data = m(data)
            
        data = self.representation(data)
        
        for m in self.output_modules:
            data = m(data)
        
        return self.extract_outputs(data)

    def initialize_modules(self, datamodule: LightningDataModule) -> None:
        for module in self.modules():
            if hasattr(module, "setup_from_datamodule"):
                module.setup_from_datamodule(datamodule)
            elif hasattr(module, "datamodule"):
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
        if 'all' in self.model_outputs:
            return data
        else: 
            return {k: data[k] for k in self.model_outputs}
    
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
        normalize_domain_loss: Union[bool, str] = False,
        debug_rescale: bool = False,
        debug_rescale_path: Optional[str] = None,
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
        self.normalize_domain_loss = normalize_domain_loss
        self._debug_rescale = bool(debug_rescale)
        self._debug_rescale_path = debug_rescale_path or "rescale_debug.log"
        self._rescale_debug_logger = self._init_rescale_debug_logger()
        logger.debug(" ".join([f'{output.name.capitalize()} loss weight: {output.loss_weight}'for output in self.outputs]))
        self._domain_loss_scales = {}

        # metrics related things
        self.metric_names_initialized = False          # for first batch
        self.metric_names_logged = False               # for first batch logging
        self.metric_names = None                       # for epoches
        self._debug_rescale_logged = {"train": False, "val": False, "test": False}
        self._col_widths = {
            "epoch": 8,
            "batch": 12,
            "domain": 10,
            "metric": 16,
            "stage": 12,
        }
        
    def setup(self, stage: Optional[str]=None) -> None:
        if stage == "fit":
            # if not self.model._initialized:  # need to initialize modules everytime to update scales and atomic energies
            self.model.initialize_modules(self.trainer.datamodule)
            self.rescale_layers = []
            for layer in self.model.output_modules:
                if hasattr(layer, "unscale"):
                    self.rescale_layers.append(layer)
            self._domain_loss_scales = self._build_domain_loss_scales()
        logger.info(self.model)
        logger.debug(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,d}")
    
    def loss_fn(self, pred: Dict, batch: Dict, subset: str):
        loss_dict = OrderedDict()
        loss_dict[subset + '_total_loss'] = 0.0
        num_obs_dict = OrderedDict()
        num_obs_dict[subset + '_total_loss'] = 1
        for output in self.outputs:
            key = subset + '_' + output.name + '_loss'
            loss_dict[key], num_obs_dict[key] = output.calculate_loss(pred, batch, True)
            loss_scale = self._loss_scale_factor(output, batch)
            if loss_scale != 1.0:
                loss_dict[key] = loss_dict[key] * loss_scale
            loss_dict[subset + '_total_loss'] += loss_dict[key]
            
        return loss_dict, num_obs_dict

    def _is_combined_batch(self, batch: Dict) -> bool:
        if not isinstance(batch, dict):
            return False
        if properties.n_atoms in batch:
            return False
        return all(isinstance(v, dict) for v in batch.values())

    def _apply_batch_weight(self, loss_dict: Dict, num_obs_dict: Dict, batch: Dict) -> None:
        weight = self._get_batch_weight(batch)
        if weight == 1.0:
            return
        for k in list(loss_dict.keys()):
            loss_dict[k] = loss_dict[k] * weight

    def _get_batch_weight(self, batch: Dict) -> float:
        weight = batch.get("weight", 1.0)
        if torch.is_tensor(weight):
            return float(weight.mean().item())
        try:
            return float(weight)
        except Exception:
            return 1.0

    def _build_domain_loss_scales(self) -> Dict[str, Dict[str, float]]:
        scales_by_domain: Dict[str, Dict[str, float]] = {}
        for layer in self.rescale_layers:
            if hasattr(layer, "domain_modules"):
                for dom, mod in layer.domain_modules.items():
                    scales_by_domain[str(dom)] = self._extract_scale_map(mod)
            elif layer.__class__.__name__ == "GlobalRescaleShift":
                scales_by_domain.setdefault("0", self._extract_scale_map(layer))
        return scales_by_domain

    def _extract_scale_map(self, module: nn.Module) -> Dict[str, float]:
        scale_map: Dict[str, float] = {}
        for sc in getattr(module, "scales", []):
            key = getattr(sc, "key", None)
            if key is None:
                continue
            val = getattr(sc, "scale", None)
            if torch.is_tensor(val):
                val = float(val.detach().cpu().view(-1)[0].item())
            else:
                val = float(val) if val is not None else 1.0
            scale_map[str(key)] = val
        return scale_map

    def _get_domain_id(self, batch: Dict) -> str:
        dom = None
        if properties.domain in batch:
            dom = batch[properties.domain]
        elif properties.domain_atom in batch:
            dom = batch[properties.domain_atom]
        if dom is None:
            return "0"
        if torch.is_tensor(dom):
            if dom.numel() == 0:
                return "0"
            dom = dom.view(-1)[0].item()
        return str(dom)

    def _loss_scale_factor(self, output: ModelOutput, batch: Dict) -> float:
        mode = self._get_domain_loss_mode()
        if mode is None:
            return 1.0
        if output.is_penalty:
            return 1.0
        dom = self._get_domain_id(batch)
        scale_map = self._domain_loss_scales.get(dom) or self._domain_loss_scales.get("0") or {}
        key = output.target_property or output.name
        scale = scale_map.get(key, None)
        if scale is None or scale == 0:
            return 1.0
        if isinstance(output.loss_fn, nn.L1Loss):
            power = 1.0
        else:
            power = 2.0
        if mode == "relative":
            return float(scale) ** (-power)
        return float(scale) ** power

    def _get_domain_loss_mode(self) -> Optional[str]:
        if self.normalize_domain_loss is False or self.normalize_domain_loss is None:
            return None
        if self.normalize_domain_loss is True:
            return "physical"
        mode = str(self.normalize_domain_loss).strip().lower()
        if mode in ("physical", "relative"):
            return mode
        return None

    def on_train_start(self):
        logger.info("\n")
        logger.debug("Start training model")
        logger.info(f"{self.optimizers()}")
    
    # def on_validation_start(self):
    #     logger.info("\nStart validating model")
        
    def on_test_start(self):
        logger.info("\n")
        logger.debug("Start testing model")

    def on_train_epoch_start(self):
        logger.info("\n")
        logger.debug("Training")
        self._debug_rescale_logged["train"] = False
        if self.metric_names is not None:
            logger.info(self._format_header(include_loader=True))
        
    def on_validation_epoch_start(self):
        torch.set_grad_enabled(True)
        logger.info("\n")
        logger.debug("Validating")
        self._debug_rescale_logged["val"] = False
        if self.metric_names is not None:
            logger.info(self._format_header(include_loader=True))
    
    def on_test_epoch_start(self):
        torch.set_grad_enabled(True)
        logger.info("\n")
        logger.debug("Testing")
        self._debug_rescale_logged["test"] = False
        logger.info(self._format_header(include_loader=True))
    
    def training_step(self, batch: Dict, batch_idx: List[int]) -> torch.Tensor:
        if self._is_combined_batch(batch):
            total_loss = torch.tensor(0.0, device=self.device)
            total_num = 0
            for idx, sub_batch in enumerate(batch.values()):
                if sub_batch is None:
                    continue
                loss_dict, num_abs_dict, batch_metrics = self._train_step_single(sub_batch)
                total_loss = total_loss + loss_dict["train_total_loss"]
                total_num += num_abs_dict.get("train_total_loss", 0)

                for k in loss_dict.keys():
                    key = f"{k}/dataloader_idx_{idx}"
                    self.log(
                        key,
                        loss_dict[k].detach().cpu().item(),
                        batch_size=max(1, num_abs_dict.get(k, 1)),
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True,
                        sync_dist=True,
                        add_dataloader_idx=False,
                    )
                log_metrics = {f"{k}/dataloader_idx_{idx}": v for k, v in batch_metrics.items()}
                self.log_dict(log_metrics, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, add_dataloader_idx=False)

                display_loss = dict(loss_dict)
                display_metrics = dict(batch_metrics)
                self.log_head(display_loss, display_metrics, "train")
                if batch_idx % self.trainer.log_every_n_steps == 0:
                    values = self._format_metric_values("train", display_loss, display_metrics)
                    logger.info(self._format_row(self.current_epoch, batch_idx, self._loader_label(idx), values))

            self.log(
                "train_total_loss",
                total_loss.detach().cpu().item(),
                batch_size=max(1, total_num),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            return total_loss

        loss_dict, num_abs_dict, batch_metrics = self._train_step_single(batch)

        for k in loss_dict.keys():
            self.log(
                k,
                loss_dict[k].detach().cpu().item(),
                batch_size=max(1, num_abs_dict[k]),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        self.log_dict(batch_metrics, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        self.log_head(loss_dict, batch_metrics, 'train')

        if batch_idx % self.trainer.log_every_n_steps == 0:
            values = self._format_metric_values("train", loss_dict, batch_metrics)
            logger.info(self._format_row(self.current_epoch, batch_idx, self._loader_label(0), values))

        return loss_dict['train_total_loss']

    def _train_step_single(self, batch: Dict) -> Tuple[Dict, Dict, Dict]:
        pred = self.model(batch)
        pred.update({k: v for k, v in batch.items() if k not in pred.keys()})

        debug = self._should_debug_rescale("train")
        if debug:
            self._log_rescale_debug("train", batch=batch, pred=pred, note="pre-unscale")

        unscaled_batch = batch
        for layer in self.rescale_layers:
            unscaled_batch = layer.unscale(unscaled_batch, force_process=True)
        if debug:
            self._log_rescale_debug("train", batch=unscaled_batch, pred=pred, note="post-unscale")
        loss_dict, num_abs_dict = self.loss_fn(pred, unscaled_batch, 'train')
        self._apply_batch_weight(loss_dict, num_abs_dict, batch)

        scaled_pred = pred
        for layer in self.rescale_layers[::-1]:
            scaled_pred = layer.scale(scaled_pred, force_process=True)
        if debug:
            self._log_rescale_debug("train", batch=unscaled_batch, pred=scaled_pred, note="post-scale")

        batch_metrics = {}
        for output in self.outputs:
            batch_metrics.update(output.calculate_metrics(scaled_pred, batch, 'train'))

        return loss_dict, num_abs_dict, batch_metrics

    # no combined helper; keep training_step logic inline for clarity

    def validation_step(self, batch: Dict, batch_idx: List[int], dataloader_idx: int = 0) -> torch.Tensor:
        pred = self.model(batch)
        pred.update({k: v for k, v in batch.items() if k not in pred.keys()})

        debug = self._should_debug_rescale("val")
        if debug:
            self._log_rescale_debug("val", batch=batch, pred=pred, note="pre-unscale")
        
        # calculate loss, metrics
        # both batch and pred need to be normalized for calculating loss in validation mode
        unscaled_batch, unscaled_pred = batch, pred
        for layer in self.rescale_layers:
            unscaled_batch = layer.unscale(unscaled_batch, force_process=True)
            unscaled_pred = layer.unscale(unscaled_pred, force_process=True)
        if debug:
            self._log_rescale_debug("val", batch=unscaled_batch, pred=unscaled_pred, note="post-unscale")
        loss_dict, num_abs_dict = self.loss_fn(unscaled_pred, unscaled_batch, 'val')
        self._apply_batch_weight(loss_dict, num_abs_dict, batch)
        for k in loss_dict.keys():
            self.log(k, loss_dict[k].detach().cpu().item(), batch_size=num_abs_dict[k], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True) 
        
        # nothing need to be scaled for calculating metrics        
        batch_metrics = {}
        for output in self.outputs:
            batch_metrics.update(output.calculate_metrics(pred, batch, 'val'))
        self.log_dict(batch_metrics, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # get metric names for first epoch
        self.log_head(loss_dict, batch_metrics, 'val')
        
        # logging metrics to console
        if batch_idx % self.trainer.log_every_n_steps == 0:
            values = self._format_metric_values("val", loss_dict, batch_metrics)
            logger.info(self._format_row(self.current_epoch, batch_idx, self._loader_label(dataloader_idx), values))
        
        return loss_dict['val_total_loss']
    
    def test_step(self, batch: Dict, batch_idx: List[int], dataloader_idx: int = 0) -> torch.Tensor:
        pred = self.model(batch)
        pred.update({k: v for k, v in batch.items() if k not in pred.keys()})

        debug = self._should_debug_rescale("test")
        if debug:
            self._log_rescale_debug("test", batch=batch, pred=pred, note="pre-unscale")
        
        # calculate loss, metrics
        # both targets and pred need to be normalized for calculating loss in validation mode
        unscaled_batch, unscaled_pred = batch, pred
        for layer in self.rescale_layers:
            unscaled_targets = layer.unscale(unscaled_batch, force_process=True)
            unscaled_pred = layer.unscale(unscaled_pred, force_process=True)
        if debug:
            self._log_rescale_debug("test", batch=unscaled_targets, pred=unscaled_pred, note="post-unscale")
        loss_dict, num_abs_dict = self.loss_fn(unscaled_pred, unscaled_targets, 'test')
        self._apply_batch_weight(loss_dict, num_abs_dict, batch)
        for k in loss_dict.keys():
            self.log(k, loss_dict[k].detach().cpu().item(), batch_size=num_abs_dict[k], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True) 
               
        batch_metrics = {}
        for output in self.outputs:
            batch_metrics.update(output.calculate_metrics(pred, batch, 'test'))
        self.log_dict(batch_metrics, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # logging metrics to console
        if batch_idx % self.trainer.log_every_n_steps == 0:
            values = self._format_metric_values("test", loss_dict, batch_metrics)
            logger.info(self._format_row(self.current_epoch, batch_idx, self._loader_label(dataloader_idx), values))
        
        return loss_dict['test_loss']
    
    def on_train_epoch_end(self):
        pass
    
    def on_validation_epoch_end(self):
        logger.info("\n")
        logger.debug("Epoch summary:")
        logger.info(self._format_epoch_header())
        train_loader_indices = sorted(self._collect_dataloader_indices("train"))
        if train_loader_indices:
            for loader_idx in train_loader_indices:
                values = self._format_epoch_metric_values("train", loader_idx)
                logger.info(self._format_epoch_row("Train", self.current_epoch, self._loader_label(loader_idx), values))
        else:
            values = self._format_epoch_metric_values("train", 0)
            logger.info(self._format_epoch_row("Train", self.current_epoch, self._loader_label(0), values))

        loader_indices = sorted(self._collect_dataloader_indices("val"))
        if loader_indices:
            for loader_idx in loader_indices:
                values = self._format_epoch_metric_values("val", loader_idx)
                logger.info(self._format_epoch_row("Validation", self.current_epoch, self._loader_label(loader_idx), values))
        else:
            values = self._format_epoch_metric_values("val", 0)
            logger.info(self._format_epoch_row("Validation", self.current_epoch, self._loader_label(0), values))

        for output in self.outputs:
            output.reset_metrics(subset='train')
            output.reset_metrics(subset='val')
        if not self.metric_names_initialized:
            self.metric_names_initialized = True

    def on_test_epoch_end(self):
        logger.info("\n")
        logger.debug("Epoch summary:")
        logger.info(self._format_epoch_header())
        loader_indices = sorted(self._collect_dataloader_indices("test"))
        if loader_indices:
            for loader_idx in loader_indices:
                values = self._format_epoch_metric_values("test", loader_idx)
                logger.info(self._format_epoch_row("Test", self.current_epoch, self._loader_label(loader_idx), values))
        else:
            values = self._format_epoch_metric_values("test", 0)
            logger.info(self._format_epoch_row("Test", self.current_epoch, self._loader_label(0), values))
        for output in self.outputs:
            output.reset_metrics(subset='test')

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
                    "params": self.model.input_modules.parameters(),
                    "weight_decay": 0.0,
                },
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
                {
                    "name": "products",
                    "params": self.model.representation.products.parameters(),
                    "weight_decay": self.optimizer.keywords['weight_decay'],
                },
                {
                    "name": "readout",
                    "params": self.model.representation.readout.parameters(),
                    "weight_decay": 0.0,
                },
                {
                    "name": "output_modules",
                    "params": self.model.output_modules.parameters(),
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

    # ------------------------------------------------------------------ #
    # Logging and debug helpers
    # ------------------------------------------------------------------ #
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
                logger.info(self._format_header(include_loader=True))
                self.metric_names_logged = True

    def _format_metric_values(self, stage: str, loss_dict: Dict, metrics_dict: Dict) -> List[float]:
        combined = {}
        combined.update(loss_dict or {})
        combined.update(metrics_dict or {})
        values: List[float] = []
        for name in self.metric_names or []:
            key = f"{stage}_{name}"
            if key in combined:
                val = combined[key]
            else:
                val = combined.get(name, 0.0)
            if torch.is_tensor(val):
                val = val.detach().cpu()
                val = val.mean().item() if val.numel() > 1 else val.item()
            values.append(float(val))
        return values

    def _format_header(self, include_loader: bool) -> str:
        parts = [
            f'{"# epoch":>{self._col_widths["epoch"]}s}',
            f'{"batch":>{self._col_widths["batch"]}s}',
        ]
        if include_loader:
            parts.append(f'{"domain":>{self._col_widths["domain"]}s}')
        parts.append("".join([f'{m:>{self._col_widths["metric"]}s}' for m in self.metric_names or []]))
        return "".join(parts)

    def _format_row(self, epoch: int, batch: int, loader: Optional[str], values: List[float]) -> str:
        parts = [
            f'{epoch:>{self._col_widths["epoch"]}d}',
            f'{batch:>{self._col_widths["batch"]}d}',
        ]
        if loader is not None:
            parts.append(f'{loader:>{self._col_widths["domain"]}s}')
        parts.extend([f'{v:>{self._col_widths["metric"]}.3g}' for v in values])
        return "".join(parts)

    def _format_epoch_header(self) -> str:
        parts = [
            f'{"stage":>{self._col_widths["stage"]}s}',
            f'{"epoch":>{self._col_widths["epoch"]}s}',
            f'{"domain":>{self._col_widths["domain"]}s}',
        ]
        parts.append("".join([f'{m:>{self._col_widths["metric"]}s}' for m in self.metric_names or []]))
        return "".join(parts)

    def _format_epoch_row(self, stage_label: str, epoch: int, loader: Optional[str], values: List[float]) -> str:
        loader_str = "" if loader is None else str(loader)
        parts = [
            f'{stage_label:>{self._col_widths["stage"]}s}',
            f'{epoch:>{self._col_widths["epoch"]}d}',
            f'{loader_str:>{self._col_widths["domain"]}s}',
        ]
        parts.extend([f'{v:>{self._col_widths["metric"]}.3g}' for v in values])
        return "".join(parts)

    def _loader_label(self, loader_idx: Optional[int]) -> Optional[str]:
        if loader_idx is None:
            return None
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "id_to_domain"):
            return str(dm.id_to_domain.get(int(loader_idx), loader_idx))
        return str(loader_idx)

    def _collect_dataloader_indices(self, stage: str) -> List[int]:
        indices = set()
        for key in self.trainer.callback_metrics.keys():
            key_str = str(key)
            marker = "dataloader_idx_"
            if stage in key_str and marker in key_str:
                try:
                    idx = int(key_str.split(marker, 1)[1].split()[0].strip())
                    indices.add(idx)
                except ValueError:
                    continue
        return sorted(indices)

    def _format_epoch_metric_values(self, stage: str, loader_idx: Optional[int]) -> List[float]:
        values: List[float] = []
        metrics = self.trainer.callback_metrics
        for name in self.metric_names or []:
            base = f"{stage}_{name}"
            key_epoch = f"{base}_epoch"
            key_step = f"{base}"
            if loader_idx is not None:
                key_epoch_idx = f"{key_epoch}/dataloader_idx_{loader_idx}"
                key_step_idx = f"{key_step}/dataloader_idx_{loader_idx}"
            else:
                key_epoch_idx = None
                key_step_idx = None
            if key_epoch_idx is not None and key_epoch_idx in metrics:
                val = metrics[key_epoch_idx]
            elif key_step_idx is not None and key_step_idx in metrics:
                val = metrics[key_step_idx]
            elif key_epoch in metrics:
                val = metrics[key_epoch]
            elif key_step in metrics:
                val = metrics[key_step]
            else:
                val = 0.0
            if torch.is_tensor(val):
                val = val.detach().cpu()
                val = val.mean().item() if val.numel() > 1 else val.item()
            values.append(float(val))
        return values

    def _should_debug_rescale(self, stage: str) -> bool:
        if not self._debug_rescale:
            return False
        if stage not in self._debug_rescale_logged:
            return False
        if self._debug_rescale_logged[stage]:
            return False
        self._debug_rescale_logged[stage] = True
        return True

    def _log_rescale_debug(self, stage: str, batch: Dict, pred: Optional[Dict] = None, note: str = "") -> None:
        if not self._debug_rescale or self._rescale_debug_logger is None:
            return
        prefix = f"[rescale-debug:{stage}]"
        note_str = f" {note}" if note else ""
        self._rescale_debug_logger.debug(f"{prefix}{note_str} training={self.training} batch_keys={list(batch.keys())}")
        self._log_domain_debug(prefix, batch)
        self._log_rescale_layers(prefix, batch)
        self._log_target_stats(prefix, batch, label="batch")
        if pred is not None:
            self._log_target_stats(prefix, pred, label="pred")

    def _log_domain_debug(self, prefix: str, batch: Dict) -> None:
        dom = None
        if properties.domain in batch:
            dom = batch[properties.domain]
        elif properties.domain_atom in batch:
            dom = batch[properties.domain_atom]
        if dom is None:
            self._rescale_debug_logger.debug(f"{prefix} domain: <missing>")
            return
        if torch.is_tensor(dom):
            unique = torch.unique(dom.detach().cpu()).tolist()
            self._rescale_debug_logger.debug(f"{prefix} domain tensor unique={unique}")
        else:
            self._rescale_debug_logger.debug(f"{prefix} domain value={dom}")

    def _log_rescale_layers(self, prefix: str, batch: Dict) -> None:
        self._rescale_debug_logger.debug(f"{prefix} rescale_layers={len(self.rescale_layers)}")
        for layer in self.rescale_layers:
            layer_name = layer.__class__.__name__
            self._rescale_debug_logger.debug(f"{prefix} layer={layer_name} training={layer.training}")
            if layer_name == "GlobalRescaleShift":
                self._log_global_rescale(prefix, layer)
            elif layer_name == "MultiDomainRescaleShift":
                self._log_multi_domain_rescale(prefix, layer, batch)

    def _log_global_rescale(self, prefix: str, layer: nn.Module) -> None:
        for i, head in enumerate(layer.heads):
            scale = layer.scales[i].scale.detach().cpu().view(-1).tolist()
            shift_mod = layer.shifts[i]
            if hasattr(shift_mod, "shift"):
                shift_val = shift_mod.shift.detach().cpu().view(-1).tolist()
            else:
                shift_val = [0.0]
            per_species = layer.atomic_shifts[i]
            if hasattr(per_species, "enabled") and bool(per_species.enabled):
                nonzero = int((per_species.values != 0).sum().item())
                per_species_info = f"enabled nonzero={nonzero}"
            else:
                per_species_info = "disabled"
            self._rescale_debug_logger.debug(
                f"{prefix} head={head.key} scale={scale} shift={shift_val} per_species={per_species_info}"
            )

    def _log_multi_domain_rescale(self, prefix: str, layer: nn.Module, batch: Dict) -> None:
        dom = None
        if properties.domain in batch:
            dom = batch[properties.domain]
        elif properties.domain_atom in batch:
            dom = batch[properties.domain_atom]
        if torch.is_tensor(dom):
            dom_val = dom.view(-1)[0].item() if dom.numel() > 0 else None
        elif dom is None:
            dom_val = None
        else:
            dom_val = dom
        domains = list(layer.domain_modules.keys())
        if dom_val is None:
            dom_val = "0" if "0" in layer.domain_modules else (domains[0] if domains else "0")
        dom_val = str(dom_val)
        self._rescale_debug_logger.debug(f"{prefix} multi-domain selected={dom_val} available={domains}")
        if dom_val in layer.domain_modules:
            self._log_global_rescale(prefix, layer.domain_modules[dom_val])

    def _log_target_stats(self, prefix: str, data: Dict, label: str) -> None:
        keys = [
            properties.energy,
            properties.forces,
            properties.stress,
            properties.virial,
            properties.atomic_energy,
            properties.total_charge,
            properties.atomic_charge,
        ]
        for key in keys:
            if key not in data:
                continue
            val = data[key]
            if not torch.is_tensor(val) or val.numel() == 0:
                self._rescale_debug_logger.debug(f"{prefix} {label}.{key}: <non-tensor or empty>")
                continue
            t = val.detach().float().cpu()
            mean = t.mean().item()
            std = t.std(unbiased=False).item()
            minv = t.min().item()
            maxv = t.max().item()
            self._rescale_debug_logger.debug(
                f"{prefix} {label}.{key}: shape={tuple(t.shape)} mean={mean:.6g} std={std:.6g} min={minv:.6g} max={maxv:.6g}"
            )

    def _init_rescale_debug_logger(self) -> Optional[logging.Logger]:
        if not self._debug_rescale:
            return None
        rescale_logger = logging.getLogger("curator.rescale_debug")
        rescale_logger.setLevel(logging.DEBUG)
        rescale_logger.propagate = False
        log_path = os.path.abspath(self._debug_rescale_path)
        has_handler = False
        for handler in rescale_logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_path:
                has_handler = True
                break
        if not has_handler:
            file_handler = logging.FileHandler(log_path, mode="a")
            formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
            file_handler.setFormatter(formatter)
            rescale_logger.addHandler(file_handler)
        return rescale_logger
