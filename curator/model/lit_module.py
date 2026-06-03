import torch
import os
from torch import nn
from typing import List, Optional, Dict, Type, Any, Union, Tuple
from curator.data import properties
from curator.train.model_output import ModelOutput
import warnings
import pytorch_lightning as pl
from omegaconf import DictConfig
import logging
from collections import OrderedDict
from ase.data import atomic_numbers, chemical_symbols

logger = logging.getLogger(__name__)    # console output
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from curator.model.base import (
    NeuralNetworkPotential,
    ParameterGroup,
    collect_unique_parameters,
)
from curator.layer.wrappers import collect_adapter_parameter_groups, get_model_wrapper_config

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
        optimizer_groups: Optional[Dict[str, Dict[str, Any]]] = None,
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
        self.optimizer_groups = dict(optimizer_groups or {})
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
            if not self.model._initialized:
                self.model.initialize_modules(self.trainer.datamodule)
            self.rescale_layers = []
            for layer in self.model.output_modules:
                if hasattr(layer, "unscale"):
                    self.rescale_layers.append(layer)
            self._domain_loss_scales = self._build_domain_loss_scales()
            self._log_runtime_configuration()
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
        self._write_log_only("\n")
        logger.debug("Start training model")
        logger.info(f"{self.optimizers()}")
    
    # def on_validation_start(self):
    #     logger.info("\nStart validating model")
        
    def on_test_start(self):
        self._write_log_only("\n")
        logger.debug("Start testing model")

    def on_train_epoch_start(self):
        self._write_log_only("\n")
        self._write_log_and_console("Training", level=logging.DEBUG, progress=False)
        self._debug_rescale_logged["train"] = False
        if self.metric_names is not None:
            self._write_log_only(self._format_header(include_loader=True))
        
    def on_validation_epoch_start(self):
        torch.set_grad_enabled(True)
        self._write_log_only("\n")
        self._write_log_and_console("Validation", level=logging.DEBUG, progress=False)
        self._debug_rescale_logged["val"] = False
        if self.metric_names is not None:
            self._write_log_only(self._format_header(include_loader=True))
    
    def on_test_epoch_start(self):
        torch.set_grad_enabled(True)
        self._write_log_only("\n")
        self._write_log_only("Testing")
        self._debug_rescale_logged["test"] = False
        self._write_log_only(self._format_header(include_loader=True))
    
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
                        prog_bar=False,
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
                    self._write_log_only(self._format_row(self.current_epoch, batch_idx, self._loader_label(idx), values))

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
                prog_bar=(k == "train_total_loss"),
                sync_dist=True,
            )
        self.log_dict(batch_metrics, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        self.log_head(loss_dict, batch_metrics, 'train')

        if batch_idx % self.trainer.log_every_n_steps == 0:
            values = self._format_metric_values("train", loss_dict, batch_metrics)
            self._write_log_only(self._format_row(self.current_epoch, batch_idx, self._loader_label(0), values))

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
            self.log(
                k,
                loss_dict[k].detach().cpu().item(),
                batch_size=num_abs_dict[k],
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        
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
            self._write_log_only(self._format_row(self.current_epoch, batch_idx, self._loader_label(dataloader_idx), values))
        
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
            self.log(
                k,
                loss_dict[k].detach().cpu().item(),
                batch_size=num_abs_dict[k],
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
               
        batch_metrics = {}
        for output in self.outputs:
            batch_metrics.update(output.calculate_metrics(pred, batch, 'test'))
        self.log_dict(batch_metrics, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # logging metrics to console
        if batch_idx % self.trainer.log_every_n_steps == 0:
            values = self._format_metric_values("test", loss_dict, batch_metrics)
            self._write_log_only(self._format_row(self.current_epoch, batch_idx, self._loader_label(dataloader_idx), values))
        
        return loss_dict['test_loss']
    
    def on_train_epoch_end(self):
        pass
    
    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        
        # used for monitoring total loss in multi-domain training
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "domain_modules"):
            metrics = self.trainer.callback_metrics
            total = 0.0
            weight_sum = 0.0
            domain_to_id = getattr(dm, "domain_to_id", {})
            for name, module in dm.domain_modules.items():
                dom_id = domain_to_id.get(name)
                if dom_id is None or module.val_dataset is None:
                    continue
                key = f"val_total_loss_epoch/dataloader_idx_{dom_id}"
                if key not in metrics:
                    continue
                val = metrics[key]
                if torch.is_tensor(val):
                    val = val.detach().cpu()
                    val = val.mean().item() if val.numel() > 1 else val.item()
                weight = len(module.val_dataset)
                total += float(val) * weight
                weight_sum += weight
            if weight_sum > 0:
                self.log("val_total_loss", total / weight_sum, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        self._write_log_only("\n")
        self._write_log_and_console("Epoch summary", level=logging.DEBUG, progress=False)
        header = self._format_epoch_header()
        self._write_log_only(header)
        self._write_console(header)
        train_loader_indices = sorted(self._collect_dataloader_indices("train"))
        if train_loader_indices:
            for loader_idx in train_loader_indices:
                values = self._format_epoch_metric_values("train", loader_idx)
                row = self._format_epoch_row("Train", self.current_epoch, self._loader_label(loader_idx), values)
                self._write_log_only(row)
                self._write_console(row)
        else:
            values = self._format_epoch_metric_values("train", 0)
            row = self._format_epoch_row("Train", self.current_epoch, self._loader_label(0), values)
            self._write_log_only(row)
            self._write_console(row)

        loader_indices = sorted(self._collect_dataloader_indices("val"))
        if loader_indices:
            for loader_idx in loader_indices:
                values = self._format_epoch_metric_values("val", loader_idx)
                row = self._format_epoch_row("Validation", self.current_epoch, self._loader_label(loader_idx), values)
                self._write_log_only(row)
                self._write_console(row)
        else:
            values = self._format_epoch_metric_values("val", 0)
            row = self._format_epoch_row("Validation", self.current_epoch, self._loader_label(0), values)
            self._write_log_only(row)
            self._write_console(row)

        for output in self.outputs:
            output.reset_metrics(subset='train')
            output.reset_metrics(subset='val')
        if not self.metric_names_initialized:
            self.metric_names_initialized = True
        self._write_console("")

    def on_test_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        self._write_log_only("\n")
        self._write_log_and_console("Epoch summary", level=logging.DEBUG, progress=False)
        header = self._format_epoch_header()
        self._write_log_only(header)
        self._write_console(header)
        loader_indices = sorted(self._collect_dataloader_indices("test"))
        if loader_indices:
            for loader_idx in loader_indices:
                values = self._format_epoch_metric_values("test", loader_idx)
                row = self._format_epoch_row("Test", self.current_epoch, self._loader_label(loader_idx), values)
                self._write_log_only(row)
                self._write_console(row)
        else:
            values = self._format_epoch_metric_values("test", 0)
            row = self._format_epoch_row("Test", self.current_epoch, self._loader_label(0), values)
            self._write_log_only(row)
            self._write_console(row)
        for output in self.outputs:
            output.reset_metrics(subset='test')
        self._write_console("")

    def save_configuration(self, config: DictConfig):
        self.config = config
        
    def on_save_checkpoint(self, checkpoint):
        checkpoint['data_params'] = self.config.data
        checkpoint['model_params'] = self.config.model
        checkpoint['wrapper_config'] = get_model_wrapper_config(self.model).to_dict()
        checkpoint['outputs'] = self.outputs
        checkpoint['optimizer'] = self.optimizer
        if self.save_entire_model:
            checkpoint['model'] = self.model

    def _optimizer_parameter_groups(self) -> List[ParameterGroup]:
        adapter_groups = collect_adapter_parameter_groups(self.model)
        adapter_param_ids = {id(param) for group in adapter_groups for param in group.params}

        groups: List[ParameterGroup] = []
        for group in self.model.parameter_groups():
            params = [param for param in group.params if id(param) not in adapter_param_ids]
            if params:
                groups.append(
                    ParameterGroup(
                        name=group.name,
                        params=params,
                        defaults=dict(group.defaults) if group.defaults else None,
                    )
                )

        groups.extend(adapter_groups)
        seen = {id(param) for group in groups for param in group.params}
        extra_params = collect_unique_parameters([self.outputs], seen=seen)
        if extra_params:
            groups.append(ParameterGroup(name="task_outputs", params=extra_params))
        return [group for group in groups if group.params]

    def _build_optimizer_param_groups(self) -> List[Dict[str, Any]]:
        parameter_groups = self._optimizer_parameter_groups()
        available = {group.name for group in parameter_groups}
        optimizer_groups = dict(self.optimizer_groups or {})
        optimizer_groups.pop("_delete_", None)

        alias_map = {
            "readout": ("readout_domains", "readout_shared"),
            "output_modules": ("output_domains", "output_shared"),
        }
        for source_name, target_names in alias_map.items():
            if source_name not in optimizer_groups:
                continue
            for target_name in target_names:
                if target_name in available and target_name not in optimizer_groups:
                    optimizer_groups[target_name] = dict(optimizer_groups[source_name])

        unknown = sorted(set(optimizer_groups) - available)
        if unknown:
            warnings.warn(
                "Ignoring optimizer group overrides with no matching parameters: "
                f"{unknown}. Available groups: {sorted(available)}"
            )

        param_groups: List[Dict[str, Any]] = []
        for parameter_group in parameter_groups:
            optimizer_group: Dict[str, Any] = {
                "name": parameter_group.name,
                "params": list(parameter_group.params),
            }
            if parameter_group.defaults:
                optimizer_group.update(dict(parameter_group.defaults))
            overrides = optimizer_groups.get(optimizer_group["name"])
            if overrides:
                optimizer_group.update(dict(overrides))
            param_groups.append(optimizer_group)
        return param_groups
    
    def configure_optimizers(self) -> Type[torch.optim.Optimizer]:
        param_groups = self._build_optimizer_param_groups()
        default_lr = self._resolve_optimizer_default_lr()
        trainable_groups = []
        frozen_groups = []
        for group in param_groups:
            name = str(group.get("name", "<unnamed>"))
            lr = group.get("lr", default_lr)
            lr = None if lr is None else float(lr)
            has_trainable_param = any(getattr(param, "requires_grad", False) for param in group.get("params", []))
            if has_trainable_param and (lr is None or lr > 0.0):
                trainable_groups.append(name)
            else:
                frozen_groups.append(name)
        logger.info(
            "Trainable groups: %s",
            ", ".join(trainable_groups) if trainable_groups else "<none>",
        )
        if frozen_groups:
            logger.info("Frozen groups: %s", ", ".join(frozen_groups))

        optimizer = self.optimizer(params=param_groups)
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

    def _resolve_optimizer_default_lr(self) -> Optional[float]:
        optimizer_factory = self.optimizer
        keywords = getattr(optimizer_factory, "keywords", None)
        if isinstance(keywords, dict) and keywords.get("lr") is not None:
            return float(keywords["lr"])
        defaults = getattr(optimizer_factory, "defaults", None)
        if isinstance(defaults, dict) and defaults.get("lr") is not None:
            return float(defaults["lr"])
        return None

    # ------------------------------------------------------------------ #
    # Logging and debug helpers
    # ------------------------------------------------------------------ #
    def _log_runtime_configuration(self) -> None:
        wrapper_cfg = get_model_wrapper_config(self.model)
        logger.info(
            "Model wrapper backend: %s (adapter=%s)",
            wrapper_cfg.backend,
            wrapper_cfg.adapter,
        )
        self._log_rescale_configuration()

    @staticmethod
    def _format_numeric_sequence(values: List[float]) -> str:
        return "[" + ", ".join(f"{float(value):.6g}" for value in values) + "]"

    @staticmethod
    def _format_species_values(
        values: torch.Tensor,
        *,
        default: float,
        species_numbers: Optional[List[int]] = None,
    ) -> str:
        entries = []
        values_cpu = values.detach().cpu().reshape(-1)
        if species_numbers is None:
            indices = range(1, min(len(values_cpu), len(chemical_symbols)))
        else:
            indices = species_numbers
        for z in indices:
            if z <= 0 or z >= min(len(values_cpu), len(chemical_symbols)):
                continue
            value = float(values_cpu[z].item())
            entries.append(f"{chemical_symbols[z]}={value:.10g}")
        if not entries:
            for z in range(1, min(len(values_cpu), len(chemical_symbols))):
                value = float(values_cpu[z].item())
                if abs(value - default) <= 1e-12:
                    continue
                entries.append(f"{chemical_symbols[z]}={value:.10g}")
        return ", ".join(entries) if entries else "<none>"

    def _current_species_numbers(self) -> Optional[List[int]]:
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            return None
        species = getattr(datamodule, "species", None)
        if species in (None, "auto"):
            get_species = getattr(datamodule, "_get_species", None)
            if callable(get_species):
                species = get_species()
        if not isinstance(species, list):
            return None
        numbers = []
        for symbol in species:
            if isinstance(symbol, str) and symbol in atomic_numbers:
                numbers.append(int(atomic_numbers[symbol]))
        return numbers or None

    def _describe_global_rescale_head(self, layer: nn.Module, head_idx: int) -> str:
        head = layer.heads[head_idx]
        scale = layer.scales[head_idx].scale.detach().cpu().view(-1).tolist()
        shift_module = layer.shifts[head_idx]
        if hasattr(shift_module, "shift"):
            shift = shift_module.shift.detach().cpu().view(-1).tolist()
        else:
            shift = [0.0]

        scale_module = layer.atomic_scales[head_idx]
        shift_species_module = layer.atomic_shifts[head_idx]
        species_numbers = self._current_species_numbers()
        if hasattr(scale_module, "enabled") and bool(scale_module.enabled):
            per_species_scale = self._format_species_values(
                scale_module.values,
                default=1.0,
                species_numbers=species_numbers,
            )
        else:
            per_species_scale = "<disabled>"
        if hasattr(shift_species_module, "enabled") and bool(shift_species_module.enabled):
            per_species_shift = self._format_species_values(
                shift_species_module.values,
                default=0.0,
                species_numbers=species_numbers,
            )
        else:
            per_species_shift = "<disabled>"

        target_key = getattr(scale_module, "data_key", head.key)
        return (
            f"head={head.key} target={target_key} "
            f"scale={self._format_numeric_sequence(scale)} "
            f"shift={self._format_numeric_sequence(shift)} "
            f"per_species_scale={per_species_scale} per_species_shift={per_species_shift}"
        )

    def _log_rescale_configuration(self) -> None:
        if not hasattr(self, "rescale_layers"):
            return
        logger.info("Rescale layers: %d", len(self.rescale_layers))
        for layer_idx, layer in enumerate(self.rescale_layers):
            layer_name = layer.__class__.__name__
            logger.info("Rescale[%d]: %s", layer_idx, layer_name)
            if layer_name == "GlobalRescaleShift":
                for head_idx in range(len(layer.heads)):
                    logger.info(
                        "Rescale[%d] %s",
                        layer_idx,
                        self._describe_global_rescale_head(layer, head_idx),
                    )
            elif layer_name == "MultiDomainRescaleShift":
                for domain, domain_layer in layer.domain_modules.items():
                    logger.info("Rescale[%d] domain=%s", layer_idx, domain)
                    for head_idx in range(len(domain_layer.heads)):
                        logger.info(
                            "Rescale[%d] domain=%s %s",
                            layer_idx,
                            domain,
                            self._describe_global_rescale_head(domain_layer, head_idx),
                        )

    def _write_log_only(
        self,
        message: str,
        level: int = logging.INFO,
        progress: bool = True,
    ) -> None:
        # Log progress lines to file only; console stream filters these out.
        logger.log(level, message, extra={"progress": progress})

    def _write_console(self, message: str) -> None:
        # Write to the terminal without breaking tqdm progress bars.
        if self.trainer is not None and not getattr(self.trainer, "is_global_zero", True):
            return
        if tqdm is None:
            print(message)
        else:
            tqdm.write(message)

    def _write_log_and_console(
        self,
        message: str,
        level: int = logging.INFO,
        progress: bool = False,
    ) -> None:
        # Keep a log record and echo to the terminal.
        self._write_log_only(message, level=level, progress=progress)
        self._write_console(message)

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
                self._write_log_only(self._format_header(include_loader=True))
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
