import torch
from torch import nn
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage
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
        
    def forward(self, data: properties.Type) -> properties.Type:
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
    
    def collect_outputs(self):
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


class ModelOutput(nn.Module):
    """ Base class for model outputs."""
    def __init__(
        self,
        name: str,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: float = 1.0,
        metrics: Optional[Dict[str, Metric]] = None,
        target_property: Optional[str]=None,
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
        self.val_metrics = nn.ModuleDict(copy.deepcopy(metrics))
        self.test_metrics = nn.ModuleDict(copy.deepcopy(metrics))
        self.metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }
    
    def calculate_loss(self, pred: Dict, target: Dict) -> torch.Tensor:
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0

        loss = self.loss_weight * self.loss_fn(
            pred[self.name], target[self.target_property]
        )
        return loss

    def update_metrics(self, pred: Dict, target: Dict, subset: str) -> None:
        for metric in self.metrics[subset].values():
            metric(pred[self.name].detach(), target[self.target_property].detach())
            
    def add_metrics(self, metrics: Dict[str, Metric]):
        for k in self.metrics:
            self.metrics[k].update(metrics)


class LitNNP(pl.LightningModule):
    """ Base class for neural network potentials using PyTorch Lightning."""
    def __init__(
        self,
        model: NeuralNetworkPotential,
        outputs: List[ModelOutput],
        optimizer: Type[torch.optim.Optimizer],
        scheduler: Optional[Type] = None,
        scheduler_monitor: Optional[str] = None,
        use_ema: bool=False,
        ema_decay: Optional[float] = None,
        warmup_steps: int = 0,
    ) -> None:
        """ Base class for neural network potentials using PyTorch Lightning.

        Args:
            model (NeuralNetworkPotential): Neural network potential model
            outputs (List[ModelOutput]): List of model outputs
            optimizer (Type[torch.optim.Optimizer]): Optimizer
            scheduler (Optional[Type], optional): Scheduler. Defaults to None.
            scheduler_monitor (Optional[str], optional): Scheduler monitor. Defaults to None.
            use_ema (bool, optional): Use exponential moving average. Defaults to False.
            ema_decay (Optional[float], optional): EMA decay. Defaults to None.
            warmup_steps (int, optional): Warmup steps. Defaults to 0.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.outputs = nn.ModuleList(outputs)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_monitor = scheduler_monitor
        self.warmup_steps = warmup_steps
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema = None
        
    def setup(self, stage: Optional[str]=None) -> None:
        if stage == "fit":
            if not self.model._initialized:
                self.model.initialize_modules(self.trainer.datamodule)
    
    def loss_fn(self, pred: Dict, batch: Dict) -> torch.Tensor:
        loss = 0.0
        for output in self.outputs:
            loss += output.calculate_loss(pred, batch)
        return loss
    
    def log_metrics(self, pred: Dict, batch: Dict, subset: str) -> None:
        for output in self.outputs:
            output.update_metrics(pred, batch, subset)
            for metric_name, metric in output.metrics[subset].items():
                self.log(
                    f"{subset}_{output.name}_{metric_name}",
                    metric,
                    on_step=(subset == "train"),
                    on_epoch=(subset != "train"),
                    prog_bar=False,
                )
    
    def training_step(self, batch: Dict, batch_idx: List[int]) -> torch.Tensor:
        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
        }
        pred = self.model(batch)
        loss = self.loss_fn(pred, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log_metrics(pred, targets, "train")
        
        if self.use_ema:
            self.ema.update()
            
        return loss
    
    def validation_step(self, batch: Dict, batch_idx: List[int]) -> torch.Tensor:
        torch.set_grad_enabled(True)

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
        }
        pred = self.model(batch)
        loss = self.loss_fn(pred, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(pred, targets, "val")
        return loss
    
    def test_step(self, batch: Dict, batch_idx: List[int]) -> torch.Tensor:
        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
        }
        pred = self.model(batch)
        loss = self.loss_fn(pred, targets)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(pred, targets, "test")
        return loss
    
    def save_configuration(self, config: DictConfig):
        self.config = config
        
    def on_save_checkpoint(self, checkpoint):
        checkpoint['data_params'] = self.config.data
        checkpoint['mdoel_params'] = self.config.model
    
    def configure_optimizers(self) -> Type[torch.optim.Optimizer]:
        optimizer = self.optimizer(params=self.parameters())
        if self.use_ema and self.ema is None:
            self.ema = ExponentialMovingAverage(
                self.parameters(),
                decay=self.ema_decay,
            )
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