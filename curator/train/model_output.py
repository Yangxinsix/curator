from torch import nn
from typing import Optional, Dict
import copy
import torchmetrics
from torchmetrics import Metric
import torch
from collections import OrderedDict
from .metrics import AtomsMetric

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

        self.loss = 0.0
        self.num_obs = 0

    def calculate_loss(self, pred: Dict, target: Dict, return_num_obs=True) -> torch.Tensor:
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0

        loss = self.loss_weight * self.loss_fn(
            pred[self.name], target[self.target_property]
        )
        num_obs = target[self.target_property].view(-1).shape[0]
        
        if return_num_obs:
            return loss, num_obs

        self.loss += loss.item() * num_obs
        self.num_obs += num_obs

        return loss

    def update_metrics(self, pred: Dict, target: Dict, subset: str) -> None:
        for metric in self.metrics[subset].values():
            metric(pred[self.name], target[self.target_property])

    def calculate_metrics(self, pred: Dict, target: Dict, subset: str) -> None:
        batch_val = OrderedDict()
        for k in self.metrics[subset]:
            if isinstance(self.metrics[subset][k], AtomsMetric):
                metric = self.metrics[subset][k](pred, target)
                for k2, v in metric.items():
                    batch_val[f"{subset}_{self.name}_{k2}"] = v
            else:
                metric = self.metrics[subset][k](pred[self.name].detach(), target[self.target_property].detach())
                batch_val[f"{subset}_{self.name}_{k}"] = metric
        
        return batch_val
    
    def accumulate_loss(self):
        loss = self.loss / self.num_obs
        self.loss = 0.0
        self.num_obs = 0
        return loss
    
    def accumulate_metrics(self, subset):
        all_metrics = {}
        for k, v in self.metrics[subset].items():
            all_metrics[k] = v.compute()
        return all_metrics
    
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