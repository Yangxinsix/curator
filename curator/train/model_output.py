from pathlib import Path
from torch import nn
from typing import Optional, Dict, Any, Callable
import copy
from torchmetrics import Metric
import torch
from collections import OrderedDict
from omegaconf import OmegaConf
from .metrics import AtomsMetric
from .sampling import OutputSampler

class ModelOutput(nn.Module):
    """ Base class for model outputs."""
    def __init__(
        self,
        name: str,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: float = 1.0,
        metrics: Optional[Dict[str, Metric]] = None,
        prediction_property: Optional[str] = None,
        target_property: Optional[str] = None,
        num_samples: Optional[int] = None,
        sample_indices: Optional[Any] = None,
        sample_index_key: Optional[str] = None,
        sample_fn: Optional[Callable] = None,
        is_penalty: bool = False,
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
        self.prediction_property = prediction_property or name
        self.target_property = target_property or name
        self.is_penalty = is_penalty
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        self.sampler = OutputSampler(
            num_samples=num_samples,
            sample_indices=sample_indices,
            sample_index_key=sample_index_key,
            sample_fn=sample_fn,
        )
        if metrics is not None:
            self.train_metrics = nn.ModuleDict(metrics)
            self.val_metrics = nn.ModuleDict({k: copy.copy(v) for k, v in metrics.items()})
            self.test_metrics = nn.ModuleDict({k: copy.copy(v) for k, v in metrics.items()})
            
            # here we found a serious bug that deepcopy is not working in hydra instantiate!!!
            self.metrics = {
                "train": self.train_metrics,
                "val": self.val_metrics,
                "test": self.test_metrics,
            }
        else:
            self.metrics = None

        self.loss = 0.0
        self.num_obs = 0

    def _resolve_inputs(self, pred: Dict, target: Optional[Dict] = None, apply_sampling: bool = False):
        if target is not None:
            pred = dict(target, **pred)
            target = dict(target)
        if not apply_sampling:
            return pred, target
        return self.sampler.resolve_inputs(
            pred,
            target,
            prediction_property=self.prediction_property,
            target_property=self.target_property,
        )

    def sample(
        self,
        data: Dict,
        key: str,
        peer: Optional[Dict] = None,
        indices: Optional[Any] = None,
    ):
        return self.sampler.sample(data, key, peer=peer, indices=indices)

    def _flatten_value(self, value):
        if isinstance(value, list):
            if len(value) == 0:
                raise ValueError(f"{self.__class__.__name__} received an empty list for '{self.prediction_property}'.")
            return torch.cat([v.reshape(-1) for v in value])
        return value

    def _resolve_value_pair(
        self,
        pred: Dict,
        target: Optional[Dict] = None,
        apply_sampling: bool = False,
    ):
        pred, target = self._resolve_inputs(pred, target, apply_sampling=apply_sampling)
        pred_value = self._flatten_value(pred[self.prediction_property])
        if target is None:
            return pred_value, None
        target_value = self._flatten_value(target[self.target_property])
        return pred_value, target_value

    def calculate_loss(self, pred: Dict, target: Optional[Dict] = None, return_num_obs=True) -> torch.Tensor:
        if self.loss_weight == 0:
            return 0.0

        pred_value, target_value = self._resolve_value_pair(pred, target, apply_sampling=True)
        if self.is_penalty:
            loss = self.loss_weight * pred_value.square().mean()
            num_obs = 1
        elif self.loss_fn is not None:
            loss = self.loss_weight * self.loss_fn(
                pred_value, target_value
            )
            num_obs = target_value.view(-1).shape[0]
        else:
            return 0.0

        self.loss += loss.item() * num_obs
        self.num_obs += num_obs

        if return_num_obs:
            return loss, num_obs
        return loss

    def update_metrics(self, pred: Dict, target: Dict, subset: str) -> None:
        # If metrics is None, do nothing
        if self.metrics is None:
            return
        
        # If the subset does not exist (e.g. "train", "val", "test"), skip
        if subset not in self.metrics:
            return

        pred, target = self._resolve_inputs(pred, target, apply_sampling=False)
        pred_value = self._flatten_value(pred[self.prediction_property])
        target_value = self._flatten_value(target[self.target_property])
        
        for metric in self.metrics[subset].values():
            if isinstance(metric, AtomsMetric):
                metric(pred, target)
            else:
                metric(
                    pred_value,
                    target_value,
                )

    def calculate_metrics(self, pred: Dict, target: Dict, subset: str) -> None:
        if self.metrics is None:
            return {}

        pred, target = self._resolve_inputs(pred, target, apply_sampling=False)
        pred_value = self._flatten_value(pred[self.prediction_property]).detach()
        target_value = self._flatten_value(target[self.target_property]).detach()
        
        batch_val = OrderedDict()
        for k in self.metrics[subset]:
            if isinstance(self.metrics[subset][k], AtomsMetric):
                metric = self.metrics[subset][k](pred, target)
                for k2, v in metric.items():
                    batch_val[f"{subset}_{self.name}_{k2}"] = v
            else:
                metric = self.metrics[subset][k](
                    pred_value,
                    target_value,
                )
                batch_val[f"{subset}_{self.name}_{k}"] = metric
        
        return batch_val
    
    def accumulate_loss(self):
        loss = self.loss / self.num_obs
        return loss
    
    def accumulate_metrics(self, subset):
        if self.metrics is None or subset not in self.metrics:
            return {}
        
        all_metrics = {}
        for k, v in self.metrics[subset].items():
            all_metrics[k] = v.compute()
        return all_metrics
    
    def reset_loss(self) -> None:
        self.loss = 0.0
        self.num_obs = 0

    def reset_metrics(self, subset: Optional[str]=None) -> None:
        if self.metrics is None:
            return
        
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


class DistillOutput(ModelOutput):
    """Minimal teacher-student distillation loss for a single property."""

    def __init__(
        self,
        name: str,
        teacher_model_path: Optional[str],
        student_property: Optional[str] = None,
        teacher_property: Optional[str] = None,
        teacher_output_property: Optional[str] = None,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: float = 1.0,
        metrics: Optional[Dict[str, Metric]] = None,
        num_samples: Optional[int] = None,
        sample_indices: Optional[Any] = None,
        sample_index_key: Optional[str] = None,
        sample_fn: Optional[Callable] = None,
        teacher_cfg: Optional[Any] = None,
        only_train: bool = True,
        cache_key: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=name,
            loss_fn=loss_fn if loss_fn is not None else nn.MSELoss(),
            loss_weight=loss_weight,
            metrics=metrics,
            prediction_property=student_property or name,
            target_property=student_property or name,
            num_samples=num_samples,
            sample_indices=sample_indices,
            sample_index_key=sample_index_key,
            sample_fn=sample_fn,
            is_penalty=False,
        )
        self.teacher_model_path = str(teacher_model_path) if teacher_model_path is not None else None
        self.student_property = student_property or name
        self.teacher_property = teacher_property or self.student_property
        self.teacher_output_property = teacher_output_property or self.student_property
        self.teacher_cfg = (
            OmegaConf.create(teacher_cfg)
            if isinstance(teacher_cfg, dict)
            else teacher_cfg
        )
        self.only_train = bool(only_train)
        self.cache_key = cache_key or self.teacher_model_path or self.name
        self._teacher_state: Dict[str, Any] = {}

    def _teacher_target_from_batch(self, target: Dict) -> torch.Tensor:
        if self.teacher_property not in target:
            raise KeyError(
                f"Offline distillation batch is missing '{self.teacher_property}'. "
                f"Available keys: {sorted(target.keys())}"
            )
        return target[self.teacher_property]

    def _resolve_teacher_model(self, reference: torch.Tensor):
        teacher = self._teacher_state.get("model")
        if teacher is None:
            from curator.utils import find_best_model, load_model

            if not self.teacher_model_path:
                raise ValueError(f"{self.__class__.__name__} requires `teacher_model_path`.")
            teacher_path = Path(self.teacher_model_path)
            if teacher_path.is_dir():
                resolved = find_best_model(teacher_path)
                if resolved is None:
                    raise FileNotFoundError(
                        f"Could not find a teacher checkpoint under '{teacher_path}'."
                    )
                teacher_path = resolved[0]

            teacher = load_model(
                teacher_path,
                device=reference.device,
                load_compiled=False,
                load_weights_only=False,
                cfg=self.teacher_cfg,
            )
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            self._teacher_state["model"] = teacher
            self._teacher_state["rescale_layers"] = [
                layer for layer in getattr(teacher, "output_modules", []) if hasattr(layer, "unscale")
            ]

        teacher.to(device=reference.device)
        if reference.is_floating_point():
            teacher.to(dtype=reference.dtype)
        return teacher

    def _teacher_prediction(self, pred: Dict, target: Dict) -> Dict:
        cache = target.setdefault("__distill_teacher_cache__", {})
        cached = cache.get(self.cache_key)
        if cached is not None:
            return cached

        student_value = pred[self.student_property]
        teacher = self._resolve_teacher_model(student_value)
        teacher_input = {
            key: value
            for key, value in target.items()
            if not (isinstance(key, str) and key.startswith("__distill_teacher_cache__"))
        }
        teacher_pred = teacher(teacher_input)
        for layer in self._teacher_state.get("rescale_layers", []):
            teacher_pred = layer.unscale(teacher_pred, force_process=True)

        cache[self.cache_key] = teacher_pred
        return teacher_pred

    def _resolve_inputs(self, pred: Dict, target: Optional[Dict] = None, apply_sampling: bool = False):
        if target is None:
            raise ValueError("DistillOutput requires the current batch as `target`.")

        source_target = target
        pred = dict(source_target, **pred)
        target = dict(source_target)

        if self.teacher_model_path:
            teacher_pred = self._teacher_prediction(pred, source_target)
            if self.teacher_output_property not in teacher_pred:
                raise KeyError(
                    f"Teacher prediction is missing '{self.teacher_output_property}'. "
                    f"Available keys: {sorted(teacher_pred.keys())}"
                )
            teacher_value = teacher_pred[self.teacher_output_property]
        else:
            teacher_value = self._teacher_target_from_batch(target)

        target[self.target_property] = teacher_value
        if not apply_sampling:
            return pred, target
        return self.sampler.resolve_inputs(
            pred,
            target,
            prediction_property=self.prediction_property,
            target_property=self.target_property,
        )

    def calculate_loss(self, pred: Dict, target: Optional[Dict] = None, return_num_obs=True) -> torch.Tensor:
        student_value, target_value = self._resolve_value_pair(pred, target, apply_sampling=True)
        num_obs = student_value.view(-1).shape[0]
        zero = student_value.new_zeros(())

        if self.loss_weight == 0 or (self.only_train and not self.training):
            if return_num_obs:
                return zero, num_obs
            return zero

        loss = self.loss_weight * self.loss_fn(student_value, target_value)
        self.loss += loss.item() * num_obs
        self.num_obs += num_obs

        if return_num_obs:
            return loss, num_obs
        return loss
