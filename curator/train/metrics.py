import torchmetrics
from torchmetrics import Metric

import torch
from torchmetrics import Metric
from ase.data import chemical_symbols
from abc import ABC, abstractmethod

class AtomsMetric(Metric, ABC):
    @abstractmethod
    def compute_error(self, pred, target):
        """Abstract method for error calculation."""
        pass

    @abstractmethod
    def get_metric_name(self):
        """Abstract method for returning the metric name."""
        pass

class LabelWiseMetricBase(AtomsMetric):
    def __init__(self, label_key, value_key, dist_sync_on_step=False, compute_overall=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.label_key = label_key
        self.value_key = value_key
        self.compute_overall = compute_overall
        
        # Register states for label-wise metrics
        self.add_state("labels", default=[], dist_reduce_fx="cat")
        self.add_state("errors", default=[], dist_reduce_fx="cat")
        self.add_state("counts", default=[], dist_reduce_fx="cat")
        
        if self.compute_overall:
            # Register states for overall metrics
            self.add_state("overall_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("overall_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: dict, targets: dict):
        # Assume preds and targets are dictionaries with 'label' and 'value' keys
        labels = preds[self.label_key]
        pred_values = preds[self.value_key]
        target_values = targets[self.value_key]

        unique_labels = labels.unique()
        for label in unique_labels:
            mask = (labels == label)
            pred_for_label = pred_values[mask]
            target_for_label = target_values[mask]
            error_for_label = self.compute_error(pred_for_label, target_for_label)
            count_for_label = mask.sum()
            if label in self.labels:
                index = self.labels.index(label)
                self.errors[index] += error_for_label.sum()
                self.counts[index] += count_for_label
            else:
                self.labels.append(label)
                self.errors.append(error_for_label.sum())
                self.counts.append(count_for_label)
            
            if self.compute_overall:
                # stupid torchmetrics bug
                self.overall_error.to(pred_values.device)
                self.overall_count.to(pred_values.device)
                self.overall_error += error_for_label.sum()
                self.overall_count += count_for_label

    def compute(self):
        # Calculate the metric for each label
        results = {}

        if self.compute_overall:
            results[self.get_metric_name()] = (self.overall_error / self.overall_count).item()        

        for label, error, count in zip(self.labels, self.errors, self.counts):
            results[self.get_metric_name() + '_' + chemical_symbols[label]] = (error / count).item()
        
        return results

    def reset(self):
        # Reset the metrics for the next epoch or batch
        self.labels = []
        self.errors = []
        self.counts = []
        
        if self.compute_overall:
            self.overall_error = torch.tensor(0.0, device=self.overall_error.device)
            self.overall_count = torch.tensor(0, device=self.overall_count.device)

    def compute_error(self, pred, target):
        raise NotImplementedError("This method should be implemented by subclasses")

    def get_metric_name(self):
        raise NotImplementedError("This method should be implemented by subclasses")

class PerSpeciesMAE(LabelWiseMetricBase):
    def compute_error(self, pred, target):
        # MAE error calculation
        return torch.abs(pred - target)

    def get_metric_name(self):
        return "mae"

class PerSpeciesRMSE(LabelWiseMetricBase):
    def compute_error(self, pred, target):
        # RMSE error calculation (we'll accumulate squared errors and compute square root later)
        return (pred - target) ** 2

    def compute(self):
        # Calculate RMSE for each label by taking the square root of the mean squared error
        results = {}

        if self.compute_overall:
            results["rmse"] = torch.sqrt(self.overall_error / self.overall_count).item()

        for label, error, count in zip(self.labels, self.errors, self.counts):
            mean_squared_error = error / count
            results['rmse_' + chemical_symbols[label]] = torch.sqrt(mean_squared_error).item()
        
        return results

    def get_metric_name(self):
        return "rmse"

class SizeIndependentMetricBase(AtomsMetric):
    def __init__(self, size_key, value_key, dist_sync_on_step=False, compute_overall=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.size_key = size_key
        self.value_key = value_key
        self.compute_overall = compute_overall
        
        # Register states for size-independent metrics
        self.add_state("size_independent_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("size_independent_count", default=torch.tensor(0), dist_reduce_fx="sum")
        
        if self.compute_overall:
            # Register states for overall metrics without size division
            self.add_state("size_dependent_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("size_dependent_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: dict, targets: dict):
        # Assume preds and targets are dictionaries with 'size' and 'value' keys
        normalized_pred = preds[self.value_key] / preds[self.size_key]
        normalized_target = targets[self.value_key] / targets[self.size_key]

        self.size_independent_error += self.compute_error(normalized_pred, normalized_target).sum()
        self.size_independent_count += preds[self.value_key].shape[0]

        if self.compute_overall:
            self.size_dependent_error += self.compute_error(preds[self.value_key], targets[self.value_key]).sum()
            self.size_dependent_count += preds[self.value_key].shape[0]

    def compute(self):
        # Calculate size-independent metric
        size_independent_metric = self.size_independent_error / self.size_independent_count
        
        results = {self.get_metric_name(): size_independent_metric.item()}

        if self.compute_overall:
            size_dependent_metric = self.size_dependent_error / self.size_dependent_count
            results[self.get_overall_metric_name()] = size_dependent_metric.item()
        
        return results

    def reset(self):
        # Reset the metrics for the next epoch or batch
        self.size_independent_error = torch.tensor(0.0, device=self.size_independent_error.device)
        self.size_independent_count = torch.tensor(0, device=self.size_independent_count.device)
        
        if self.compute_overall:
            self.size_dependent_error = torch.tensor(0.0, device=self.size_dependent_error.device)
            self.size_dependent_count = torch.tensor(0, device=self.size_dependent_count.device)

    def compute_error(self, pred, target):
        raise NotImplementedError("This method should be implemented by subclasses")

    def get_metric_name(self):
        raise NotImplementedError("This method should be implemented by subclasses")

    def get_overall_metric_name(self):
        raise NotImplementedError("This method should be implemented by subclasses")
    
class PerAtomMAE(SizeIndependentMetricBase):
    def compute_error(self, pred, target):
        # MAE error calculation
        return torch.abs(pred - target)

    def get_metric_name(self):
        return "mae_pa"

    def get_overall_metric_name(self):
        return "mae"
    
class PerAtomRMSE(SizeIndependentMetricBase):
    def compute_error(self, pred, target):
        # RMSE error calculation
        return (pred - target) ** 2

    def compute(self):
        # Calculate RMSE from accumulated squared errors
        size_independent_rmse = torch.sqrt(self.size_independent_error / self.size_independent_count)
        
        results = {self.get_metric_name(): size_independent_rmse.item()}

        if self.compute_overall:
            size_dependent_rmse = torch.sqrt(self.size_dependent_error / self.size_dependent_count)
            results[self.get_overall_metric_name()] = size_dependent_rmse.item()
        
        return results

    def get_metric_name(self):
        return "rmse_pa"

    def get_overall_metric_name(self):
        return "rmse"
