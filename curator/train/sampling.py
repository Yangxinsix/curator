from typing import Optional, Dict, Any, Callable

import torch
from torch import nn


class OutputSampler(nn.Module):
    def __init__(
        self,
        num_samples: Optional[int] = None,
        sample_indices: Optional[Any] = None,
        sample_index_key: Optional[str] = None,
        sample_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.sample_indices = sample_indices
        self.sample_index_key = sample_index_key
        self.sample_fn = sample_fn

    def enabled(self) -> bool:
        return (
            self.num_samples is not None
            or self.sample_indices is not None
            or self.sample_index_key is not None
            or self.sample_fn is not None
        )

    def sampled_key(self, key: str) -> str:
        return f"{key}_sampled"

    def sample_indices_key(self, key: str) -> str:
        return self.sample_index_key or f"{key}_sample_indices"

    def resolve_sample_indices(
        self,
        data: Dict,
        key: str,
        peer: Optional[Dict] = None,
        indices: Optional[Any] = None,
    ):
        if indices is not None:
            return indices
        if self.sample_index_key is not None:
            if self.sample_index_key in data:
                return data[self.sample_index_key]
            if peer is not None and self.sample_index_key in peer:
                return peer[self.sample_index_key]
            raise KeyError(
                f"{self.__class__.__name__} requires sample index key '{self.sample_index_key}' "
                f"for '{key}', but it was not found in either view."
            )
        default_key = f"{key}_sample_indices"
        if default_key in data:
            return data[default_key]
        if peer is not None and default_key in peer:
            return peer[default_key]
        return self.sample_indices

    def sample_first_dim(
        self,
        value: Any,
        indices: Optional[Any] = None,
        num_samples: Optional[int] = None,
    ):
        if torch.is_tensor(value):
            if indices is None:
                if value.shape[0] == 0:
                    raise ValueError(f"{self.__class__.__name__} cannot sample from an empty tensor.")
                indices = torch.randperm(value.shape[0], device=value.device)[: min(num_samples, value.shape[0])]
            return value[indices], indices
        if isinstance(value, list):
            if indices is None:
                if len(value) == 0:
                    raise ValueError(f"{self.__class__.__name__} cannot sample from an empty list.")
                indices = torch.randperm(len(value))[: min(num_samples, len(value))]
            if torch.is_tensor(indices):
                indices = indices.tolist()
            return [value[i] for i in indices], indices
        raise TypeError(
            f"{self.__class__.__name__} does not know how to sample values of type "
            f"{type(value).__name__}; provide `sample_fn`."
        )

    def sample(
        self,
        data: Dict,
        key: str,
        peer: Optional[Dict] = None,
        indices: Optional[Any] = None,
    ):
        sampled_key = self.sampled_key(key)
        indices = self.resolve_sample_indices(data, key, peer=peer, indices=indices)
        if sampled_key in data:
            return data[sampled_key], indices
        if key not in data:
            raise KeyError(
                f"{self.__class__.__name__} cannot sample '{key}' because the key is missing. "
                f"Available keys: {sorted(data.keys())}"
            )
        if indices is None and self.num_samples is None:
            return data[key], None
        if self.sample_fn is not None:
            sampled_value, used_indices = self.sample_fn(
                data,
                key,
                indices=indices,
                num_samples=self.num_samples,
            )
        else:
            sampled_value, used_indices = self.sample_first_dim(
                data[key],
                indices=indices,
                num_samples=self.num_samples,
            )
        data[sampled_key] = sampled_value
        if used_indices is not None:
            data[self.sample_indices_key(key)] = used_indices
        return sampled_value, used_indices

    def resolve_inputs(
        self,
        pred: Dict,
        target: Optional[Dict],
        prediction_property: str,
        target_property: str,
    ):
        if target is None or not self.enabled():
            return pred, target
        pred_value, sample_indices = self.sample(pred, prediction_property, peer=target)
        target_value, _ = self.sample(target, target_property, peer=pred, indices=sample_indices)
        pred[prediction_property] = pred_value
        target[target_property] = target_value
        return pred, target
