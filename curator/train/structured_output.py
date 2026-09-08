"""Trainer protocol for losses with independently normalized component terms."""

import math

import torch
from torch import nn


class StructuredModelOutput(nn.Module):
    """Aggregate loss numerators and denominators separately for each stage.

    ``subset=None`` preserves the plain trainer's historical loss accumulator.
    Its subsequent ``calculate_metrics(..., subset)`` assigns that batch to the
    corresponding metric stage. Lightning can pass ``subset`` directly to
    ``calculate_loss`` so even loss-only calls have an unambiguous stage.

    These losses already apply geometry weights and physical scales internally.
    Their components cannot share a scalar domain rescaling factor.
    """

    supports_subsets = True
    handles_batch_weight = True
    handles_domain_loss = True
    target_property = None
    is_penalty = False

    _subsets = (None, "train", "val", "test")

    def __init__(self, loss_fn, name, term_names, loss_weight=1.0):
        super().__init__()
        loss_weight = float(loss_weight)
        if not math.isfinite(loss_weight) or loss_weight < 0:
            raise ValueError("loss_weight must be finite and nonnegative")
        self.loss_fn = loss_fn
        self.name = str(name)
        self.term_names = tuple(term_names)
        self.loss_weight = loss_weight
        shape = (len(self._subsets), len(self.term_names))
        self.register_buffer("_epoch_numerator", torch.zeros(shape, dtype=torch.float64), persistent=False)
        self.register_buffer("_epoch_denominator", torch.zeros(shape, dtype=torch.float64), persistent=False)
        self._last_result = {}

    @classmethod
    def _subset_index(cls, subset):
        if subset not in cls._subsets:
            raise ValueError("subset must be 'train', 'val', 'test', or None")
        return cls._subsets.index(subset)

    def _add_terms(self, subset, terms):
        index = self._subset_index(subset)
        with torch.no_grad():
            self._epoch_numerator[index].add_(terms[0].to(self._epoch_numerator))
            self._epoch_denominator[index].add_(terms[1].to(self._epoch_denominator))

    def _record_result(self, result, return_num_obs, subset):
        self._subset_index(subset)
        terms = [result["terms"][name] for name in self.term_names]
        # Include the coefficient at accumulation time, since callbacks may
        # change loss_weight before the epoch aggregate is requested.
        if terms:
            snapshot = (
                torch.stack([
                    term["numerator"].detach() * term["coefficient"] * self.loss_weight
                    for term in terms
                ]),
                torch.stack([term["denominator"].detach() for term in terms]),
                torch.stack([term["loss"].detach() * self.loss_weight for term in terms]),
            )
        else:
            snapshot = tuple(self._epoch_numerator.new_zeros(0) for _ in range(3))
        self._add_terms(subset, snapshot)
        self._last_result[subset] = snapshot
        loss = result["loss"] * self.loss_weight
        if return_num_obs:
            return loss, max((term["n_geometries"] for term in terms), default=0)
        return loss

    def _global_sums(self, subset=None):
        index = self._subset_index(subset)
        numerator = self._epoch_numerator[index].clone()
        denominator = self._epoch_denominator[index].clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(numerator)
            torch.distributed.all_reduce(denominator)
        return numerator, denominator

    def accumulate_loss(self, subset=None):
        numerator, denominator = self._global_sums(subset)
        active = denominator > 0
        return float((numerator[active] / denominator[active]).sum().cpu())

    def calculate_metrics(self, prediction, batch, subset):
        del prediction, batch
        self._subset_index(subset)
        result = self._last_result.pop(subset, None)
        if result is None and subset is not None:
            result = self._last_result.pop(None, None)
            if result is not None:
                self._add_terms(subset, result)
        if result is None:
            return {}
        return {
            "%s_%s_%s_loss" % (subset, self.name, name.replace(":", "_")): result[2][index]
            for index, name in enumerate(self.term_names)
        }

    def accumulate_metrics(self, subset):
        numerator, denominator = self._global_sums(subset)
        return {
            name.replace(":", "_") + "_loss": float((numerator[index] / denominator[index]).cpu())
            for index, name in enumerate(self.term_names)
            if denominator[index] > 0
        }

    def reset_loss(self, subset=None):
        """Clear every accumulator, or just the explicitly requested stage."""
        if subset is None:
            self._epoch_numerator.zero_()
            self._epoch_denominator.zero_()
            self._last_result.clear()
        else:
            index = self._subset_index(subset)
            self._epoch_numerator[index].zero_()
            self._epoch_denominator[index].zero_()
            self._last_result.pop(subset, None)

    def reset_metrics(self, subset=None):
        self.reset_loss(subset)

    def __getstate__(self):
        # Curator also supports checkpoints containing complete output objects;
        # nonpersistent buffers alone only cover state_dict checkpoints.
        state = super().__getstate__()
        state["_buffers"] = dict(state["_buffers"])
        for name in ("_epoch_numerator", "_epoch_denominator"):
            state["_buffers"][name] = torch.zeros_like(state["_buffers"][name])
        state["_last_result"] = {}
        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Older ACVR adapters stored epoch totals as persistent buffers.
        for name in ("_epoch_numerator", "_epoch_denominator"):
            state_dict.pop(prefix + name, None)
        self.reset_loss()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
