"""Initialize an experiment branch's optimizer without resetting resumed runs."""

import copy
import hashlib
import math
from pathlib import Path

import torch
from pytorch_lightning import Callback


def optimizer_parameter_names(module, optimizer):
    """Stable ordered names for detecting unsafe optimizer-state remapping."""
    names = {id(p): name for name, p in module.named_parameters()}
    return [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups]


def restore_optimizer_state(optimizer, state, lr_multipliers=None):
    """Restore moments exactly; change only requested group learning rates."""
    multipliers = dict(lr_multipliers or {})
    current = optimizer.param_groups
    saved = state["param_groups"]
    if len(current) != len(saved):
        raise ValueError("optimizer group count changed")
    for old, new in zip(saved, current):
        if old.get("name") != new.get("name") or len(old["params"]) != len(new["params"]):
            raise ValueError("optimizer group order or parameter count changed")
        for index, parameter in zip(old["params"], new["params"]):
            for name, value in state["state"].get(index, {}).items():
                if torch.is_tensor(value) and name != "step" and value.shape != parameter.shape:
                    raise ValueError("optimizer moment shape mismatch")
    if set(multipliers) - {group.get("name") for group in current}:
        raise ValueError("unknown group in lr_multipliers")
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in multipliers.values()):
        raise ValueError("LR multipliers must be finite and positive")
    optimizer.load_state_dict(copy.deepcopy(state))
    restored = optimizer.state_dict()
    for old_group, new_group in zip(saved, restored["param_groups"]):
        for old_id, new_id in zip(old_group["params"], new_group["params"]):
            old_state = state["state"].get(old_id, {})
            new_state = restored["state"].get(new_id, {})
            if old_state.keys() != new_state.keys():
                raise ValueError("optimizer state keys changed during restore")
            for key, before in old_state.items():
                after = new_state[key]
                equal = torch.equal(before.cpu(), after.cpu()) if torch.is_tensor(before) else before == after
                if not equal:
                    raise ValueError("optimizer state changed during restore: " + key)
    audit = {}
    for group in optimizer.param_groups:
        before = float(group["lr"])
        group["lr"] = before * float(multipliers.get(group["name"], 1.0))
        audit[group["name"]] = {"source_lr": before, "lr": group["lr"],
                                 "parameters": len(group["params"])}
    return audit


class OptimizerStateInitializer(Callback):
    """One-time branch initialization; owns no training or checkpoint policy.

    The input is a small torch file with ``optimizer_state`` and ordered
    ``parameter_names`` obtained from the source checkpoint, plus optional
    ``provenance``. Weights must already have been strictly loaded by the caller.
    Full Lightning resumption restores this callback's state and never reapplies
    the initialization or LR multiplier.
    """

    def __init__(self, state_path, lr_multipliers=None):
        self.state_path = str(state_path)
        self.lr_multipliers = dict(lr_multipliers or {})
        self.audit = None

    def on_fit_start(self, trainer, pl_module):
        if self.audit is not None:
            return
        if trainer.ckpt_path:
            raise ValueError("a full resume must contain the initializer callback state")
        if len(trainer.optimizers) != 1:
            raise ValueError("exactly one optimizer is supported")
        optimizer = trainer.optimizers[0]
        if not isinstance(optimizer, torch.optim.Adam):
            raise TypeError("this branch initializer supports Adam/AMSGrad only")
        path = Path(self.state_path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["parameter_names"] != optimizer_parameter_names(pl_module, optimizer):
            raise ValueError("ordered optimizer parameter names differ from the source checkpoint")
        audit = restore_optimizer_state(optimizer, payload["optimizer_state"], self.lr_multipliers)
        self.audit = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "groups": audit, "provenance": payload.get("provenance", {})}

    def state_dict(self):
        return {"audit": self.audit}

    def load_state_dict(self, state_dict):
        self.audit = state_dict["audit"]
