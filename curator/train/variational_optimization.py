"""Controlled block updates for joint variational energy/force training.

The model and its force graph are unchanged.  Only the parameters receiving a
backward gradient and the number of proposal updates differ between protocols.
"""

import copy
import math

import torch

from curator.data import properties
from curator.model.lit_module import LitNNP
from curator.train.variational_losses import VariationalModelOutput


class VariationalOptimizationTask(LitNNP):
    """Single-device, float32 ablations with one persistent optimizer.

    ``joint`` gives the original accumulated joint update. ``alternating_joint``
    adds proposal-only residual steps before it. ``alternating_ef`` instead ends
    with an energy/force update of every non-proposal group. Frozen parameters
    retain their input derivatives: this class never detaches the forward graph.

    An incomplete final window divides by ``accumulation_steps``, matching
    Lightning's automatic accumulation rather than increasing its step size.
    Checkpoints are supported at completed outer-update boundaries only.
    """

    def __init__(self, *args, update_rule="joint", inner_steps=0,
                 accumulation_steps=3, gradient_clip_norm=10.0,
                 proposal_group="proposal", **kwargs):
        super().__init__(*args, **kwargs)
        if update_rule not in {"joint", "alternating_joint", "alternating_ef"}:
            raise ValueError("unknown variational update_rule")
        if int(accumulation_steps) != accumulation_steps or accumulation_steps < 1:
            raise ValueError("accumulation_steps must be a positive integer")
        if int(inner_steps) != inner_steps or inner_steps < 0:
            raise ValueError("inner_steps must be a nonnegative integer")
        if (update_rule == "joint") != (inner_steps == 0):
            raise ValueError("joint requires zero inner_steps; alternating requires positive inner_steps")
        if not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be finite and positive")
        if len(self.outputs) != 1 or not isinstance(self.outputs[0], VariationalModelOutput):
            raise TypeError("exactly one VariationalModelOutput is supported")
        expected = {"condition_energy", "condition_forces", "residual:stationarity"}
        if set(self.outputs[0].loss_fn.term_coefficients) != expected:
            raise ValueError("only energy, force and stationarity terms are supported")
        if self.warmup_steps:
            raise ValueError("warmup is unsupported by this block-update protocol")
        if self.scheduler is not None and not self.scheduler_monitor:
            self.scheduler_monitor = "val_joint_predicted_loss"
        if self.normalize_domain_loss:
            raise ValueError("domain loss normalization is unsupported")
        self.automatic_optimization = False
        self.update_rule = update_rule
        self.inner_steps = int(inner_steps)
        self.accumulation_steps = int(accumulation_steps)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.proposal_group = str(proposal_group)
        self._pending_batches = []
        self._outer_objective = None
        self._validated_epoch = -1
        self._last_scheduler_epoch = -1
        self.update_counts = dict(outer_steps=0, inner_steps=0,
                                  ef_structures=0, residual_only_structures=0,
                                  forward_minibatches=0)
        self.save_hyperparameters("update_rule", "inner_steps", "accumulation_steps",
                                  "gradient_clip_norm", "proposal_group")

    def configure_optimizers(self):
        configured = super().configure_optimizers()
        if self.scheduler is not None:
            scheduler = configured["lr_scheduler"]["scheduler"]
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                raise TypeError("only native ReduceLROnPlateau is supported")
        return configured

    def on_train_start(self):
        trainer = self.trainer
        if trainer.world_size != 1 or str(trainer.precision) not in {"32", "32-true"}:
            raise ValueError("VariationalOptimizationTask supports single-device float32 only")
        if trainer.accumulate_grad_batches != 1 or trainer.gradient_clip_val is not None:
            raise ValueError("set trainer.accumulate_grad_batches=1 and gradient_clip_val=null")
        if not math.isfinite(trainer.num_training_batches):
            raise ValueError("finite training dataloaders are required")
        interval = trainer.val_check_interval
        epoch_validation = isinstance(interval, float) and interval == 1.0
        if not epoch_validation and (not isinstance(interval, int) or interval % self.accumulation_steps):
            raise ValueError("validation must coincide with an outer-update boundary")
        optimizer = self.optimizers(use_pl_optimizer=False)
        if self.scheduler is not None:
            if not epoch_validation:
                raise ValueError("plateau scheduling currently requires validation at epoch end")
            scheduler = self.lr_schedulers()
            lrs = [float(group["lr"]) for group in optimizer.param_groups]
            if any(lr <= 0 for lr in lrs):
                raise ValueError("scheduled groups must have positive learning rates")
            fractions = [floor / lr for floor, lr in zip(scheduler.min_lrs, lrs)]
            if any(fraction < 0 or fraction > 1 for fraction in fractions) or not all(
                    math.isclose(fraction, fractions[0], rel_tol=1e-7, abs_tol=1e-12)
                    for fraction in fractions):
                raise ValueError("scheduler min_lr values must preserve optimizer group LR ratios")
        names = [group.get("name") for group in optimizer.param_groups]
        if names.count(self.proposal_group) != 1:
            raise ValueError("optimizer must have exactly one named proposal group")
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        if len({id(p) for p in parameters}) != len(parameters):
            raise ValueError("optimizer groups must not overlap")
        if {id(p) for p in parameters if p.requires_grad} != {
                id(p) for p in self.parameters() if p.requires_grad}:
            raise ValueError("optimizer groups must cover every trainable parameter")
        if self.inner_steps:
            for module in self.model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.track_running_stats:
                    raise ValueError("running BatchNorm state confounds repeated inner forwards")
                if isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p:
                    raise ValueError("stochastic dropout is unsupported by this paired protocol")
        super().on_train_start()

    def loss_fn(self, pred, batch, subset):
        result = super().loss_fn(pred, batch, subset)
        if subset == "train" and self.update_rule == "alternating_ef":
            output = self.outputs[0]
            terms = output.loss_fn(pred, batch)["terms"]
            self._outer_objective = output.loss_weight * (
                terms["condition_energy"]["loss"] + terms["condition_forces"]["loss"])
        return result

    def _active_parameters(self, optimizer, phase):
        return [p for group in optimizer.param_groups for p in group["params"]
                if p.requires_grad and (phase == "joint" or
                (group["name"] == self.proposal_group) == (phase == "residual"))]

    def _update(self, optimizer, batches, phase, batch_idx):
        active = self._active_parameters(optimizer, phase)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for offset, pristine in enumerate(batches):
            batch = copy.deepcopy(pristine)
            if phase == "residual":
                prediction = self.model(batch)
                output = self.outputs[0]
                loss = output.loss_weight * output.loss_fn(prediction, batch)["terms"][
                    "residual:stationarity"]["loss"]
            else:
                loss = super().training_step(batch, batch_idx - len(batches) + 1 + offset)
                if phase == "ef":
                    loss = self._outer_objective
                    self._outer_objective = None
            # Supplying inputs chooses parameter gradients without cutting any
            # coordinate or embedding derivative in the energy/force graph.
            self.manual_backward(loss / self.accumulation_steps, inputs=active)
            losses.append(loss.detach())
            self.update_counts["forward_minibatches"] += 1
        norm = torch.nn.utils.clip_grad_norm_(active, self.gradient_clip_norm,
                                              error_if_nonfinite=True)
        self.log("optimization/" + phase + "_preclip_norm", norm,
                 on_step=True, on_epoch=True, batch_size=len(batches))
        self.log("optimization/" + phase + "_clip_factor",
                 torch.clamp(self.gradient_clip_norm / (norm + 1e-6), max=1),
                 on_step=True, on_epoch=True, batch_size=len(batches))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return torch.stack(losses).mean()

    def training_step(self, batch, batch_idx):
        if self._is_combined_batch(batch):
            raise ValueError("combined/domain dataloaders are unsupported")
        self._pending_batches.append(copy.deepcopy(batch))
        last = batch_idx + 1 == self.trainer.num_training_batches
        if len(self._pending_batches) < self.accumulation_steps and not last:
            return None
        batches, self._pending_batches = self._pending_batches, []
        optimizer = self.optimizers()
        n_structures = sum(int(b[properties.n_atoms].numel()) for b in batches)
        for _ in range(self.inner_steps):
            self._update(optimizer, batches, "residual", batch_idx)
            self.update_counts["inner_steps"] += 1
            self.update_counts["residual_only_structures"] += n_structures
        phase = "ef" if self.update_rule == "alternating_ef" else "joint"
        loss = self._update(optimizer, batches, phase, batch_idx)
        self.update_counts["outer_steps"] += 1
        self.update_counts["ef_structures"] += n_structures
        self.log_dict({"optimization/" + key: float(value)
                       for key, value in self.update_counts.items()},
                      on_step=True, on_epoch=False, batch_size=n_structures)
        return loss

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if not self.trainer.sanity_checking:
            self._validated_epoch = int(self.current_epoch)

    def on_train_epoch_end(self):
        if self._pending_batches:
            raise RuntimeError("unflushed accumulation window at epoch end")
        super().on_train_epoch_end()
        # JointValidationLoss callback has already logged this epoch's complete
        # validation loss. Step after epoch-end callbacks, as automatic
        # Lightning does; ResumeCheckpoint saves this state at the next epoch
        # start or fit end. Early stopping still observes the pre-step LR.
        if (self.scheduler is not None and self._validated_epoch == self.current_epoch
                and self._last_scheduler_epoch != self.current_epoch):
            metrics = self.trainer.callback_metrics
            if self.scheduler_monitor not in metrics:
                raise RuntimeError("missing scheduler monitor: " + self.scheduler_monitor)
            value = metrics[self.scheduler_monitor]
            if not torch.isfinite(torch.as_tensor(value)).all():
                raise ValueError("scheduler monitor must be finite")
            scheduler = self.lr_schedulers()
            before = [float(group["lr"]) for group in scheduler.optimizer.param_groups]
            scheduler.step(value)
            after = [float(group["lr"]) for group in scheduler.optimizer.param_groups]
            factors = [new / old for old, new in zip(before, after)]
            if not all(math.isclose(factor, factors[0], rel_tol=1e-7, abs_tol=1e-12)
                       for factor in factors):
                raise RuntimeError("scheduler eps changed group LR ratios; use eps=0")
            self._last_scheduler_epoch = int(self.current_epoch)

    def on_save_checkpoint(self, checkpoint):
        if self._pending_batches:
            raise RuntimeError("save checkpoints only at completed outer-update boundaries")
        super().on_save_checkpoint(checkpoint)
        checkpoint["variational_optimization"] = {
            "update_counts": dict(self.update_counts), "update_rule": self.update_rule,
            "inner_steps": self.inner_steps, "accumulation_steps": self.accumulation_steps,
            "last_scheduler_epoch": self._last_scheduler_epoch}

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint.get("variational_optimization")
        if state is None:
            raise ValueError("use strict weight + optimizer initialization to start a new branch")
        for key in ("update_rule", "inner_steps", "accumulation_steps"):
            if state[key] != getattr(self, key):
                raise ValueError("cannot resume with changed " + key)
        self.update_counts = dict(state["update_counts"])
        self._last_scheduler_epoch = int(state.get("last_scheduler_epoch", -1))


class VariationalGradientTask(LitNNP):
    """Native automatic optimization with proposal-only residual gradients.

    Energy/force terms update every connected parameter. Residual terms reuse
    the live energy graph but update proposal parameters only. Lightning still
    owns optimizer steps, accumulation, clipping, schedules and checkpoints.
    Single-device true float32/float64 are supported; AMP and DDP are rejected.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.outputs) != 1 or not isinstance(self.outputs[0], VariationalModelOutput):
            raise TypeError("VariationalGradientTask requires exactly one VariationalModelOutput")
        self._gradient_objectives = None

    def _proposal_parameters(self):
        heads = [module for module in self.model.modules()
                 if getattr(module, "joint_variational_output", False)
                 and getattr(module, "residual_evaluation", "replay") == "reuse"
                 and module.residual_gradient_policy == "proposal_only"]
        if len(heads) != 1:
            raise ValueError("VariationalGradientTask needs one reused proposal_only output head")
        parameters = tuple(parameter for parameter in heads[0].proposal.parameters()
                           if parameter.requires_grad)
        head_path = next(name for name, module in self.model.named_modules() if module is heads[0])
        prefix = (head_path + "." if head_path else "") + "proposal."
        proposal_ids = {id(parameter) for parameter in parameters}
        if any(id(parameter) in proposal_ids and not name.startswith(prefix)
               for name, parameter in self.model.named_parameters(remove_duplicate=False)):
            raise ValueError("proposal-only routing requires separate proposal and physical parameters")
        return parameters

    def on_train_start(self):
        if self.trainer.world_size != 1 or str(self.trainer.precision) not in {
                "32", "32-true", "64", "64-true"}:
            raise ValueError("VariationalGradientTask requires one device and true float32/float64; "
                             "AMP and distributed training are not supported")
        self._proposal_parameters()
        super().on_train_start()

    def training_step(self, batch, batch_idx):
        if self._is_combined_batch(batch):
            raise ValueError("VariationalGradientTask does not support combined dataloaders")
        self._gradient_objectives = None
        return super().training_step(batch, batch_idx)

    def loss_fn(self, pred, batch, subset):
        if subset != "train":
            return super().loss_fn(pred, batch, subset)
        output = self.outputs[0]
        result = output.loss_fn(pred, batch, allow_parameter_routing=True)
        total, count = output._record_result(result, return_num_obs=True, subset=subset)
        self._gradient_objectives = (
            total, output.loss_weight * result["unrestricted_loss"],
            output.loss_weight * result["parameter_routed_loss"],
        )
        # StructuredModelOutput already handles physical/domain/batch weights.
        key = subset + "_" + output.name + "_loss"
        return {subset + "_total_loss": total, key: total}, {
            subset + "_total_loss": 1, key: count}

    def backward(self, loss, *args, **kwargs):
        if self._gradient_objectives is None:
            raise RuntimeError("routed backward requires the preceding training loss")
        total, unrestricted, residual = self._gradient_objectives
        self._gradient_objectives = None
        if not total.requires_grad:
            if loss.requires_grad:
                raise RuntimeError("routed loss must originate from the recorded training objective")
            return  # All terms disabled/absent: leave every gradient, including None, unchanged.
        # Lightning divides the returned loss during accumulation. Obtain its
        # actual graph scale instead of guessing from trainer settings, including
        # the incomplete final accumulation window. This is a scalar VJP only.
        scale = torch.autograd.grad(loss, total, retain_graph=True)[0].detach()
        parameters = self._proposal_parameters()
        gradients = (torch.autograd.grad(residual, parameters, grad_outputs=scale,
                                        retain_graph=True, allow_unused=True)
                     if parameters and residual.requires_grad else ())
        if unrestricted.requires_grad:
            super().backward(unrestricted * scale, *args, **kwargs)
        # Preserve None for disconnected parameters: AdamW may update a parameter
        # with a zero gradient through weight decay, but must skip an unused one.
        for parameter, gradient in zip(parameters, gradients):
            if gradient is not None:
                if parameter.grad is None:
                    parameter.grad = gradient.detach()
                else:
                    parameter.grad.add_(gradient.detach())
