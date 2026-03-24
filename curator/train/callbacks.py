import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch_ema import ExponentialMovingAverage as EMA
import logging

logger = logging.getLogger(__name__)

class ExponentialMovingAverage(Callback):
    def __init__(self, decay: float, use_num_updates: bool=True, *args, **kwargs):
        self.decay = decay
        self.ema = None
        self.use_num_updates = use_num_updates
        self._to_load = None

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.ema is None:
            self.ema = EMA(pl_module.model.parameters(), decay=self.decay, use_num_updates=self.use_num_updates)
        if self._to_load is not None:
            self.ema.load_state_dict(self._to_load)
            self._to_load = None

        # load average parameters, to have same starting point as after validation
        self.ema.store()
        self.ema.copy_to()

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.ema.restore()

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        self.ema.update()

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module, *args, **kwargs
    ):
        self.ema.store()
        self.ema.copy_to()

    def load_state_dict(self, state_dict):
        if "ema" in state_dict:
            if self.ema is None:
                self._to_load = state_dict["ema"]
            else:
                self.ema.load_state_dict(state_dict["ema"])

    def state_dict(self):
        return {"ema": self.ema.state_dict() if self.ema is not None else None}


class FreezeSchedule(Callback):
    def __init__(self, stages=None):
        self.stages = sorted(
            [dict(stage) for stage in (stages or [])],
            key=lambda stage: int(stage.get("start_epoch", 0)),
        )
        self._applied_stage_indices = set()

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self._apply_due_stages(trainer, pl_module, include_past=True)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._apply_due_stages(trainer, pl_module, include_past=False)

    def _apply_due_stages(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        *,
        include_past: bool,
    ) -> None:
        current_epoch = int(getattr(trainer, "current_epoch", 0))
        for index, stage in enumerate(self.stages):
            if index in self._applied_stage_indices:
                continue
            start_epoch = int(stage.get("start_epoch", 0))
            if start_epoch > current_epoch:
                continue
            if not include_past and start_epoch != current_epoch:
                continue
            self._apply_stage(trainer, pl_module, stage)
            self._applied_stage_indices.add(index)

    def _apply_stage(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage) -> None:
        module_groups = getattr(pl_module.model, "module_groups")()
        freeze_summary = self._set_trainable(module_groups, stage.get("freeze"), trainable=False)
        unfreeze_summary = self._set_trainable(module_groups, stage.get("unfreeze"), trainable=True)
        lr_summary = self._set_group_lrs(trainer, stage.get("lr"))
        self._log_stage(stage, freeze_summary, unfreeze_summary, lr_summary)

    def _set_trainable(self, module_groups, names, *, trainable: bool):
        requested = self._normalize_group_names(names)
        if not requested:
            return []
        selected = self._select_groups(module_groups, requested, kind="module")
        summaries = []
        for group_name in requested:
            modules = selected.get(group_name)
            if not modules:
                continue
            summaries.append(self._set_group_trainable(group_name, modules, trainable=trainable))
        return summaries

    def _set_group_trainable(self, group_name, modules, *, trainable: bool):
        modules = list(modules)
        seen = set()
        total_tensors = 0
        changed_tensors = 0
        total_elements = 0
        changed_elements = 0
        for module in modules:
            for parameter in module.parameters():
                param_id = id(parameter)
                if param_id in seen:
                    continue
                seen.add(param_id)
                total_tensors += 1
                total_elements += parameter.numel()
                if bool(parameter.requires_grad) != bool(trainable):
                    changed_tensors += 1
                    changed_elements += parameter.numel()
                parameter.requires_grad_(trainable)
        return {
            "group": str(group_name),
            "trainable": bool(trainable),
            "modules": len(modules),
            "params": total_tensors,
            "updated_params": changed_tensors,
            "scalars": total_elements,
            "updated_scalars": changed_elements,
        }

    def _set_group_lrs(self, trainer: "pl.Trainer", lr_overrides):
        if not lr_overrides:
            return []
        optimizer_groups = {}
        for optimizer in getattr(trainer, "optimizers", []):
            for group in optimizer.param_groups:
                name = group.get("name")
                if name is None:
                    continue
                optimizer_groups.setdefault(str(name), []).append(group)
        selected = self._select_groups(optimizer_groups, lr_overrides.keys(), kind="optimizer")
        summaries = []
        for group_name, lr in dict(lr_overrides).items():
            groups = selected.get(str(group_name))
            if not groups:
                continue
            old_lrs = sorted({float(group.get("lr", 0.0)) for group in groups})
            new_lr = float(lr)
            for group in groups:
                group["lr"] = new_lr
            summaries.append(
                {
                    "group": str(group_name),
                    "old_lrs": old_lrs,
                    "new_lr": new_lr,
                    "optimizer_groups": len(groups),
                }
            )
        return summaries

    def _normalize_group_names(self, names):
        if names is None:
            return []
        if isinstance(names, str):
            return [names]
        return [str(name) for name in names]

    def _select_groups(self, available_groups, requested_names, *, kind: str):
        available = {str(name): value for name, value in dict(available_groups).items()}
        requested = [str(name) for name in requested_names]
        missing = sorted(name for name in requested if name not in available)
        if missing:
            raise KeyError(
                f"Unknown {kind} group names: {missing}. "
                f"Available groups: {sorted(available)}"
            )
        return {name: available[name] for name in requested if name in available}

    def _log_stage(self, stage, freeze_summary, unfreeze_summary, lr_summary) -> None:
        epoch = int(stage.get("start_epoch", 0))
        logger.info("")
        logger.info("Applying freeze schedule stage at epoch %s", epoch)

        trainability_rows = [("freeze", summary) for summary in freeze_summary]
        trainability_rows.extend(("unfreeze", summary) for summary in unfreeze_summary)
        if trainability_rows:
            logger.info("  Trainability updates:")
            logger.info(
                "    %-10s %-20s %8s %8s %16s %10s %16s",
                "action",
                "group",
                "modules",
                "params",
                "updated_params",
                "scalars",
                "updated_scalars",
            )
            for action, summary in trainability_rows:
                logger.info(
                    "    %-10s %-20s %8d %8d %16d %10d %16d",
                    action,
                    summary["group"],
                    summary["modules"],
                    summary["params"],
                    summary["updated_params"],
                    summary["scalars"],
                    summary["updated_scalars"],
                )

        if lr_summary:
            logger.info("  Learning-rate updates:")
            logger.info(
                "    %-20s %14s %14s %12s",
                "group",
                "old_lr",
                "new_lr",
                "opt_groups",
            )
            for summary in lr_summary:
                old = ", ".join(f"{lr:.6g}" for lr in summary["old_lrs"])
                logger.info(
                    "    %-20s %14s %14.6g %12d",
                    summary["group"],
                    old,
                    summary["new_lr"],
                    summary["optimizer_groups"],
                )

        if not trainability_rows and not lr_summary:
            logger.info("  no freeze, unfreeze, or lr updates in this stage")
