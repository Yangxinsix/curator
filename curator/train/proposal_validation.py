"""Bounded validation of amortized states against the same learned functional."""

import math
from pathlib import Path

from pytorch_lightning import Callback
from torch.utils.data import Subset

from curator.evaluate.variational import VariationalEvaluator


class ProposalEquilibriumValidation(Callback):
    """Audit a fixed validation prefix and optionally save a qualified state model.

    ``supervised_limits`` contains physical RMSE limits, normally frozen from the
    matched pointwise pilot (for example 1.05 times its validation RMSE). Without
    limits this callback records diagnostics but does not qualify a checkpoint.
    It is intended for single-process pilots, where reference solves are small.
    """

    def __init__(self, every_n_epochs=5, num_geometries=4, supervised_limits=None,
                 compute_responses=True, output_dir=None):
        if every_n_epochs < 1 or num_geometries < 1:
            raise ValueError("validation interval and subset size must be positive")
        self.every_n_epochs = int(every_n_epochs)
        self.num_geometries = int(num_geometries)
        self.supervised_limits = dict(supervised_limits or {})
        if any(not math.isfinite(value) or value <= 0 for value in self.supervised_limits.values()):
            raise ValueError("supervised RMSE limits must be positive and finite")
        self.compute_responses = bool(compute_responses)
        self.output_dir = output_dir
        self.best_state_rmse = math.inf
        self._subset_identity = None
        self._save_pending = False

    def state_dict(self):
        return {"best_state_rmse": self.best_state_rmse,
                "supervised_limits": self.supervised_limits, "num_geometries": self.num_geometries,
                "every_n_epochs": self.every_n_epochs, "compute_responses": self.compute_responses,
                "subset_identity": self._subset_identity}

    def load_state_dict(self, state):
        if (state["supervised_limits"] != self.supervised_limits
                or state["num_geometries"] != self.num_geometries
                or state["every_n_epochs"] != self.every_n_epochs
                or state["compute_responses"] != self.compute_responses):
            raise ValueError("proposal validation gate, schedule or fixed subset differs from checkpoint")
        self.best_state_rmse = state["best_state_rmse"]
        self._subset_identity = state["subset_identity"]
        self._save_pending = False

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or (trainer.current_epoch + 1) % self.every_n_epochs:
            return
        if trainer.world_size != 1:
            raise ValueError("ProposalEquilibriumValidation currently requires a single-process pilot")
        loader = trainer.val_dataloaders
        if isinstance(loader, (list, tuple)):
            if len(loader) != 1:
                raise ValueError("proposal validation requires one validation dataloader")
            loader = loader[0]
        source = loader.dataset
        if not len(source):
            raise ValueError("proposal validation requires a nonempty validation dataset")
        subset = Subset(source, range(min(len(source), self.num_geometries)))
        identity = []
        for sample in subset:
            payload = getattr(sample, "variational", None)
            if payload is None:
                raise ValueError("proposal validation requires named variational geometries")
            identity.append((payload.geometry_uid, payload.geometry_hash))
        identity = tuple(identity)
        if self._subset_identity is not None and self._subset_identity != identity:
            raise ValueError("proposal validation fixed geometry identities differ from checkpoint")
        self._subset_identity = identity
        if len(pl_module.outputs) != 1:
            raise ValueError("proposal validation requires one joint variational loss output")
        directory = Path(self.output_dir or (Path(trainer.default_root_dir) / "proposal_validation"))
        evaluator = VariationalEvaluator(
            pl_module.model, pl_module.outputs[0].loss_fn,
            output_dir=directory / f"epoch_{trainer.current_epoch:04d}", batch_size=1,
            device=pl_module.device, save_data=True, state_mode="predicted",
            compare_converged=True, compute_responses=self.compute_responses,
            record_reference_failures=True,
        )
        metrics = evaluator.evaluate(subset)
        state = metrics["comparison"]["difference_from_predicted"].get("variable_value", {})
        rmse = state.get("scalar_rmse", math.inf)
        pl_module.log("val_proposal_state_rmse", rmse, sync_dist=False)
        stationary = metrics.get("physical_stationarity", {}).get("rms", math.inf)
        pl_module.log("val_proposal_stationarity_rms", stationary, sync_dist=False)
        reference = metrics["reference_solver"]
        qualified = (bool(self.supervised_limits) and not reference["n_failed_conditions"]
                     and reference["n_converged"] == reference["n_conditions"])
        qualified = qualified and not metrics.get("constraints", {}).get("n_failed_conditions", 1)
        for name, limit in self.supervised_limits.items():
            value = metrics["terms"].get(name, {}).get("rmse")
            qualified = qualified and value is not None and math.isfinite(value) and value <= limit
        if qualified and math.isfinite(rmse) and rmse < self.best_state_rmse:
            self.best_state_rmse = rmse
            self._save_pending = True

    def on_train_epoch_end(self, trainer, pl_module):
        if self._save_pending:
            self._save_pending = False
            path = Path(trainer.default_root_dir) / "model_path" / "best_proposal_equilibrium.ckpt"
            path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(path)


__all__ = ["ProposalEquilibriumValidation"]
