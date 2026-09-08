"""Evaluation of geometry-grouped variational observables in physical units."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch.utils.data import DataLoader, Dataset

from curator.data import properties
from curator.data.utils import iter_batches
from curator.train.variational_losses import VariationalJointLoss


class VariationalEvaluator:
    """Evaluate the same masked, condition-aware objective used for training.

    Each dataset item must contain all conditions for one geometry.  Metrics
    aggregate loss numerators and denominators over the complete dataset, rather
    than averaging batch losses.  ``rmse`` uses the loss's geometry/condition
    weighting and is expressed in the supervised label's physical unit.

    ``results.pt`` preserves each CPU batch and its variational prediction,
    including ragged pointers, masks, units and boundary metadata.  Those records
    can be passed back to ``loss_fn`` to reproduce the reported metrics.
    """

    def __init__(
        self,
        model,
        loss_fn: VariationalJointLoss,
        output_dir="evaluate",
        batch_size: int = 8,
        device=None,
        num_workers: int = 0,
        pin_memory=None,
        save_data: bool = True,
    ) -> None:
        if not isinstance(loss_fn, VariationalJointLoss):
            raise TypeError("loss_fn must be a VariationalJointLoss")
        self.model = model
        self.loss_fn = loss_fn
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.save_data = save_data

    def evaluate(self, dataset):
        if not isinstance(dataset, (Dataset, DataLoader)):
            raise TypeError("VariationalEvaluator requires a Dataset or DataLoader")
        source = dataset.dataset if isinstance(dataset, DataLoader) else dataset
        for module in self.model.modules():
            validate_metadata = getattr(module, "validate_dataset_metadata", None)
            if callable(validate_metadata):
                validate_metadata(getattr(source, "metadata", None))
        reference = next(self.model.parameters(), None)
        if reference is None:
            reference = next(self.model.buffers(), torch.empty((), dtype=torch.get_default_dtype()))
        device = torch.device(self.device) if self.device is not None else reference.device
        dtype = reference.dtype if reference.is_floating_point() else torch.get_default_dtype()
        self.model.to(device)
        self.loss_fn.to(device)
        was_training = self.model.training
        self.model.eval()
        terms = {
            name: {
                "coefficient": coefficient,
                "numerator": 0.0,
                "denominator": 0.0,
                "n_geometries": 0,
                "n_conditions": 0,
            }
            for name, coefficient in self.loss_fn.term_coefficients.items()
        }
        num_structures = num_conditions = 0
        records = []
        try:
            # The caller may have entered no_grad/inference_mode.  Force and
            # response outputs still need coordinate/field derivative graphs.
            with torch.inference_mode(False), torch.enable_grad():
                for batch in iter_batches(
                    dataset,
                    batch_size=self.batch_size,
                    device=device,
                    dtype=dtype,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                ):
                    for key in (properties.positions, properties.edge_diff):
                        if key in batch:
                            batch[key].requires_grad_()
                    prediction = self.model(batch)
                    result = self.loss_fn(prediction, batch)
                    target = batch["variational"]
                    num_structures += target["condition_ptr"].numel() - 1
                    num_conditions += target["condition_weight"].numel()
                    for name, term in result["terms"].items():
                        aggregate = terms[name]
                        for key in ("numerator", "denominator"):
                            aggregate[key] += float(term[key].detach().cpu())
                        for key in ("n_geometries", "n_conditions"):
                            aggregate[key] += term[key]
                        label = target.get("labels", {}).get(name)
                        if label is not None and term["coefficient"] > 0:
                            if "unit" in aggregate and aggregate["unit"] != label["unit"]:
                                raise ValueError(f"label {name} changes units between evaluation batches")
                            aggregate["unit"] = label["unit"]
                    if self.save_data:
                        records.append(apply_to_collection(
                            {"batch": batch, "prediction": {
                                "variational_prediction": prediction["variational_prediction"],
                            }},
                            torch.Tensor,
                            lambda value: value.detach().cpu().clone(),
                        ))
        finally:
            self.model.train(was_training)

        total_loss = 0.0
        for name, term in terms.items():
            active = term["denominator"] > 0
            term["status"] = "active" if active else "disabled" if term["coefficient"] == 0 else "inactive"
            term["coverage"] = term["n_conditions"] / max(1, num_conditions)
            term["normalized_mse"] = term["numerator"] / term["denominator"] if active else None
            term["loss"] = term["coefficient"] * term["normalized_mse"] if active else 0.0
            if name in self.loss_fn.supervised_scales:
                term["scale"] = self.loss_fn.supervised_scales[name]
                term["rmse"] = term["scale"] * math.sqrt(term["normalized_mse"]) if active else None
            total_loss += term["loss"]
        self.metrics = {
            "num_structures": num_structures,
            "num_conditions": num_conditions,
            "balance_mode": self.loss_fn.balance_mode,
            "loss": total_loss,
            "terms": terms,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.save_data:
            torch.save({"format_version": 1, "batches": records}, self.output_dir / "results.pt")
        (self.output_dir / "metrics.json").write_text(
            json.dumps(self.metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        return self.metrics
