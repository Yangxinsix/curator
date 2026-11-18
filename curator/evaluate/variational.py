"""Evaluation of geometry-grouped variational observables in physical units."""

from __future__ import annotations

import json
import math
import time
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

    ``state_mode`` explicitly chooses the proposal or a local equilibrium of
    the same learned functional. ``compare_converged`` retains predicted-state
    primary metrics and reports the second calculation separately. Neither a
    converged state nor a small residual certifies a global energy minimum.
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
        state_mode=None,
        compare_converged: bool = False,
        compute_responses=None,
        record_reference_failures: bool = False,
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
        if state_mode not in {None, "predicted", "converged"}:
            raise ValueError("state_mode must be predicted, converged, or None")
        if compare_converged and state_mode == "converged":
            raise ValueError("compare_converged requires predicted primary evaluation")
        self.state_mode = state_mode
        self.compare_converged = bool(compare_converged)
        if compute_responses is not None and not isinstance(compute_responses, bool):
            raise TypeError("compute_responses must be bool or None")
        self.compute_responses = compute_responses
        self.record_reference_failures = bool(record_reference_failures)

    def _predict(self, batch, mode):
        owners = [module for module in self.model.modules()
                  if hasattr(module, "supports_state_modes")]
        if (mode is not None or self.compute_responses is not None) and not owners:
            raise ValueError("model does not support explicit variational state modes")
        original = [module.state_mode for module in owners]
        original_responses = [getattr(module, "compute_responses", None) for module in owners]
        original_graphs = [getattr(module, "retain_response_graph", None) for module in owners]
        try:
            if mode is not None:
                for module in owners:
                    if mode not in module.supports_state_modes:
                        raise ValueError(f"model does not support state_mode={mode}")
                    module.state_mode = mode
                    if mode == "converged" and hasattr(module, "retain_response_graph"):
                        module.retain_response_graph = False
            if self.compute_responses is not None:
                for module in owners:
                    if not hasattr(module, "compute_responses"):
                        raise ValueError("model does not support explicit response evaluation")
                    module.compute_responses = self.compute_responses
            prediction = self.model(batch)
            if mode is not None:
                metadata = prediction["variational_prediction"].get("metadata", {})
                required = {"training_mode", "energy_state", "force_derivative",
                            "residual_gradient_policy"}
                if required - metadata.keys() or metadata["energy_state"] != mode:
                    raise ValueError("prediction metadata does not describe the requested state mode")
            return prediction
        finally:
            for module, previous in zip(owners, original):
                module.state_mode = previous
            for module, previous in zip(owners, original_responses):
                if previous is not None:
                    module.compute_responses = previous
            for module, previous in zip(owners, original_graphs):
                if previous is not None:
                    module.retain_response_graph = previous

    def _empty_terms(self):
        return {
            name: {"coefficient": coefficient, "numerator": 0.0, "denominator": 0.0,
                   "n_geometries": 0, "n_conditions": 0}
            for name, coefficient in self.loss_fn.term_coefficients.items()
        }

    @staticmethod
    def _accumulate_terms(terms, result, target):
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

    def _finalize_terms(self, terms, num_conditions):
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
        return total_loss

    @staticmethod
    def _metadata(prediction, previous):
        metadata = prediction["variational_prediction"].get("metadata")
        if previous is not None and metadata != previous:
            raise ValueError("variational prediction semantics change between batches")
        return metadata

    @staticmethod
    def _accumulate_difference(differences, primary, reference):
        labels = primary["variational_prediction"]["labels"]
        other = reference["variational_prediction"]["labels"]
        for name in labels.keys() & other.keys():
            first, second = labels[name], other[name]
            if first["unit"] != second["unit"] or first["shape"] != second["shape"]:
                raise ValueError("predicted and converged label contracts differ")
            if not torch.equal(first["ptr"], second["ptr"]):
                raise ValueError("predicted and converged ragged label layouts differ")
            delta = (first["values"].detach() - second["values"].detach()).reshape(-1)
            entry = differences.setdefault(name, {"sum_squared": 0.0, "reference_sum_squared": 0.0, "n_scalars": 0,
                                                  "unit": first["unit"]})
            entry["sum_squared"] += float(delta.square().sum().cpu())
            entry["reference_sum_squared"] += float(second["values"].detach().square().sum().cpu())
            entry["n_scalars"] += delta.numel()

    @staticmethod
    def _accumulate_response_difference(differences, primary, reference):
        def blocks(prediction):
            entry = prediction["variational_prediction"].get("responses")
            result = {}
            if entry is None:
                return result
            for index in range(len(entry["unit"])):
                key = VariationalJointLoss._response_key(
                    entry["condition_index"][index], entry["input_ref"][index],
                    entry["output_ref"][index], entry["shape"][index], entry["unit"][index],
                    entry["gauge"][index], entry["boundary_hash"][index],
                )
                if key in result:
                    raise ValueError("duplicate response identity in comparison")
                identity = {name: entry[name][index] for name in ("input_ref", "output_ref", "unit", "gauge")}
                value = entry["values"][entry["ptr"][index]:entry["ptr"][index + 1]]
                result[key] = identity, value
            return result

        first, second = blocks(primary), blocks(reference)
        if first.keys() != second.keys():
            raise ValueError("predicted and converged response identities differ")
        for key, (identity, value) in first.items():
            delta = value.detach() - second[key][1].detach()
            # Aggregate only the same observable, units and gauge. Conditions
            # are matched above, before combining their scalar errors.
            group = json.dumps(identity, sort_keys=True)
            entry = differences.setdefault(group, {**identity, "sum_squared": 0.0, "n_scalars": 0})
            entry["sum_squared"] += float(delta.square().sum().cpu())
            entry["n_scalars"] += delta.numel()

    def evaluate(self, dataset):
        if not isinstance(dataset, (Dataset, DataLoader)):
            raise TypeError("VariationalEvaluator requires a Dataset or DataLoader")
        source = dataset.dataset if isinstance(dataset, DataLoader) else dataset
        metadata_source = source
        while not hasattr(metadata_source, "metadata") and hasattr(metadata_source, "dataset"):
            metadata_source = metadata_source.dataset
        for module in self.model.modules():
            validate_metadata = getattr(module, "validate_dataset_metadata", None)
            if callable(validate_metadata):
                validate_metadata(getattr(metadata_source, "metadata", None))
        reference = next(self.model.parameters(), None)
        if reference is None:
            reference = next(self.model.buffers(), torch.empty((), dtype=torch.get_default_dtype()))
        device = torch.device(self.device) if self.device is not None else reference.device
        dtype = reference.dtype if reference.is_floating_point() else torch.get_default_dtype()
        self.model.to(device)
        self.loss_fn.to(device)
        was_training = self.model.training
        self.model.eval()
        terms = self._empty_terms()
        comparison_terms = self._empty_terms() if self.compare_converged else None
        primary_metadata = comparison_metadata = None
        differences = {}
        response_differences = {}
        diagnostics = []
        num_structures = num_conditions = 0
        records = []
        raw_residuals, constraint_errors = [], []
        reference_failures = []
        phase_seconds = {"predicted": 0.0, "reference": 0.0}
        stationarity_unit = None
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
                    mode = self.state_mode or ("predicted" if self.compare_converged else None)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    prediction = self._predict(batch, mode)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    phase_seconds["predicted"] += time.perf_counter() - started
                    entry = prediction["variational_prediction"].get("residuals", {}).get("stationarity", {})
                    normalization = entry.get("normalization", {})
                    if "scale" in normalization:
                        raw_residuals.append(entry["values"].detach().cpu() * normalization["scale"])
                        stationarity_unit = normalization.get("unit")
                    violation = prediction["variational_prediction"].get("constraint_violation")
                    if violation is not None:
                        for left, right in zip(violation["ptr"][:-1], violation["ptr"][1:]):
                            values = violation["values"][left:right].detach()
                            constraint_errors.append(float(values.abs().max().cpu()) if values.numel() else 0.0)
                    primary_metadata = self._metadata(prediction, primary_metadata)
                    result = self.loss_fn(prediction, batch)
                    target = batch["variational"]
                    num_structures += target["condition_ptr"].numel() - 1
                    num_conditions += target["condition_weight"].numel()
                    self._accumulate_terms(terms, result, target)
                    record = {"batch": batch, "prediction": {
                        "variational_prediction": prediction["variational_prediction"],
                    }}
                    if self.compare_converged:
                        started = time.perf_counter()
                        try:
                            comparison = self._predict(batch, "converged")
                        except (RuntimeError, ValueError) as error:
                            if not self.record_reference_failures:
                                raise
                            reference_failures.append({"first_condition": num_conditions - target["condition_weight"].numel(),
                                "n_conditions": target["condition_weight"].numel(), "reason": str(error),
                                "scope": "whole_batch_reference_unavailable"})
                            record["reference_failure"] = reference_failures[-1]
                            if self.save_data:
                                records.append(apply_to_collection(record, torch.Tensor, lambda value: value.detach().cpu().clone()))
                            continue
                        finally:
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            phase_seconds["reference"] += time.perf_counter() - started
                        comparison_metadata = self._metadata(comparison, comparison_metadata)
                        comparison_result = self.loss_fn(comparison, batch)
                        self._accumulate_terms(comparison_terms, comparison_result, target)
                        self._accumulate_difference(differences, prediction, comparison)
                        self._accumulate_response_difference(response_differences, prediction, comparison)
                        diagnostics.extend(comparison["variational_prediction"].get("diagnostics", []))
                        record["comparison_prediction"] = {
                            "variational_prediction": comparison["variational_prediction"],
                        }
                    elif mode == "converged":
                        diagnostics.extend(prediction["variational_prediction"].get("diagnostics", []))
                    if self.save_data:
                        records.append(apply_to_collection(
                            record,
                            torch.Tensor,
                            lambda value: value.detach().cpu().clone(),
                        ))
        finally:
            self.model.train(was_training)

        total_loss = self._finalize_terms(terms, num_conditions)
        self.metrics = {
            "num_structures": num_structures,
            "num_conditions": num_conditions,
            "balance_mode": self.loss_fn.balance_mode,
            "loss": total_loss,
            "terms": terms,
        }
        if primary_metadata is not None:
            self.metrics["metadata"] = primary_metadata
        self.metrics["phase_seconds"] = phase_seconds
        if raw_residuals:
            values = torch.cat(raw_residuals)
            self.metrics["physical_stationarity"] = {
                "rms": float(values.square().mean().sqrt()), "absolute_p95": float(values.abs().quantile(.95)),
                "absolute_max": float(values.abs().max()), "unit": stationarity_unit,
                "n_components": values.numel()}
        if constraint_errors:
            self.metrics["constraints"] = {"max_absolute": max(constraint_errors), "tolerance": 1e-8,
                "n_failed_conditions": sum(value > 1e-8 for value in constraint_errors),
                "n_conditions": len(constraint_errors)}
        if self.compare_converged:
            for value in (*differences.values(), *response_differences.values()):
                if "reference_sum_squared" in value:
                    denominator = value.pop("reference_sum_squared")
                    value["relative_norm_error"] = (math.sqrt(value["sum_squared"] / denominator)
                                                    if denominator > 0 else None)
                squared = value.pop("sum_squared")
                value["scalar_rmse"] = math.sqrt(squared / value["n_scalars"]) if value["n_scalars"] else None
            self.metrics["comparison"] = {
                "metadata": comparison_metadata,
                "loss": self._finalize_terms(comparison_terms, num_conditions),
                "terms": comparison_terms,
                "difference_from_predicted": differences,
                "response_differences": list(response_differences.values()),
            }
        if diagnostics or self.compare_converged:
            self.metrics["reference_solver"] = {
                "scope": "local_stationary_point",
                "n_conditions": num_conditions,
                "n_converged": sum(bool(item["converged"]) for item in diagnostics),
                "n_failed_conditions": sum(item["n_conditions"] for item in reference_failures),
                "failures": reference_failures,
            }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.save_data:
            torch.save({"format_version": 2 if primary_metadata is not None else 1,
                        "batches": records}, self.output_dir / "results.pt")
        (self.output_dir / "metrics.json").write_text(
            json.dumps(self.metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        return self.metrics
