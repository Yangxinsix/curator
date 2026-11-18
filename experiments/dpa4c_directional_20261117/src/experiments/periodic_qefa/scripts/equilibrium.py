"""Read-only native equilibrium and same-weight charge counterfactuals."""
from contextlib import contextmanager
import hashlib
import json

from common import ROOT, SOURCE, load_model, read_json, write_json

import numpy as np
import torch
from curator.data import properties as p
from curator.data.collate_atoms_data import collate_atoms_data
from curator.layer import VariationalEnergyOutput
from curator.train.train import tree_to_device


GEOMETRY_KEYS = (
    p.atomic_numbers, p.positions, p.cell, p.pbc, p.n_atoms, p.image_idx,
    p.domain, p.domain_atom, p.atomic_types, p.edge_idx, p.edge_diff,
    p.edge_dist, p.n_pairs, p.cell_displacements, p.total_charge,
)
CONDITION_KEYS = (
    "condition_ptr", "condition_weight", "condition_geometry_index",
    "condition_state_ptr", "boundary_spec", "boundary_hash", "condition_inputs",
)


def model_inputs(batch, device):
    """Never expose energy, force, or charge targets to the model."""
    inputs = {key: batch[key].clone() for key in GEOMETRY_KEYS if key in batch}
    inputs["variational"] = {key: batch["variational"][key]
                             for key in CONDITION_KEYS if key in batch["variational"]}
    return tree_to_device(inputs, device)


def _fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


@contextmanager
def _read_only(model, device):
    heads = [head for head in model.output_modules
             if isinstance(head, VariationalEnergyOutput)]
    if len(heads) != 1:
        raise ValueError("Expected one native VariationalEnergyOutput")
    head = heads[0]
    actual, requested = next(model.parameters()).device, torch.device(device)
    if actual.type != requested.type or (
            requested.index is not None and actual.index != requested.index):
        raise ValueError("Move the model to the requested device before evaluation")
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Equilibrium forces require inference_mode=False")
    modes = [(module, module.training) for module in model.modules()]
    gradients = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    keys = ("state_mode", "compute_forces", "retain_response_graph",
            "compute_responses", "residual_gradient_policy")
    settings = {key: getattr(head, key) for key in keys}
    before = _fingerprint(model)
    try:
        model.eval()
        for parameter, _ in gradients:
            parameter.requires_grad_(False)
        head.compute_forces = True
        head.retain_response_graph = head.compute_responses = False
        head.residual_gradient_policy = "proposal_only"
        with torch.enable_grad():
            yield head
    finally:
        for key, value in settings.items():
            setattr(head, key, value)
        for parameter, flag in gradients:
            parameter.requires_grad_(flag)
        for module, training in modes:
            module.training = training
        if _fingerprint(model) != before:
            raise RuntimeError("Read-only evaluation changed model parameters or buffers")


def capture_context(model, batch, device="cpu"):
    """Capture detached native geometry for charge-only derivative calibration."""
    calls = []
    with _read_only(model, device) as head:
        head.state_mode = "predicted"
        head.compute_forces = False
        hook = head.internal_energy.register_forward_pre_hook(
            lambda module, args: calls.append(args))
        try:
            result = model(model_inputs(batch, device))["variational_prediction"]
        finally:
            hook.remove()
        if not calls:
            raise RuntimeError("Native forward did not call the charge functional")
        context, live_q = calls[0]
        context = {key: value.detach() if torch.is_tensor(value) else value
                   for key, value in context.items()}
        q = result["labels"]["variable_value"]["values"].detach().reshape(-1, 1)
        torch.testing.assert_close(q, live_q.detach(), atol=0, rtol=0)
    return context, q, head


def _forward(model, head, batch, device, mode):
    head.state_mode = "converged" if mode == "equilibrium" else "predicted"
    hook = None
    if mode in ("uniform", "geometry"):
        def replace(module, args):
            context, state = args
            matrix, target = context["constraint_matrix"], context["constraint_target"]
            torch.testing.assert_close(matrix, state.new_ones((1, state.numel())),
                                       atol=0, rtol=0)
            if (torch.count_nonzero(context["field"])
                    or context["control_matrix"].numel()
                    or context["environment_matrix"].numel()):
                raise ValueError("Counterfactuals require fixed total charge without external fields")
            replacement = torch.zeros_like(state)
            if mode == "uniform":
                replacement = replacement + target.detach()[0] / state.numel()
            # Keep a zero state derivative for the native residual machinery.
            return context, replacement + 0 * state
        hook = head.internal_energy.register_forward_pre_hook(replace)
    try:
        prediction = model(model_inputs(batch, device))["variational_prediction"]
    finally:
        if hook is not None:
            hook.remove()
    if prediction["value_space"] != "physical":
        raise ValueError("Expected physical prediction units")
    labels = prediction["labels"]
    for name, unit in (("condition_energy", "eV"), ("condition_forces", "eV/Angstrom"),
                       ("variable_value", "e")):
        if labels[name]["unit"] != unit:
            raise ValueError(f"Unexpected {name} unit")
    energy = labels["condition_energy"]["values"].detach().double().cpu().reshape(-1)
    forces = labels["condition_forces"]["values"].detach().double().cpu().reshape(-1, 3)
    q = labels["variable_value"]["values"].detach().double().cpu().reshape(-1)
    if energy.numel() != 1:
        raise ValueError("Evaluator requires one condition of one geometry")
    values = {"energy": float(energy[0]), "forces": forces, "q": q,
              "solver": prediction["diagnostics"]}
    if mode in ("equilibrium", "proposal"):
        residual = prediction["residuals"]["stationarity"]
        r = residual["values"].detach().double().cpu().reshape(-1)
        r = r * float(residual["normalization"]["scale"])
        violation = prediction["constraint_violation"]["values"].detach().double().cpu()
        values.update(stationarity=r, constraint_max=float(violation.abs().max()))
    if mode == "equilibrium":
        if prediction["metadata"]["force_derivative"] != "stationary_envelope":
            raise RuntimeError("Equilibrium prediction lacks stationary envelope forces")
        if not values["solver"] or any(
                not row["converged"] or not row["stable"] for row in values["solver"]):
            raise RuntimeError("Native equilibrium solve did not converge stably")
    return values


def _metrics(value, energy_target, force_target, count, physical_state):
    error = (value["energy"] - energy_target) / count
    force_error = value["forces"] - force_target
    row = {
        "energy_centered_eV": value["energy"],
        "energy_error_eV_per_atom": error,
        "energy_mse_eV2_per_atom2": error ** 2,
        "force_mse_eV2_A2": float(force_error.square().mean()),
        "force_squared_error_sum_eV2_A2": float(force_error.square().sum()),
    }
    if physical_state:
        q, r = value["q"], value["stationarity"]
        row.update(
            charge_e=q.tolist(), charge_rms_e=float(q.square().mean().sqrt()),
            charge_centered_rms_e=float((q - q.mean()).square().mean().sqrt()),
            stationarity_rms_V=float(r.square().mean().sqrt()),
            stationarity_squared_sum_V2=float(r.square().sum()),
            constraint_max_e=value["constraint_max"], solver=value["solver"],
        )
    return row


def aggregate_frames(rows):
    if not rows:
        raise ValueError("Expected nonempty evaluation frames")
    aggregate = {"structures": len(rows), "atoms": sum(row["n_atoms"] for row in rows),
                 "modes": {}}
    for mode in rows[0]["modes"]:
        values = [row["modes"][mode] for row in rows]
        energy_mse = float(np.mean([value["energy_mse_eV2_per_atom2"] for value in values]))
        force_mse = float(np.mean([value["force_mse_eV2_A2"] for value in values]))
        result = {
            "energy_RMSE_meV_per_atom": 1000 * energy_mse ** .5,
            "force_structure_RMSE_meV_A": 1000 * force_mse ** .5,
            "force_component_RMSE_meV_A": 1000 * (
                sum(value["force_squared_error_sum_eV2_A2"] for value in values)
                / (3 * aggregate["atoms"])) ** .5,
            "EF_score": 200 * (energy_mse + force_mse),
        }
        if mode in ("equilibrium", "proposal"):
            result.update(
                stationarity_rms_V=(sum(value["stationarity_squared_sum_V2"]
                                       for value in values) / aggregate["atoms"]) ** .5,
                constraint_max_e=max(value["constraint_max_e"] for value in values),
            )
        aggregate["modes"][mode] = result
    return aggregate


def evaluate_frames(model, data, indices, device="cpu", include_proposal=True,
                    variants=None):
    """Evaluate original source indices; the caller owns split eligibility."""
    indices = list(indices)
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("Evaluation source indices must be unique and nonempty")
    samples = data.datasets["train"].samples
    modes = tuple(variants) if variants is not None else (
        "equilibrium", "uniform", "geometry", "proposal")
    if not include_proposal:
        modes = tuple(mode for mode in modes if mode != "proposal")
    if ("equilibrium" not in modes or len(set(modes)) != len(modes)
            or not set(modes) <= {"equilibrium", "uniform", "geometry", "proposal"}):
        raise ValueError("Expected unique supported variants including equilibrium")
    rows = []
    with _read_only(model, device) as head:
        for index in indices:
            batch = collate_atoms_data([samples[index]])
            if batch["meta"][0]["source_index"] != index:
                raise ValueError("Sample source index does not match global lookup")
            count = int(batch[p.n_atoms].sum())
            target_e = float(batch[p.energy].detach().double().sum())
            target_f = batch[p.forces].detach().double().cpu().reshape(count, 3)
            values = {mode: _forward(model, head, batch, device, mode) for mode in modes}
            row = {"source_index": index, "n_atoms": count,
                   "energy_target_centered_eV": target_e,
                   "modes": {mode: _metrics(value, target_e, target_f, count,
                                           mode in ("equilibrium", "proposal"))
                             for mode, value in values.items()}}
            if "geometry" in modes:
                geometry_error = target_f - values["geometry"]["forces"]
                correction = values["equilibrium"]["forces"] - values["geometry"]["forces"]
                dot = float((geometry_error * correction).mean())
                norm = float(correction.square().mean())
                row["force_alignment"] = {
                    "geometry_error_dot_charge_force_eV2_A2": dot,
                    "charge_force_mse_eV2_A2": norm,
                    "predicted_geometry_mse_improvement_eV2_A2": 2 * dot - norm,
                }
            if "uniform" in modes:
                row["equilibrium_minus_uniform_energy_eV"] = (
                    values["equilibrium"]["energy"] - values["uniform"]["energy"])
                row["equilibrium_minus_uniform_force_rms_eV_A"] = float(
                    (values["equilibrium"]["forces"] - values["uniform"]["forces"])
                    .square().mean().sqrt())
            if "proposal" in modes:
                row["proposal_minus_equilibrium_charge_rms_e"] = float(
                    (values["proposal"]["q"] - values["equilibrium"]["q"])
                    .square().mean().sqrt())
                row["proposal_minus_equilibrium_energy_eV"] = (
                    values["proposal"]["energy"] - values["equilibrium"]["energy"])
                row["proposal_minus_equilibrium_force_rms_eV_A"] = float(
                    (values["proposal"]["forces"] - values["equilibrium"]["forces"])
                    .square().mean().sqrt())
            rows.append(row)
    result = {"aggregate": aggregate_frames(rows), "frames": rows,
              "conventions": {
                  "geometry": "same frozen model with charge energy evaluated at q=0",
                  "uniform": "same frozen model with detached q=Q/N; artificial residual ignored",
                  "equilibrium": "native converged mode; stationary-envelope force",
                  "gate_unit": "one structure, not an atom or force component",
              }}
    json.dumps(result, allow_nan=False)
    return result


def paired_gate(frames, *, bootstrap_samples=10000, seed=20261020,
                numerical_mse_floor=None, group_ids=None):
    """One-sided 95% paired bootstrap gate for both E/F and both controls.

    numerical_mse_floor maps the two metric names below to measured MSE
    uncertainty from solver/precision audits. Without it the statistical
    result is reported, but cannot authorize formal training.
    """
    if len(frames) < 2 or bootstrap_samples < 1000:
        raise ValueError("Need at least two frames and 1000 bootstrap replicates")
    metrics = ("energy_mse_eV2_per_atom2", "force_mse_eV2_A2")
    if numerical_mse_floor is not None and any(
            name not in numerical_mse_floor
            or not np.isfinite(numerical_mse_floor[name])
            or numerical_mse_floor[name] < 0 for name in metrics):
        raise ValueError("Supply finite nonnegative measured numerical MSE floors")
    groups = list(range(len(frames))) if group_ids is None else list(group_ids)
    if len(groups) != len(frames):
        raise ValueError("group_ids must align with frame order")
    unique = list(dict.fromkeys(groups))
    if len(unique) < 2:
        raise ValueError("Need at least two independent resampling groups")
    membership = np.array([[value == group for value in groups] for group in unique])
    sizes = membership.sum(axis=1)
    rng = np.random.default_rng(seed)
    draw = rng.integers(len(unique), size=(bootstrap_samples, len(unique)))
    comparisons = {}
    for control in ("geometry", "uniform"):
        comparison = {}
        for metric in metrics:
            differences = np.array([row["modes"]["equilibrium"][metric]
                                    - row["modes"][control][metric] for row in frames])
            group_sums = membership @ differences
            bootstrap = group_sums[draw].sum(axis=1) / sizes[draw].sum(axis=1)
            upper = float(np.quantile(bootstrap, .95))
            floor = None if numerical_mse_floor is None else float(numerical_mse_floor[metric])
            comparison[metric] = {
                "mean_paired_difference": float(differences.mean()),
                "one_sided_95pct_upper": upper, "statistical_passed": upper < 0,
                "numerical_mse_floor": floor,
                "passed": floor is not None and upper < -floor,
            }
        comparisons[control] = comparison
    checks = [value for comparison in comparisons.values() for value in comparison.values()]
    result = {
        "passed": all(value["passed"] for value in checks),
        "statistical_passed": all(value["statistical_passed"] for value in checks),
        "numerical_audit_provided": numerical_mse_floor is not None,
        "structures": len(frames), "resampling_groups": len(unique),
        "bootstrap_samples": bootstrap_samples, "seed": seed,
        "criterion": "equilibrium improves both E/N MSE and structure-mean F MSE against both geometry and same-weight uniform q",
        "comparisons": comparisons,
    }
    json.dumps(result, allow_nan=False)
    return result
