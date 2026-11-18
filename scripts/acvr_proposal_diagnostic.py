#!/usr/bin/env python3
"""Bounded proposal-only diagnosis on a fixed physical functional.

Without --model-path, use the periodic-charge config and its exact teacher.
With --model-path, preserve that checkpoint's backbone and physical functional.
Only train/validation geometries are used. This is not a production trainer.
"""

import argparse
from dataclasses import dataclass, fields, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

from curator.commands.common import ensure_resolvers
from curator.commands.train import _load_model
from curator.data import collate_atoms_data, properties
from curator.data.periodic_charge_oracle import PeriodicChargeOracleDataModule, PeriodicChargeOracleDataset, periodic_charge_coefficients
from curator.layer import BoundaryConditionedProposal, EwaldChargeEnergy, VariationalEnergyOutput
from curator.layer._variational_output import VariationalStateHead
from curator.train.train import tree_to_device
from curator.variational.functionals import solve_functional_reference


class FixedTeacherEnergy(EwaldChargeEnergy):
    """The dataset's exact intrinsic energy, using the shared Ewald runtime."""

    def bind(self, scalar_channels, **kwargs):
        self.scalar_channels = int(scalar_channels)
        return self

    def geometry_context(self, data):
        contexts = super().geometry_context(data)
        ptr = torch.nn.functional.pad(data[properties.n_atoms].cumsum(0), (1, 0))
        edges = data[properties.edge_idx]
        for index, context in enumerate(contexts):
            start, stop = int(ptr[index]), int(ptr[index + 1])
            mask = (edges[:, 0] >= start) & (edges[:, 0] < stop)
            atoms = {properties.atomic_numbers: data[properties.atomic_numbers][start:stop],
                     properties.positions: data[properties.positions][start:stop],
                     properties.edge_idx: edges[mask] - start,
                     properties.edge_dist: data[properties.edge_dist][mask]}
            chi, hardness, constant = periodic_charge_coefficients(atoms)
            context.update(teacher_chi=chi[:, None], teacher_hardness=hardness[:, None], teacher_constant=constant)
        return contexts

    def coefficients(self, context):
        return {"electronegativity": context["teacher_chi"], "hardness": context["teacher_hardness"]}

    def forward(self, context, state):
        return super().forward(context, state) + context["teacher_constant"]


class ZeroBaseline(torch.nn.Module):
    def __init__(self, key):
        super().__init__()
        self.key = key

    def forward(self, data):
        data[self.key] = data[properties.positions].new_zeros(data[properties.n_atoms].numel())
        return data


@dataclass
class FrozenCondition:
    geometry_uid: str
    condition_uid: str
    boundary_kind: str
    context: dict
    compiled: object
    baseline: torch.Tensor
    group_index: torch.Tensor
    group_names: tuple
    provenance: dict


def synchronized_time(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def parameter_hash(model, excluded):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if id(value) not in excluded:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def select_subsets(datasets, args, checkpoint_path, generator):
    """Replay an audited subset independently of the new proposal seed."""
    if args.subset_manifest is None:
        indices = {split: torch.randperm(len(dataset), generator=generator)[:min(limit, len(dataset))].tolist()
                   for (split, dataset), limit in zip(datasets.items(), (args.train_size, args.val_size))}
        return indices, {"selection_seed": args.seed, "subset_manifest": None}
    path = Path(args.subset_manifest).resolve()
    source = json.loads(path.read_text())
    indices = source["dataset_subset_positions"]
    if checkpoint_path and source.get("checkpoint_sha256"):
        if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != source["checkpoint_sha256"]:
            raise ValueError("subset manifest checkpoint hash differs from the fixed functional")
    for split, limit in (("train", args.train_size), ("val", args.val_size)):
        selected = indices[split]
        if len(selected) != limit or len(set(selected)) != len(selected):
            raise ValueError(f"{split} subset manifest must contain exactly {limit} unique positions")
        if any(type(index) is not int or index < 0 or index >= len(datasets[split]) for index in selected):
            raise ValueError(f"{split} subset position is outside the configured split")
        metadata = getattr(datasets[split], "metadata", {})
        if source.get("source_hashes") is not None and metadata.get("source_hashes") != source["source_hashes"]:
            raise ValueError("subset manifest source hashes differ from the configured data")
        if split in source.get("source_indices", {}):
            actual = [datasets[split].indices[index] for index in selected]
            if actual != source["source_indices"][split]:
                raise ValueError(f"{split} source indices differ from the original subset manifest")
    return {split: list(indices[split]) for split in ("train", "val")}, {
        "selection_seed": source.get("seed"), "subset_manifest": str(path),
        "subset_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def prepare_conditions(model, head, samples, device):
    """Cache only frozen tensors; preserve exactly the head's energy assembly."""
    records = []
    model.eval()
    for sample in samples:
        data = tree_to_device(collate_atoms_data([sample]), device=device)
        for module in model.input_modules:
            data = module(data)
        data = model.representation(data)
        for module in model.output_modules:
            if module is head:
                break
            data = module(data)
        scalar, vector, features = head._features(data)
        group_index = torch.full((scalar.shape[0],), -1, dtype=torch.long)
        payload = sample.variational
        for group in range(len(payload.group_names)):
            start, stop = payload.group_member_ptr[group:group + 2]
            members = payload.group_atom_index[start:stop]
            group_index[members] = torch.where(group_index[members] == -1, group, -2)
        atom_ptr = torch.nn.functional.pad(data[properties.n_atoms].cumsum(0), (1, 0))
        indices, boundaries = head._conditions(data)
        extras = head.internal_energy.geometry_context(data) if hasattr(head.internal_energy, "geometry_context") else None
        for index, (geometry, boundary) in enumerate(zip(indices.tolist(), boundaries)):
            context, compiled, start, stop = head._context(data, scalar, vector, geometry, boundary, atom_ptr, index)
            context["features"] = features[start:stop]
            if extras is not None:
                context.update(extras[geometry])
            if hasattr(head.internal_energy, "coefficients"):
                context.update(head.internal_energy.coefficients(context))
            context = {key: value.detach() for key, value in context.items()}
            compiled = replace(compiled, **{field.name: getattr(compiled, field.name).detach()
                                           for field in fields(compiled)
                                           if isinstance(getattr(compiled, field.name), torch.Tensor)})
            baseline = data[head.baseline_key].reshape(-1)[geometry].detach()
            condition = sample.variational.conditions[index]
            records.append(FrozenCondition(
                sample.variational.geometry_uid, condition.condition_uid,
                condition.provenance.get("boundary_kind", "unspecified"), context, compiled, baseline,
                group_index, payload.group_names, condition.provenance))
    return records


def prediction(head, record, context=None):
    context = record.context if context is None else context
    problem = head._problem(context, record.compiled, record.baseline)
    state = problem.project_state(head.proposal(context).reshape(-1))
    return problem, state


def audit(head, records, device, jacobian_geometries, deadline):
    """Keep failed and budget-skipped references visible in the denominator."""
    head.eval()
    rows = []
    jacobian_uids = set(dict.fromkeys(record.geometry_uid for record in records))
    jacobian_uids = set(sorted(jacobian_uids)[:jacobian_geometries])
    started = synchronized_time(device)
    for record in records:
        row = {"geometry_uid": record.geometry_uid, "condition_uid": record.condition_uid,
               "boundary_kind": record.boundary_kind, "reference_status": "not_attempted",
               "jacobian_requested": record.geometry_uid in jacobian_uids,
               "group_index": record.group_index, "group_names": record.group_names,
               "provenance": record.provenance}
        rows.append(row)
        if time.perf_counter() >= deadline:
            row["reference_status"] = "budget_exhausted"
            continue
        context = dict(record.context)
        wants_jacobian = record.geometry_uid in jacobian_uids
        if wants_jacobian:
            field = context["field"].detach().clone().requires_grad_()
            matrix = context["constraint_matrix"]
            tangent = field.flatten()
            if matrix.shape[0]:
                tangent = tangent - matrix.T @ torch.linalg.solve(matrix @ matrix.T, matrix @ tangent)
            context.update(field=field, network_field=tangent.reshape_as(field))
        try:
            before = synchronized_time(device)
            problem, state = prediction(head, record, context)
            row["proposal_seconds"] = synchronized_time(device) - before
            residual = problem.stationarity(state).stationarity
            row.update(state=state.detach().cpu(), residual=residual.detach().cpu(),
                       constraint_error=problem.constraints(state).detach().cpu(),
                       energy=problem.energy(state).detach().cpu())
            if not torch.isfinite(state).all() or not torch.isfinite(residual).all():
                raise ValueError("nonfinite proposal state or stationarity")
            before = synchronized_time(device)
            solved = solve_functional_reference(problem, state.detach())
            row["reference_seconds"] = synchronized_time(device) - before
            row["reference_status"] = solved.status
            row["reference_converged"], row["reference_stable"] = solved.converged, solved.stable
            if not solved.converged or not solved.stable:
                continue
            row["reference_state"] = solved.state.detach().cpu()
            row["reference_energy"] = problem.energy(solved.state).detach().cpu()
            row["reference_residual"] = solved.residuals.stationarity_norm.detach().cpu()
            if wants_jacobian:
                before = synchronized_time(device)
                head.state_mode = "predicted"
                first = head._field_response(problem, state, context["field"])
                head.state_mode = "converged"
                second = head._field_response(problem, solved.state.detach(), context["field"])
                row.update(proposal_jacobian=first.detach().cpu(), reference_jacobian=second.detach().cpu(),
                           jacobian_seconds=synchronized_time(device) - before, jacobian_status="completed")
                atoms = context["field"].shape[0]
                same = torch.eye(atoms, dtype=torch.bool)
                neighbors = torch.zeros_like(same)
                edges = context["edge_index"].detach().cpu()
                neighbors[edges[:, 0], edges[:, 1]] = True
                groups = record.group_index
                far_group = (groups[:, None] != groups[None, :]) & (groups[:, None] >= 0) & (groups[None, :] >= 0)
                row["jacobian_masks"] = {
                    name: mask.repeat_interleave(head.components, 0).repeat_interleave(head.components, 1)
                    for name, mask in (("self", same), ("cross_atom", ~same),
                                       ("neighbor", neighbors & ~same), ("far_group", far_group))}
                if head.components == 1 and all(name in record.group_names for name in ("left", "right")):
                    left = groups == record.group_names.index("left")
                    right = groups == record.group_names.index("right")
                    if bool(left.any()) and bool(right.any()):
                        row["zero_sum_jacobians"] = {}
                        for name in ("proposal", "reference"):
                            block = row[name + "_jacobian"][left][:, right]
                            centered = block - block.mean(1, keepdim=True)
                            row["zero_sum_jacobians"][name] = {
                                "right_zero_sum": centered,
                                "both_zero_sum": centered - centered.mean(0, keepdim=True),
                            }
        except (RuntimeError, ValueError) as error:
            key = "jacobian_status" if "reference_state" in row else "reference_status"
            row[key] = f"{type(error).__name__}: {error}"
        finally:
            head.state_mode = "predicted"
    return summarize(rows, head), rows, synchronized_time(device) - started


def summarize(rows, head):
    valid = [row for row in rows if "reference_state" in row]
    residuals = [row["residual"].flatten() for row in rows if "residual" in row and torch.isfinite(row["residual"]).all()]
    summary = {"conditions": len(rows), "proposal_finite_conditions": len(residuals),
               "reference_successes": len(valid), "reference_failures": len(rows) - len(valid),
               "reference_failures_detail": [{key: row.get(key) for key in ("condition_uid", "reference_status")}
                                             for row in rows if "reference_state" not in row],
               "units": {"residual": head.conjugate_unit, "state": head.state_unit, "energy": head.energy_unit},
               "aggregation": "component-weighted; failed/skipped conditions counted separately, never treated as zero"}
    if residuals:
        residual = torch.cat(residuals)
        constraint = torch.cat([row["constraint_error"].flatten() for row in rows
                                if "constraint_error" in row and torch.isfinite(row["constraint_error"]).all()])
        summary.update(stationarity_rms=float(residual.square().mean().sqrt()),
                       stationarity_p95=float(torch.quantile(residual.abs(), 0.95)),
                       stationarity_max=float(residual.abs().max()),
                       normalized_stationarity_rms=float(residual.square().mean().sqrt()) / head.residual_scale,
                       constraint_max=float(constraint.abs().max()) if constraint.numel() else 0.0,
                       constraint_failed_conditions=sum(bool(row["constraint_error"].abs().gt(1e-8).any())
                                                       for row in rows if "constraint_error" in row))
    if valid:
        delta = torch.cat([row["state"] - row["reference_state"] for row in valid])
        reference = torch.cat([row["reference_state"] for row in valid])
        gaps = torch.stack([row["energy"] - row["reference_energy"] for row in valid])
        summary.update(state_rmse=float(delta.square().mean().sqrt()),
                       state_relative_error=float(delta.norm() / reference.norm().clamp_min(1e-15)),
                       energy_gap_mean=float(gaps.mean()), energy_gap_max=float(gaps.max()))
    jacobians = [row for row in rows if "reference_jacobian" in row]
    summary["jacobian_requested_conditions"] = sum(row.get("jacobian_requested", False) for row in rows)
    summary["jacobian_failed_conditions"] = summary["jacobian_requested_conditions"] - len(jacobians)
    summary["jacobian_failures_detail"] = [
        {"condition_uid": row["condition_uid"], "status": row.get("jacobian_status", row["reference_status"])}
        for row in rows if row.get("jacobian_requested") and "reference_jacobian" not in row]
    if jacobians:
        delta = torch.cat([(row["proposal_jacobian"] - row["reference_jacobian"]).flatten() for row in jacobians])
        reference = torch.cat([row["reference_jacobian"].flatten() for row in jacobians])
        summary.update(jacobian_conditions=len(jacobians), jacobian_rmse=float(delta.square().mean().sqrt()),
                       jacobian_relative_error=float(delta.norm() / reference.norm().clamp_min(1e-15)))
        summary["jacobian_by_location"] = {}
        for name in ("self", "cross_atom", "neighbor", "far_group"):
            selected = [row for row in jacobians if bool(row["jacobian_masks"][name].any())]
            if not selected:
                continue
            delta = torch.cat([(row["proposal_jacobian"] - row["reference_jacobian"])[row["jacobian_masks"][name]] for row in selected])
            reference = torch.cat([row["reference_jacobian"][row["jacobian_masks"][name]] for row in selected])
            summary["jacobian_by_location"][name] = {
                "components": delta.numel(), "rmse": float(delta.square().mean().sqrt()),
                "relative_error": float(delta.norm() / reference.norm().clamp_min(1e-15)),
                "reference_norm": float(reference.norm()),
            }
        for name in ("right_zero_sum", "both_zero_sum"):
            selected = [row["zero_sum_jacobians"] for row in jacobians if "zero_sum_jacobians" in row]
            if selected:
                delta = torch.cat([(row["proposal"][name] - row["reference"][name]).flatten() for row in selected])
                reference = torch.cat([row["reference"][name].flatten() for row in selected])
                summary["jacobian_by_location"][name] = {
                    "components": delta.numel(), "rmse": float(delta.square().mean().sqrt()),
                    "relative_error": float(delta.norm() / reference.norm().clamp_min(1e-15)),
                    "reference_norm": float(reference.norm()),
                }
    paired = [row for row in rows if row.get("provenance", {}).get("local_drive") is True
              and "left" in row.get("group_names", ())]
    delta, reference = [], []
    for row in paired:
        base = next((candidate for candidate in valid if candidate["geometry_uid"] == row["geometry_uid"]
                     and candidate["boundary_kind"] == row["boundary_kind"]
                     and candidate.get("provenance", {}).get("amplitude") == row["provenance"].get("amplitude")
                     and candidate.get("provenance", {}).get("local_drive") is False), None)
        if base is None or "reference_state" not in row:
            continue
        left = row["group_index"] == row["group_names"].index("left")
        actual = (row["state"] - base["state"])[left]
        expected = (row["reference_state"] - base["reference_state"])[left]
        delta.append(actual - expected)
        reference.append(expected)
    if paired:
        summary["paired_left_response"] = {"pairs_requested": len(paired), "pairs_successful": len(delta),
                                            "pairs_failed": len(paired) - len(delta)}
        if delta:
            delta, reference = torch.cat(delta), torch.cat(reference)
            summary["paired_left_response"].update(rmse=float(delta.square().mean().sqrt()),
                relative_error=float(delta.norm() / reference.norm().clamp_min(1e-15)), reference_norm=float(reference.norm()))
    return summary


def run(config, args):
    device = torch.device(args.device)
    # Match Lightning's deterministic=True contract, including CUDA scatter
    # reductions used by the two numerically identical proposal branches.
    cuda_initialized = torch.cuda.is_initialized()
    if device.type == "cuda" and cuda_initialized and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        raise RuntimeError("set CUBLAS_WORKSPACE_CONFIG before initializing CUDA for deterministic diagnosis")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = synchronized_time(device)
    deadline = started + args.time_limit
    if args.model_path:
        config.model_path = args.model_path
    checkpoint_path = None
    if config.model_path:
        from curator.utils import find_best_model

        source = config.model_path
        if not isinstance(source, str):
            if source.get("mode", "model") != "model":
                raise ValueError("diagnostic model_path requires mode=model; optimizer resume and partial weights are not supported")
            source = source["path"]
        checkpoint_path = Path(find_best_model(source)[0]).resolve()
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    write_json(output / "arguments.json", vars(args))
    sources = {}
    for source in [*sorted((ROOT / "curator").rglob("*.py")), Path(__file__).resolve()]:
        relative = source.relative_to(ROOT)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        sources[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(output / "source_manifest.json", sources)

    module = instantiate(config.data, _convert_="all")
    if isinstance(module, PeriodicChargeOracleDataModule):
        # Do not call setup(): its general training lifecycle also constructs
        # test data. The diagnostic never generates or reads test labels.
        datasets = {split: PeriodicChargeOracleDataset(split=split, **module.dataset_kwargs)
                    for split in ("train", "val")}
    else:
        if not config.model_path:
            raise ValueError("nonperiodic diagnosis requires --model-path for its fixed functional")
        module.setup("fit")
        datasets = {split: module.datasets[split] for split in ("train", "val")}
    generator = torch.Generator().manual_seed(args.seed)
    indices, selection = select_subsets(datasets, args, checkpoint_path, generator)
    if args.subset_manifest:
        shutil.copy2(args.subset_manifest, output / "subset_manifest.json")
    samples = {split: [dataset[index] for index in indices[split]] for split, dataset in datasets.items()}
    manifest = {"test_evaluated": False, **selection, "splits": {}}
    for split in samples:
        manifest["splits"][split] = {
            "dataset_metadata": getattr(datasets[split], "metadata", {}), "positions_in_split": indices[split],
            "geometries": [{"uid": sample.variational.geometry_uid, "hash": sample.variational.geometry_hash,
                            "geometry_inputs": {key: sample.atoms[key].detach().cpu().tolist()
                                                for key in ("external_field", "local_external_field") if key in sample.atoms},
                            "conditions": [{"uid": condition.condition_uid, "boundary_hash": condition.boundary_hash,
                                            "label_present": sorted(condition.labels),
                                            "label_eligible": condition.label_eligible,
                                            "provenance": condition.provenance,
                                            "inputs": {key: value.detach().cpu().tolist() for key, value in condition.inputs.items()}}
                                           for condition in sample.variational.conditions]}
                           for sample in samples[split]],
        }
    write_json(output / "data_manifest.json", manifest)
    model = _load_model(config, build_dtype=torch.float64).model
    heads = [module for module in model.modules() if isinstance(module, VariationalEnergyOutput)]
    if len(heads) != 1:
        raise ValueError("diagnostic requires exactly one VariationalEnergyOutput")
    head = heads[0]
    old_proposal = type(head.proposal).__name__
    state_scale = args.state_scale if args.state_scale is not None else getattr(head.proposal, "state_scale", 1.0)
    field_scale = args.field_scale if args.field_scale is not None else getattr(head.proposal, "field_scale", 1.0)
    if not config.model_path:
        if not isinstance(head.internal_energy, EwaldChargeEnergy):
            raise ValueError("fixed teacher requires EwaldChargeEnergy and the periodic-charge configuration")
        head.internal_energy = FixedTeacherEnergy(ewald=head.internal_energy.ewald)
        head.internal_energy.bind(head.scalar_channels)
        model.output_modules.insert(0, ZeroBaseline(head.baseline_key))
    head.observable_specs = ()
    head.compute_forces = head.produces_forces = head.compute_responses = False
    head.retain_response_graph = False
    # Keep proposal initialization independent of checkpoint construction and
    # retain the configured dataset seed/splits across repeat training seeds.
    torch.manual_seed(args.seed)
    if args.communication == "baseline":
        head.proposal = VariationalStateHead(head.scalar_channels, head.vector_channels, head.components,
                                            head.state_kind, args.hidden_features)
    else:
        head.proposal = BoundaryConditionedProposal(args.communication, args.hidden_features, args.layers, args.heads,
                                                    state_scale=state_scale, field_scale=field_scale)
        head.proposal.bind(head.feature_spec, head.components, head.state_kind, head.cutoff)
    model.to(device=device, dtype=torch.float64)
    model.requires_grad_(False)
    head.proposal.requires_grad_(True)
    proposal_ids = {id(parameter) for parameter in head.proposal.parameters()}
    frozen_hash = parameter_hash(model, proposal_ids)
    migration = {"source_model_path": str(config.model_path) if config.model_path else None,
                 "source_checkpoint": str(checkpoint_path) if checkpoint_path else None,
                 "source_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() if checkpoint_path else None,
                 "old_proposal": old_proposal, "new_proposal": type(head.proposal).__name__,
                 "new_proposal_contract": head.proposal.checkpoint_contract() if hasattr(head.proposal, "checkpoint_contract") else {"class": "VariationalStateHead", "hidden_features": args.hidden_features},
                 "optimizer": "new Adam; source optimizer never restored",
                 "functional": "checkpoint unchanged" if config.model_path else "periodic fixed teacher; zero baseline",
                 "frozen_parameters": [name for name, parameter in model.named_parameters() if not parameter.requires_grad],
                 "proposal_parameters": sum(parameter.numel() for parameter in head.proposal.parameters()),
                 "frozen_parameter_sha256": frozen_hash}
    write_json(output / "migration.json", migration)
    torch.save(head.proposal.state_dict(), output / "initial_proposal.pt")
    before = synchronized_time(device)
    records = {split: prepare_conditions(model, head, values, device) for split, values in samples.items()}
    cache_seconds = synchronized_time(device) - before
    before = synchronized_time(device)
    full = model(tree_to_device(collate_atoms_data([samples["val"][0]]), device))["variational_prediction"]
    full_forward_seconds = synchronized_time(device) - before
    first_uid = samples["val"][0].variational.geometry_uid
    cached = torch.cat([prediction(head, record)[1].detach() for record in records["val"] if record.geometry_uid == first_uid])
    torch.testing.assert_close(full["labels"]["variable_value"]["values"].detach(), cached, rtol=1e-10, atol=1e-12)
    initial, initial_rows, initial_seconds = audit(head, records["val"], device, args.jacobian_geometries, deadline)
    initial["by_boundary"] = {kind: summarize([row for row in initial_rows if row["boundary_kind"] == kind], head)
                              for kind in sorted({row["boundary_kind"] for row in initial_rows})}
    write_json(output / "initial.json", initial)
    torch.save(initial_rows, output / "initial_states.pt")
    optimizer = torch.optim.Adam(head.proposal.parameters(), lr=args.lr)
    history, status = [], "completed"
    geometries = list(dict.fromkeys(record.geometry_uid for record in records["train"]))
    grouped = [[record for record in records["train"] if record.geometry_uid == uid] for uid in geometries]
    order, cursor = [], 0
    head.train()
    for step in range(args.steps):
        if time.perf_counter() >= deadline:
            status = "budget_exhausted"
            break
        before = synchronized_time(device)
        if cursor >= len(order):
            order = torch.randperm(len(grouped), generator=generator).tolist()
            cursor = 0
        selected = order[cursor:cursor + args.batch_size]
        cursor += len(selected)
        optimizer.zero_grad(set_to_none=True)
        geometry_losses = []
        for index in selected:
            losses = []
            for record in grouped[index]:
                problem, state = prediction(head, record)
                residual = problem.stationarity(state, create_graph=True).stationarity / head.residual_scale
                losses.append(residual.square().mean())
            geometry_losses.append(torch.stack(losses).mean())
        loss = args.residual_weight * torch.stack(geometry_losses).mean()
        if not bool(torch.isfinite(loss)):
            status = "nonfinite_loss"
            break
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.proposal.parameters(), 10.0)
        if not bool(torch.isfinite(norm)):
            status = "nonfinite_gradient"
            break
        optimizer.step()
        history.append({"step": step + 1, "loss": float(loss.detach()), "gradient_norm": float(norm),
                        "seconds": synchronized_time(device) - before, "geometry_indices": selected})
        with (output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(history[-1], allow_nan=False) + "\n")
        if (step + 1) % 25 == 0:
            print(json.dumps(history[-1]), flush=True)
    final, final_rows, final_seconds = audit(head, records["val"], device, args.jacobian_geometries, deadline)
    final["by_boundary"] = {kind: summarize([row for row in final_rows if row["boundary_kind"] == kind], head)
                            for kind in sorted({row["boundary_kind"] for row in final_rows})}
    after_hash = parameter_hash(model, proposal_ids)
    if after_hash != frozen_hash:
        raise RuntimeError("fixed physical/backbone parameters changed during proposal diagnosis")
    result = {"status": status, "steps_requested": args.steps, "steps_completed": len(history),
              "test_evaluated": False, "time_limit_seconds": args.time_limit,
              "elapsed_seconds": synchronized_time(device) - started, "cache_build_seconds": cache_seconds,
              "full_model_forward_seconds_one_validation_geometry": full_forward_seconds,
              "full_model_forward_conditions": len(samples["val"][0].variational.conditions),
              "full_model_forward_scope": "geometry, kernel, every condition proposal/energy/residual; property derivatives disabled; one un-warmed timing",
              "initial_audit_seconds": initial_seconds, "final_audit_seconds": final_seconds,
              "training_step_seconds_mean": sum(row["seconds"] for row in history) / max(len(history), 1),
              "timing_scope": "training uses cached frozen features and physical kernels; cache and reference costs reported separately; not an end-to-end speedup claim",
              "physical_parameters_unchanged": True, "initial": initial, "final": final}
    result["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    result["runtime"] = {"torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                         "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                         "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                         "cuda_initialized_before_workspace_setup": cuda_initialized,
                         "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    write_json(output / "result.json", result)
    write_json(output / "training_history.json", history)
    torch.save(final_rows, output / "final_states.pt")
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "steps": len(history), "generator_state": generator.get_state(), "migration": migration}, output / "checkpoint.pt")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "curator/configs/acvr_periodic_charge.yaml"))
    parser.add_argument("--model-path")
    parser.add_argument("--communication", choices=("baseline", "pointwise", "mp", "attention", "ewald"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--val-size", type=int, default=32)
    parser.add_argument("--subset-manifest", help="Replay dataset_subset_positions and validate original source/checkpoint hashes")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-features", type=int, default=48)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--state-scale", type=float, help="Override the base proposal state scale; legacy checkpoints default to 1")
    parser.add_argument("--field-scale", type=float, help="Override the base proposal field scale; legacy checkpoints default to 1")
    parser.add_argument("--jacobian-geometries", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--residual-weight", type=float, default=0.1)
    parser.add_argument("--time-limit", type=float, default=3600)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--override", action="append", default=[], help="Hydra override; may be repeated")
    args = parser.parse_args(argv)
    if min(args.steps, args.train_size, args.val_size, args.batch_size, args.threads) <= 0 or args.jacobian_geometries < 0:
        parser.error("steps, split sizes, batch size and threads must be positive; Jacobian count must be nonnegative")
    if not all(value > 0 and value < float("inf") for value in (args.lr, args.residual_weight, args.time_limit)):
        parser.error("learning rate, residual weight and time limit must be finite and positive")
    if any(value is not None and not (0 < value < float("inf")) for value in (args.state_scale, args.field_scale)):
        parser.error("proposal scales must be finite and positive")
    ensure_resolvers()
    path = Path(args.config).resolve()
    with initialize_config_dir(config_dir=str(path.parent), version_base=None):
        config = compose(config_name=path.stem, overrides=args.override)
    existed = Path(args.output).exists()
    try:
        result = run(config, args)
    except Exception as error:
        if not existed and Path(args.output).is_dir():
            write_json(Path(args.output) / "failure.json", {
                "status": "failed", "error": f"{type(error).__name__}: {error}",
                "test_evaluated": False,
            })
        raise
    print(json.dumps({"status": result["status"], "steps": result["steps_completed"], "output": args.output}), flush=True)
    return result


if __name__ == "__main__":
    main()
