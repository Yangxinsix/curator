"""Evaluate a reconstructed SPICE checkpoint on its persisted distance splits."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from hydra.utils import instantiate
from lightning_utilities.core.apply_func import apply_to_collection

from curator.data import properties as p
from curator.utils import find_best_model
from curator.variational import QuadraticProblem, project_to_equalities, solve_amortized, solve_dense


def restore(path, device):
    found = find_best_model(path)
    if found is None:
        raise FileNotFoundError(f"No checkpoint under {path}")
    checkpoint_path = found[0].resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data = instantiate(payload["data_params"], _convert_="all")
    data.setup()
    model = instantiate(payload["model_params"], _convert_="all")
    model.initialize_modules(data)
    state = {key.removeprefix("model."): value for key, value in payload["state_dict"].items()
             if key.startswith("model.")}
    model.load_state_dict(state, strict=True)
    model.representation.validate_dataset_metadata(data.metadata)
    model.eval().to(device=device, dtype=torch.float64)
    return model, data, checkpoint_path, payload


def errors(prediction, target):
    delta = prediction - target
    if not torch.isfinite(delta).all():
        raise ValueError("Non-finite evaluation prediction or label")
    return {"rmse": delta.square().mean().sqrt().item(), "mae": delta.abs().mean().item(),
            "max_abs": delta.abs().max().item(), "n_components": delta.numel()}


def evaluate(model, data, split, device):
    loader = data.val_dataloader() if split == "val" else data.test_dataloader()
    chunks, rows, guesses, problems = [], [], [], []
    offset = 0
    representation = model.representation
    for batch in loader:
        batch = apply_to_collection(batch, torch.Tensor, lambda value: value.to(device))
        prediction = model(batch)
        count = batch[p.n_atoms].numel()
        record = {"positions": batch[p.positions], "atomic_numbers": batch[p.atomic_numbers],
                  "total_charge": batch[p.total_charge]}
        for name in (p.energy, p.forces, p.dipole):
            record[name + "_target"] = batch[name]
            record[name + "_prediction"] = prediction[name]
        record["atomic_charge_prediction"] = prediction[p.atomic_charge]
        record["source_index"] = torch.tensor([item["source_index"] for item in batch["meta"]], dtype=torch.long)
        record = apply_to_collection(record, torch.Tensor, lambda value: value.detach().cpu())
        chunks.append(record)
        positions = record["positions"].reshape(count, 2, 3)
        numbers = record["atomic_numbers"].reshape(count, 2)
        distances = torch.linalg.vector_norm(positions[:, 1] - positions[:, 0], dim=-1)
        directions = (positions[:, 1] - positions[:, 0]) / distances[:, None]
        for index in range(count):
            row = {"split": split, "split_index": offset + index,
                   "source_index": int(record["source_index"][index]), "source_group": batch["meta"][index]["group"],
                   "Z0": int(numbers[index, 0]), "Z1": int(numbers[index, 1]),
                   "Q_e": float(record["total_charge"].reshape(-1)[index]),
                   "distance_A": float(distances[index])}
            for kind in ("target", "prediction"):
                row["energy_eV_" + kind] = float(record[p.energy + "_" + kind].reshape(-1)[index])
                force = record[p.forces + "_" + kind].reshape(count, 2, 3)[index]
                row["force_radial_eV_A_" + kind] = float(force[1] @ directions[index])
                dipole = record[p.dipole + "_" + kind].reshape(count, 3)[index]
                for axis, value in zip("xyz", dipole):
                    row[f"dipole_{axis}_eA_{kind}"] = float(value)
            charges = record["atomic_charge_prediction"].reshape(count, 2)[index]
            row.update(q0_e=float(charges[0]), q1_e=float(charges[1]),
                       charge_constraint_error_e=float(charges.sum()) - row["Q_e"])
            rows.append(row)
        if representation.variant == "acvr":
            # Audits are deliberately on CPU: small dense solver timings are
            # not evidence for GPU throughput or a production speedup.
            for problem, guess in zip(representation.problems(batch),
                                      representation.propose(batch).reshape(count, 2)):
                problems.append(QuadraticProblem(**{
                    key: getattr(problem, key).detach().cpu()
                    for key in ("hessian", "linear", "constraint_matrix", "constraint_target", "constant")
                }))
                guesses.append(guess.detach().cpu())
        offset += count
    merged = {key: torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}
    metrics = {name: errors(merged[name + "_prediction"], merged[name + "_target"])
               for name in (p.energy, p.forces, p.dipole)}
    metrics["n_geometries"] = offset
    metrics["max_charge_constraint_error_e"] = max(abs(row["charge_constraint_error_e"]) for row in rows)
    return metrics, rows, merged, problems, guesses


def audit_proposal(problems, guesses, rows=None):
    if not problems or len(problems) != len(guesses):
        raise ValueError("proposal audit requires equally sized nonempty problem/guess lists")
    exact = [solve_dense(problem).state.detach() for problem in problems]
    projected = [project_to_equalities(guess, problem.constraint_matrix, problem.constraint_target).state
                 for problem, guess in zip(problems, guesses)]
    result = {}
    for tolerance in (1e-2, 1e-3, 1e-4, 1e-8):
        solved = [solve_amortized(problem, guess, residual_tolerance=tolerance,
                                 feasibility_tolerance=1e-10, max_correction_steps=1, fallback=True)
                  for problem, guess in zip(problems, guesses)]
        statuses = [value.status for value in solved]
        count = len(solved)
        result[str(tolerance)] = {
            "n_geometries": count,
            "accepted_projected": statuses.count("accepted_projected"),
            "mean_correction_steps": sum(value.iterations for value in solved) / count,
            "fallback_count": sum(value.used_fallback for value in solved),
            "statuses": {status: statuses.count(status) for status in sorted(set(statuses))},
            "returned_state_rmse_e": torch.cat([
                value.state.detach() - target for value, target in zip(solved, exact)
            ]).square().mean().sqrt().item(),
            "unreliable_count": sum(not value.reliable for value in solved),
            "returned_energy_error_rmse_eV": torch.stack([
                value.energy.detach() - problem.energy(target)
                for value, problem, target in zip(solved, problems, exact)
            ]).square().mean().sqrt().item(),
        }
    result["projected_proposal_state_rmse_e"] = torch.cat([
        guess - target for guess, target in zip(projected, exact)
    ]).square().mean().sqrt().item()
    result["raw_proposal_state_rmse_e"] = torch.cat([
        guess - target for guess, target in zip(guesses, exact)
    ]).square().mean().sqrt().item()
    result["scope"] = "Fixed learned physical model; raw proposal before projection; dense certification; no speed claim"
    if rows is not None:
        if len(rows) != len(problems):
            raise ValueError("proposal pair metadata must cover every problem")
        result["by_pair_symmetry"] = {}
        for name, equal in (("homonuclear", True), ("heteronuclear", False)):
            indices = [index for index, row in enumerate(rows) if (row["Z0"] == row["Z1"]) == equal]
            if indices:
                result["by_pair_symmetry"][name] = audit_proposal(
                    [problems[index] for index in indices], [guesses[index] for index in indices])
                result["by_pair_symmetry"][name]["n_geometries"] = len(indices)
        result["symmetry_note"] = (
            "For homonuclear pairs, exchange symmetry and fixed total charge force q=Q/2. "
            "Their projected proposals can therefore be exact before training. The 7 homonuclear "
            "pairs among 28 species pairs must not be counted as learned acceptance gains; "
            "compare trained/untrained heteronuclear audits separately."
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file or model_path directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--reference-checkpoint", help="Physical checkpoint used to initialize proposal")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, data, path, payload = restore(args.checkpoint, args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema": "spice_acvr_evaluation_v1", "checkpoint": str(path),
              "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "epoch": payload["epoch"], "global_step": payload["global_step"],
              "model_seed": model.representation.seed,
              "variant": model.representation.variant, "stage": model.representation.stage,
              "dataset_metadata": data.metadata, "units": {"energy": "eV", "forces": "eV/Angstrom", "dipole": "e*Angstrom"},
              "splits": {}}
    reference = None
    if args.reference_checkpoint:
        reference, reference_data, reference_path, _ = restore(args.reference_checkpoint, args.device)
        if reference_data.metadata != data.metadata:
            raise ValueError("Physical/proposal data manifests differ")
        before = dict(reference.representation.named_parameters())
        changed = []
        for name, value in model.representation.named_parameters():
            equal = torch.equal(value, before[name])
            if not name.startswith("proposal") and not equal:
                raise ValueError(f"Proposal changed physical parameter {name}")
            if name.startswith("proposal") and not equal:
                changed.append(name)
        if not changed:
            raise ValueError("No proposal parameters changed")
        report["physical_parameter_check"] = {"identical": True, "changed_proposal_parameters": changed,
                                               "reference_checkpoint": str(reference_path),
                                               "reference_checkpoint_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest()}
    for split in ("val", "test"):
        metrics, rows, raw, problems, guesses = evaluate(model, data, split, args.device)
        if problems:
            metrics["proposal"] = audit_proposal(problems, guesses, rows)
        if reference is not None:
            ref_metrics, ref_rows, ref_raw, ref_problems, ref_guesses = evaluate(reference, data, split, args.device)
            for name in (p.energy, p.forces, p.dipole):
                if not torch.equal(raw[name + "_prediction"], ref_raw[name + "_prediction"]):
                    raise ValueError(f"Proposal changed physical {name} predictions")
            metrics["untrained_proposal"] = audit_proposal(ref_problems, ref_guesses, ref_rows)
        report["splits"][split] = metrics
        torch.save(raw, output / f"{split}_predictions.pt")
        with (output / f"{split}_curves.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    temporary = output / "metrics.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "metrics.json")
    print(json.dumps({key: report[key] for key in ("checkpoint", "variant", "stage", "splits")}, indent=2))


if __name__ == "__main__":
    main()
