"""Independently check learned ACVR oracle checkpoints; never runs training."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from curator.data.variational_oracle import oracle_features
from curator.evaluate.variational import VariationalEvaluator
from curator.model.variational_oracle import LearnedVariationalOracle
from curator.train import VariationalJointLoss
from curator.variational import project_to_equalities, solve_amortized, solve_dense


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    context = OmegaConf.create({"model": checkpoint["model_params"], "data": checkpoint["data_params"]})
    # Rebuild from config/state even when a pickled full model also exists.
    # This deliberately exposes incomplete configuration snapshots.
    model = instantiate(context.model, _convert_="all")
    model.double().eval()
    if not isinstance(model.representation, LearnedVariationalOracle):
        raise TypeError("checkpoint must contain LearnedVariationalOracle in NeuralNetworkPotential")
    data = instantiate(context.data)
    data.setup()
    if "VariationalOracleDataModule" in checkpoint:
        data.load_state_dict(checkpoint["VariationalOracleDataModule"])
    model.initialize_modules(data)
    model.load_state_dict({key.removeprefix("model."): value
                           for key, value in checkpoint["state_dict"].items()
                           if key.startswith("model.")}, strict=True)
    return checkpoint, model, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", required=True, type=Path, help="Physical-stage checkpoint")
    parser.add_argument("--proposal", required=True, type=Path, help="Proposal-stage checkpoint")
    parser.add_argument("--output", required=True, type=Path, help="Directory for JSON and evaluation records")
    args = parser.parse_args()
    torch.set_num_threads(1)
    paths = {"physical": args.physical.resolve(strict=True), "proposal": args.proposal.resolve(strict=True)}
    physical_checkpoint, physical, data = load_checkpoint(paths["physical"])
    proposal_checkpoint, trained, proposal_data = load_checkpoint(paths["proposal"])
    if physical.representation.stage != "physical" or trained.representation.stage != "proposal":
        raise ValueError("checkpoints must identify physical and proposal stages respectively")
    if data.metadata != proposal_data.metadata:
        raise ValueError("physical/proposal checkpoint dataset metadata or geometry splits differ")
    before = {name: physical_checkpoint["state_dict"]["model.representation." + name]
              for name, _ in physical.representation.named_parameters()}
    after = {name: proposal_checkpoint["state_dict"]["model.representation." + name]
             for name, _ in trained.representation.named_parameters()}
    if before.keys() != after.keys():
        raise ValueError("checkpoint parameter names differ")
    changed = [name for name in before if before[name].dtype != after[name].dtype
               or not torch.equal(before[name], after[name])]
    if any(name.startswith("physical.") for name in changed):
        raise ValueError("proposal training changed physical-functional parameters")
    if not any(name.startswith("proposal.") for name in changed):
        raise ValueError("proposal parameters did not change between checkpoints")
    initialized = LearnedVariationalOracle(seed=physical.representation.seed,
                                          hidden_features=physical.representation.hidden_features)
    if any(not torch.equal(value, initialized.proposal.state_dict()[key])
           for key, value in physical.representation.proposal.state_dict().items()):
        raise ValueError("the physical checkpoint's baseline proposal differs from its seeded initialization")
    loss_functions = [output.loss_fn for output in physical_checkpoint["outputs"]
                      if isinstance(getattr(output, "loss_fn", None), VariationalJointLoss)]
    if len(loss_functions) != 1:
        raise ValueError("physical checkpoint must contain exactly one VariationalJointLoss")
    physical_metrics = VariationalEvaluator(
        physical, loss_functions[0], output_dir=args.output / "physical_holdout", device="cpu",
    ).evaluate(data.test_dataloader())
    tolerances = (1e-2, 1e-3, 1e-4, 1e-8)
    stats = {name: {"state_errors": [], "residuals": [], "tolerances": {
        str(tolerance): {"accepted": 0, "corrections": 0, "fallbacks": 0, "unreliable": 0,
                         "state_errors": []} for tolerance in tolerances}}
             for name in ("untrained", "trained")}
    geometry_uids, condition_uids = [], []
    with torch.no_grad():
        for batch in data.test_dataloader():
            geometry_uids.extend(batch["variational"]["geometry_uid"])
            condition_uids.extend(batch["variational"]["condition_uid"])
            # Both proposals use exactly the same learned physical functional.
            problems, fields = physical.representation.problems(batch)
            features = torch.cat((oracle_features(batch["positions"].reshape(-1, 2, 3))[
                batch["variational"]["condition_geometry_index"]], fields), dim=-1)
            proposals = {"untrained": physical.representation.proposal(features),
                         "trained": trained.representation.proposal(features)}
            for index, problem in enumerate(problems):
                exact = solve_dense(problem)
                if not exact.reliable:
                    raise RuntimeError("exact reference solution failed certification")
                for name, values in proposals.items():
                    projected = project_to_equalities(values[index], problem.constraint_matrix,
                                                      problem.constraint_target).state
                    stats[name]["state_errors"].append(projected - exact.state)
                    for tolerance in tolerances:
                        solved = solve_amortized(problem, values[index], residual_tolerance=tolerance,
                                                 max_correction_steps=1, fallback=True)
                        entry = stats[name]["tolerances"][str(tolerance)]
                        entry["accepted"] += int(solved.status == "accepted_projected")
                        entry["corrections"] += solved.iterations
                        entry["fallbacks"] += int(solved.used_fallback)
                        entry["unreliable"] += int(not solved.reliable)
                        entry["state_errors"].append(solved.state - exact.state)
                        if tolerance == tolerances[0]:
                            stats[name]["residuals"].append(solved.projected_residuals.normalized_stationarity_norm)
    for result in stats.values():
        residuals = torch.stack(result.pop("residuals"))
        count = residuals.numel()
        result["num_conditions"] = count
        result["projected_state_rmse"] = float(torch.cat(result.pop("state_errors")).square().mean().sqrt())
        result["normalized_stationarity"] = {
            name: float(torch.quantile(residuals, quantile))
            for name, quantile in (("p50", .50), ("p95", .95), ("p99", .99), ("max", 1.0))}
        for entry in result["tolerances"].values():
            entry["zero_correction_acceptance_rate"] = entry.pop("accepted") / count
            entry["mean_correction_steps"] = entry.pop("corrections") / count
            entry["fallback_rate"] = entry.pop("fallbacks") / count
            entry["unreliable_rate"] = entry.pop("unreliable") / count
            entry["returned_state_rmse"] = float(torch.cat(entry.pop("state_errors")).square().mean().sqrt())
    report = {
        "format_version": 1,
        "checkpoints": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for name, path in paths.items()},
        "epochs": {"physical": physical_checkpoint["epoch"], "proposal": proposal_checkpoint["epoch"]},
        "checks": {"physical_parameters_identical": True, "baseline_matches_initialization": True,
                   "changed_proposal_parameters": changed, "identical_dataset_metadata": True},
        "dataset_metadata": data.metadata, "model_metadata": physical.representation.get_extra_state(),
        "test_geometry_uids": geometry_uids, "test_condition_uids": condition_uids,
        "physical_holdout": physical_metrics, "proposal_comparison": stats,
        "method": {
            "physical_metrics": "VariationalEvaluator using the physical checkpoint's VariationalJointLoss",
            "forces": "coordinate derivatives of the learned minimized field-Legendre energy via implicit KKT",
            "response": "full dx/dfield from linearized KKT; labels are the independent analytic Hessian inverse",
            "proposal_teacher": "solve_dense of the same learned physical functional for both proposals",
            "aggregation": "equal-size geometry groups; state RMSE over all internal components; residual quantiles over conditions",
            "correction": "solve_amortized with max_correction_steps=1 and fallback=True",
            "scope": "held-out geometries and control amplitudes inside the training range; same conjugate-field boundary type",
            "timing": "not measured; acceptance/correction counts do not establish wall-clock speedup",
        },
    }
    output = args.output / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
