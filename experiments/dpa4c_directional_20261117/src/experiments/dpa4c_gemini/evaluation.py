"""Paired one-pass and uniform-charge evaluation at fixed total charge and zero field."""

import csv
import gzip
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from ase.units import Debye
from torch.utils._pytree import tree_map

from curator.data import properties as p
from curator.variational.functionals import FunctionalProblem, solve_functional_reference
from curator.variational.nonlinear import NewtonKKTOptions
from equilibrium import _forward, _read_only, model_inputs


def load_sasa_dipoles(path=None):
    """Held-out released ORCA Cartesian moments in Debye, indexed by prepared source ID."""
    path = Path(path or Path(__file__).with_name("sasa_dft_dipoles.csv.gz"))
    with gzip.open(path, "rt") as stream:
        rows = list(csv.DictReader(stream))
    values = {int(r["source_index"]): [float(r[f"DFT_{s}_D"]) for s in "xyz"] for r in rows}
    if len(values) != len(rows):
        raise ValueError("Duplicated prepared-frame ID in the dipole table")
    return values


def _pure(model, batch, device):
    data = model_inputs(batch, device)
    for module in model.input_modules:
        data = module(data)
    data = model.representation(data)
    energy = data[p.energy].reshape(())
    force, = torch.autograd.grad(-energy, data[p.positions])
    return dict(energy=energy.item(), forces=force.detach().double().cpu())


def _errors(output, ref_e, ref_f):
    return dict(energy_error_meV_atom=(output["energy"] - ref_e) * 1000 / len(ref_f),
                force_sse_meV2_A2=float(((output["forces"] - ref_f) * 1000).square().sum()))


def _oracle(head, context, qhat_cpu, native_residual):
    matrix, total = context["constraint_matrix"], context["constraint_target"]
    q = qhat_cpu.to(matrix)
    torch.testing.assert_close(matrix, torch.ones_like(q)[None], rtol=0, atol=0)
    # Uniform counterfactual already verifies zero field and a single fixed-Q equality.
    def potential(z):
        return head.internal_energy(context, z[:, None]).sum()

    problem = FunctionalProblem(potential, len(q), matrix, total)
    q0 = (total.reshape(()) / len(q)).expand_as(q)
    drive = problem.stationarity(q0).stationarity
    recomputed = problem.stationarity(q).stationarity
    torch.testing.assert_close(recomputed, native_residual.to(recomputed), atol=1e-9, rtol=1e-8)
    answer = solve_functional_reference(problem, q, NewtonKKTOptions(
        stationarity_atol=1e-11, stationarity_rtol=0., feasibility_atol=1e-12, feasibility_rtol=0.))
    star = answer.state.detach()
    certificate = problem.stationarity(star)
    residual = certificate.stationarity.abs().max().item()
    violation = certificate.feasibility.abs().max().item()
    accepted = bool(answer.converged and answer.stable and residual <= 1e-10 and violation <= 1e-10)
    result = dict(accepted=accepted, status=answer.status, iterations=int(answer.iterations),
                  max_stationarity_V=residual, feasibility_e=violation,
                  drive_sse_V2=drive.square().sum().item())
    if accepted:
        error = (q - star).square().sum().item()
        transfer = (star - q0).square().sum().item()
        result.update(charge_error_sse_e2=error, transfer_sse_e2=transfer,
                      relative_transfer_error_percent=100 * (error / transfer)**.5
                      if transfer > len(q)*1e-16 else None)
    return result


def _aggregate(rows, pure, want_dipoles):
    n = sum(r["atoms"] for r in rows)
    results = dict(frames=len(rows), atoms=n)
    for tag in (("pure",) if pure else ("proposal", "uniform")):
        results[tag] = dict(
            energy_RMSE_meV_atom=float(np.mean([r[tag]["energy_error_meV_atom"]**2 for r in rows])**.5),
            force_RMSE_meV_A=(sum(r[tag]["force_sse_meV2_A2"] for r in rows) / (3*n))**.5)
    if pure:
        return results
    results["clamp_minus_proposal"] = {
        key: results["uniform"][key] - x for key, x in results["proposal"].items()}
    results.update(stationarity_RMS_V=(sum(r["residual_sse_V2"] for r in rows)/n)**.5,
                   proposal_transfer_RMS_e=(sum(r["proposal_transfer_sse_e2"] for r in rows)/n)**.5,
                   clamp_change_in_predicted_force_RMS_meV_A=(
                       sum(r["clamp_force_change_sse_meV2_A2"] for r in rows)/(3*n))**.5,
                   max_proposal_charge_violation_e=max(r["constraint_max_e"] for r in rows))
    subset = [r for r in rows if "newton" in r]
    if subset:
        valid = [r for r in subset if r["newton"]["accepted"]]
        atoms = sum(r["atoms"] for r in valid)
        ratios = [r["newton"]["relative_transfer_error_percent"] for r in valid
                  if r["newton"]["relative_transfer_error_percent"] is not None]
        results["newton_panel"] = dict(
            requested=len(subset), certified=len(valid), relative_valid=len(ratios),
            relative_denominator_floor_RMS_e=1e-8,
            uniform_drive_RMS_V=(sum(r["newton"]["drive_sse_V2"] for r in subset)
                                 / sum(r["atoms"] for r in subset))**.5,
            charge_RMSE_e=(sum(r["newton"]["charge_error_sse_e2"] for r in valid)/atoms)**.5 if atoms else None,
            equilibrium_transfer_RMS_e=(sum(r["newton"]["transfer_sse_e2"] for r in valid)/atoms)**.5
            if atoms else None,
            mean_frame_relative_transfer_error_percent=float(np.mean(ratios)) if ratios else None,
            p95_frame_relative_transfer_error_percent=float(np.percentile(ratios,95)) if ratios else None)
    if want_dipoles:
        measured = [r for r in rows if "dipole_D" in r]
        results["dipole"] = dict(frames=len(measured), requested=len(rows))
        if measured:
            pred = np.asarray([r["dipole_D"]["proposal"] for r in measured])
            ref = np.asarray([r["dipole_D"]["DFT"] for r in measured])
            results["dipole"].update(
                signed_component_pearson_r=float(np.corrcoef(pred.ravel(), ref.ravel())[0, 1])
                if pred.std() and ref.std() else None,
                component_RMSE_D=float(np.mean((pred-ref)**2)**.5),
                mean_proposal_magnitude_D=float(np.linalg.norm(pred, axis=1).mean()),
                mean_DFT_magnitude_D=float(np.linalg.norm(ref, axis=1).mean()))
    return results


def evaluate(model, batches, *, pure=False, newton_ids=(), dipoles_D=None, device=None):
    """Score an iterable of (original source ID, collated single-geometry batch).

    Pass the DPA4C+ACVR model with its normal variational output, including for the
    pure control (only its representation is called). Use the FP64 reconstruction
    at identical weights for Newton. Dipole references are never forwarded.
    """
    parameter = next(model.parameters())
    device = parameter.device if device is None else device
    panel = set(map(int, newton_ids)) if not pure else set()
    if panel and parameter.dtype != torch.float64:
        raise ValueError("Offline Newton requires FP64 model weights")
    rows, seen = [], set()
    with _read_only(model, device) as head:
        for idx, batch in batches:
            idx = int(idx)
            if idx in seen or batch[p.n_atoms].numel() != 1:
                raise ValueError("Provide each ID once, one geometry per batch")
            seen.add(idx)
            # Preserve full-precision references when a caller evaluates an FP32 model.
            labels = batch["variational"]["labels"]
            e, f = labels["condition_energy"], labels["condition_forces"]
            if e["unit"] != "eV" or f["unit"] != "eV/Angstrom" or e["values"].numel() != 1:
                raise ValueError("Missing single-structure E/F targets in physical units")
            if not all(bool((x["present"] & x["eligible"]).all()) for x in (e, f)):
                raise ValueError("An E/F target is missing or ineligible")
            truth = (e["values"].item(), f["values"].detach().double().cpu().reshape(-1, 3))
            n = int(batch[p.n_atoms].item())
            if len(truth[1]) != n:
                raise ValueError("Reference force count mismatch")
            batch = tree_map(lambda t: t.to(dtype=parameter.dtype)
                             if torch.is_tensor(t) and t.is_floating_point() else t, batch)
            row = dict(source_index=idx, atoms=n)
            if pure:
                row["pure"] = _errors(_pure(model, batch, device), *truth)
            else:
                contexts = []
                calls = []
                with ExitStack() as hooks:
                    if idx in panel:
                        hooks.enter_context(head.internal_energy.register_forward_pre_hook(
                            lambda _, args: contexts.append(args[0])))
                    if not rows:
                        hooks.enter_context(head.proposal.register_forward_hook(
                            lambda *_: calls.append(1)))
                    output = _forward(model, head, batch, device, "proposal")
                if not rows and len(calls) != 1:
                    raise RuntimeError(f"Expected one native proposal evaluation, observed {len(calls)}")
                clamped = _forward(model, head, batch, device, "uniform")
                q = output["q"]
                total = batch[p.total_charge].item()
                row.update(proposal=_errors(output, *truth), uniform=_errors(clamped, *truth),
                           proposal_transfer_sse_e2=float((q-total/n).square().sum()),
                           residual_sse_V2=float(output["stationarity"].square().sum()),
                           constraint_max_e=output["constraint_max"],
                           clamp_force_change_sse_meV2_A2=float(
                               ((clamped["forces"]-output["forces"])*1000).square().sum()))
                if idx in panel:
                    if len(contexts) != 1:
                        raise ValueError("Newton requires a single native proposal/energy context")
                    row["newton"] = _oracle(head, contexts.pop(), q, output["stationarity"])
                if dipoles_D is not None and idx in dipoles_D:
                    if abs(total) > 1e-8 or batch[p.pbc].any():
                        raise ValueError("SA-SA dipoles require neutral isolated structures")
                    xyz = batch[p.positions].detach().cpu().double()
                    pred = q @ (xyz - xyz.mean(dim=0)) / Debye
                    row["dipole_D"] = {"proposal": pred.tolist(), "DFT": list(dipoles_D[idx])}
            rows.append(row)
    if not rows or not panel <= seen:
        raise ValueError("Missing evaluation frames or requested Newton panel")
    report = dict(summary=_aggregate(rows, pure, dipoles_D is not None), frames=rows)
    json.dumps(report, allow_nan=False)
    return report
