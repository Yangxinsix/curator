"""Held-out C9H4 rigid rotor scan: one-pass proposal and zero-charge counterfactual.

Forces are obtained by differentiating each energy with its own charge rule.
No Newton and no fitting. Energies are aligned only by subtracting frame 0 from
the same curve: a train-composition reference is therefore unnecessary.
"""
import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np


def sha(path):
    with open(path, "rb") as fd:
        return hashlib.file_digest(fd, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(fd.read()).hexdigest()


def angle_tangent(z, positions):
    """Return signed angle (deg) and dx/dtheta for the right CH2 (Å/rad).

    Released atomic order: orientation is defined using the two end carbons
    in increasing atom-index order; at each carbon identify its two bonded H.
    Right-handed rotation about left->right increases theta. Reversing labels
    of two identical H changes the angle by 180 degrees, not the physics.
    """
    c, h = np.flatnonzero(z == 6), np.flatnonzero(z == 1)
    if len(z) != 13 or len(c) != 9 or len(h) != 4:
        raise ValueError("Expected the nine-carbon cumulene scan")
    cc_dist = np.linalg.norm(positions[c, None] - positions[c][None], axis=2)
    end_indices = np.unravel_index(np.argmax(cc_dist), cc_dist.shape)
    ends = np.sort(c[np.array(end_indices)])
    # Both are found by connectivity; atom order fixes only an angle convention.
    left, right = ends
    hleft = h[np.linalg.norm(positions[h]-positions[left], axis=1) < 1.3]
    hright = h[np.linalg.norm(positions[h]-positions[right], axis=1) < 1.3]
    if len(hleft) != 2 or len(hright) != 2 or len(set(hleft) | set(hright)) != 4:
        raise ValueError("Could not identify disjoint terminal CH2 groups")
    axis = positions[right] - positions[left]
    axis /= np.linalg.norm(axis)
    l = positions[hleft[1]] - positions[hleft[0]]
    r = positions[hright[0]] - positions[hright[1]]
    l -= (l @ axis)*axis
    r -= (r @ axis)*axis
    angle = np.degrees(np.arctan2(np.cross(l, r) @ axis, l @ r)) % 360
    tangent = np.zeros_like(positions)
    tangent[hright] = np.cross(axis, positions[hright] - positions[right])
    return float(angle), tangent


def boundaries():
    from curator.data.variational_data import canonical_boundary_hash
    from curator.variational.variables.charge_boundary import fixed_total_charge_recipe
    import torch

    spec = asdict(fixed_total_charge_recipe(0.0))
    return dict(condition_geometry_index=torch.tensor([0]), boundary_spec=[spec],
                boundary_hash=[canonical_boundary_hash(spec)], condition_state_ptr=torch.tensor([0, 13]),
                condition_ptr=torch.tensor([0, 1]), condition_weight=torch.ones(1, dtype=torch.float64))


def score_scan(model, *, trajectory, out_dir, checkpoint_path, device="cuda"):
    import torch
    from ase.io.trajectory import Trajectory
    from curator.data import properties as p
    from equilibrium import _read_only, _forward

    trajectory, out_dir, checkpoint_path = map(Path, (trajectory, out_dir, checkpoint_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    param = next(model.parameters())
    if param.dtype != torch.float64:
        raise ValueError("Use the FP64 reconstruction of the trained weights for the scan.")
    scan_sha, ck_sha = sha(trajectory), sha(checkpoint_path)
    common = dict(schema="cumulene-rigid-torque-v1", scan_sha256=scan_sha, checkpoint_sha256=ck_sha)
    report_path = out_dir / "scan.json"
    if report_path.exists() and all((out_dir / f).exists() for f in ("scan.csv", "scan.png")):
        prev = json.loads(report_path.read_text())
        if all(prev.get(k) == v for k, v in common.items()) and prev.get("frames") == 1000:
            return prev

    bc = boundaries()
    rows, xyzs, dxdtheta = [], [], []
    with Trajectory(str(trajectory)) as src, _read_only(model, device) as head:
        if len(src) != 1000:
            raise ValueError("Expected all 1,000 released scans")
        for i, at in enumerate(src):
            if at.pbc.any() or at.info.get("total_charge") != 0 or int(at.info.get("source_frame", i)) != i:
                raise ValueError("Wrong boundary or source order in scan")
            xyz = np.array(at.positions, dtype=float)
            angle, tangent = angle_tangent(at.numbers, xyz)
            if not i:
                species = at.numbers.copy()
            elif not np.array_equal(species, at.numbers):
                raise ValueError("Unstable atom indexing across released scan")
            target = at.get_forces(apply_constraint=False)
            row = dict(index=i, angle_deg=angle, dft_energy_eV=float(at.get_potential_energy()),
                       dft_torque_eV_rad=float(np.sum(tangent*target)))
            xyzs.append(xyz)
            dxdtheta.append(tangent)
            # No reference energy, force or charge labels enter the model.
            inputs = {p.positions: torch.as_tensor(xyz),
                      p.atomic_numbers: torch.as_tensor(at.numbers, dtype=torch.long),
                      p.image_idx: torch.zeros(len(at), dtype=torch.long), p.n_atoms: torch.tensor([len(at)]),
                      p.cell: torch.as_tensor(at.cell.array), p.pbc: torch.zeros((1, 3), dtype=torch.bool),
                      p.total_charge: torch.zeros(1, dtype=param.dtype), "variational": bc}
            for tag, mode in (("proposal", "proposal"), ("q0", "uniform")):
                result = _forward(model, head, inputs, device, mode)
                force = result["forces"].numpy()
                row.update({tag + "_energy_eV": result["energy"],
                            tag + "_torque_eV_rad": float(np.sum(tangent*force)),
                            tag + "_force_mse_eV2_A2": float(np.square(force-target).mean())})
                if mode == "proposal":
                    row["proposal_charge_RMS_e"] = float(result["q"].square().mean().sqrt())
                    row["stationarity_RMS_V"] = float(result["stationarity"].square().mean().sqrt())
                    if abs(float(result["q"].sum())) > 1e-9:
                        raise ValueError("Proposal violated scan neutrality")
            if not np.isfinite(list(row.values())).all():
                raise ValueError(f"Nonfinite source/prediction at source frame {i}")
            rows.append(row)
            if (i+1) % 250 == 0:
                print(f"torsion scan {i+1}/{len(src)}", flush=True)

    theta = np.radians([r["angle_deg"] for r in rows])
    if not (np.all(np.diff(theta) > 0) and
            np.allclose(theta[[0, -1]], [np.pi/2, np.pi], atol=1e-5, rtol=0)):
        raise ValueError("Expected monotonically increasing 90–180 degree scan")
    # Check dx/dtheta actually follows the path; the formula is invalid if
    # other atoms relax, translate or rotate between scan frames.
    tangent_err = abs(np.gradient(xyzs, theta, axis=0, edge_order=2)-np.array(dxdtheta)).max()
    if tangent_err > 2e-3:
        raise ValueError(f"Non-rigid trajectory or incorrect torsional tangent: {tangent_err}")
    curves = {}
    for tag in ("dft", "proposal", "q0"):
        e = np.array([r[tag+"_energy_eV"] for r in rows])
        t = np.array([r[tag+"_torque_eV_rad"] for r in rows])
        rel = e-e[0]
        for r, x in zip(rows, rel):
            r[tag+"_relative_eV"] = x
        curves[tag] = dict(endpoint_barrier_eV=float(rel[-1]), max_minus_min_eV=float(np.ptp(e)),
                           torque_RMS_eV_rad=float(np.sqrt(np.mean(t*t))),
                           energy_force_consistency_RMSE_eV_rad=float(np.mean(
                               (np.gradient(rel, theta, edge_order=2)[1:-1]+t[1:-1])**2)**.5))
        if tag == "dft":
            ref_e, ref_t = rel, t
        else:
            curves[tag].update(relative_energy_RMSE_eV=float(np.mean((rel-ref_e)**2)**.5),
                endpoint_barrier_error_eV=float(rel[-1] - ref_e[-1]),
                torque_RMSE_eV_rad=float(np.mean((t-ref_t)**2)**.5),
                cartesian_force_RMSE_meV_A=1000*float(np.mean([r[tag+"_force_mse_eV2_A2"] for r in rows])**.5))
    summary = dict(**common, frames=len(rows), scan=str(trajectory), checkpoint=str(checkpoint_path),
                   max_tangent_error_A_per_rad=float(tangent_err), curves=curves,
                   note="Both model curves and DFT are anchored at the same 90-degree geometry. "
                        "q0 clamps the native energy functional at q=0. "
                        "Positive torque = F·dR/dtheta for right CH2 rotation; displayed E in eV per molecule.",
                   charge_RMS_e=float(np.mean([r["proposal_charge_RMS_e"]**2 for r in rows])**.5),
                   stationarity_RMS_V=float(np.mean([r["stationarity_RMS_V"]**2 for r in rows])**.5))
    with (out_dir/"scan.csv").open("w", newline="") as stream:
        csvout = csv.DictWriter(stream, fieldnames=rows[0])
        csvout.writeheader()
        csvout.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, constrained_layout=True)
    for tag, label, line, color in (("dft", "DFT", "-", "#151515"), ("q0", "Fixed q = 0", "--", "#cb6130"),
                                    ("proposal", "One-pass proposal", "-.", "#236aa1")):
        for ax, key in zip(axes, ("_relative_eV", "_torque_eV_rad")):
            ax.plot(np.degrees(theta), [r[tag+key] for r in rows], line, color=color, label=label)
    axes[0].set_ylabel("Energy relative to 90° (eV)")
    axes[1].set_ylabel("Torsional torque (eV/rad)")
    axes[1].set_xlabel("Terminal CH₂ angle (degrees)")
    axes[0].legend(frameon=False)
    fig.savefig(out_dir/"scan.png", dpi=155)
    plt.close(fig)
    pending = out_dir/"scan.json.tmp"
    pending.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    pending.replace(report_path)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--original-endpoint", action="store_true", help="Unmodified DPA4C experiment's Cumulene B2 final checkpoint")
    options = parser.parse_args()
    import torch
    torch.set_default_dtype(torch.float64)
    if options.original_endpoint:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"dpa4c_gemini"))
        from extension_run import build
        state = torch.load(options.checkpoint, map_location="cpu", weights_only=False)
        expected = dict(system="cumulene", arm="acvr", seed=3407, winner="B2")
        if any(state["identity"].get(k) != v for k, v in expected.items()) or state["joint_step"] != 20000:
            raise ValueError("Not the released 20k Cumulene ACVR continuation source!")
        model = build("cumulene", ["C", "H"], "acvr", 3407, "B2").double().cuda()
        model.load_state_dict(state["model"])
    else:
        from train_runner import load_tensor_checkpoint
        model = load_tensor_checkpoint(options.checkpoint, device="cuda", dtype=torch.float64)
    print(json.dumps(score_scan(model, trajectory=options.trajectory, checkpoint_path=options.checkpoint,
                                out_dir=options.out), indent=2))
