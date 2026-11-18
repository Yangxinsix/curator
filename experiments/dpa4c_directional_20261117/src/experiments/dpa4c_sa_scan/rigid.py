"""Frozen-fragment SA pull-apart predictions; only the original frame has labels."""
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import torch
from ase.io.trajectory import Trajectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dpa4c_tensor"))
from train_runner import _forward, _read_only, inputs, load_tensor_checkpoint, primary_paths, sha
from curator.data import properties as p
from curator.data.variational_data import canonical_boundary_hash
from curator.variational.variables.charge_boundary import fixed_total_charge_recipe


def describe(at):
    z, xyz, m = at.numbers, at.positions, at.get_masses()
    if sorted(z[:7]) != [1, 1, 8, 8, 8, 8, 16] or at.pbc.any():
        raise ValueError("Not the expected fixed fragment ordering / boundary")
    c, d = [np.average(x, weights=w, axis=0) for x, w in
            [(xyz[:7], m[:7]), (xyz[7:], m[7:])]]
    axis = d - c
    # For rigid pulls exclude protons transferred to, or shared with, the other fragment.
    for i in np.where(z == 1)[0]:
        n = np.where(z != 1)[0]
        dist = np.linalg.norm(xyz[n] - xyz[i], axis=1)
        same = (n < 7) == (i < 7)
        if dist[same].min() > 1.30 or dist[~same].min() - dist[same].min() < .20:
            return None
    return np.linalg.norm(axis), axis / np.linalg.norm(axis)


def data(at, xyz, charge):
    n = len(at)
    spec = asdict(fixed_total_charge_recipe(charge))
    bc = dict(condition_geometry_index=torch.tensor([0]), boundary_spec=[spec],
              boundary_hash=[canonical_boundary_hash(spec)], condition_state_ptr=torch.tensor([0, n]),
              condition_ptr=torch.tensor([0, 1]), condition_weight=torch.ones(1, dtype=torch.float64))
    return {p.positions: torch.from_numpy(xyz.copy()), p.atomic_numbers: torch.as_tensor(at.numbers),
            p.image_idx: torch.zeros(n, dtype=torch.long), p.n_atoms: torch.tensor([n]),
            p.cell: torch.as_tensor(at.cell.array), p.pbc: torch.zeros((1, 3), dtype=torch.bool),
            p.total_charge: torch.tensor([charge], dtype=torch.float64), "variational": bc}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    for sysname in ["sasa", "sabs", "sadma"]:
        trajectory, split, ref = (primary_paths if sysname == "sasa" else inputs)(sysname)
        meta = json.loads(ref.read_text())
        assert sha(trajectory) == meta["source_sha256"] and sha(split) == meta["split_sha256"]
        ids = json.loads(split.read_text())["test"]
        with Trajectory(str(trajectory)) as traj:
            candidates = []
            for i in sorted(ids):
                desc = describe(traj[i])
                if desc is not None:
                    candidates.append((float(desc[0]), i))
            candidates.sort()
            if len(candidates) < 4:
                raise ValueError("Too few intact monomer pairs")
            chosen = [candidates[round(k*(len(candidates)-1))][1] for k in [.25, .75]]
            frames = [(i, traj[i]) for i in chosen]
        rows = []
        digests = {}
        for arm in ("old", "orient2"):
            f = a.checkpoints / f"{sysname}-{arm}.pt"
            digests[arm] = sha(f)
            ck = torch.load(f, weights_only=False, map_location="cpu")
            assert (ck["identity"]["system"], ck["identity"]["arm"], ck["joint_step"], ck["identity"]["split_sha256"]) == (sysname, arm, 40000, sha(split))
            model = load_tensor_checkpoint(f)
            with _read_only(model, "cuda") as head:
                for idx, at in frames:
                    charge = float(at.info["total_charge"])
                    sep, axis = describe(at)
                    for mode in ("proposal", "uniform"):
                        curve = []
                        for delta in np.linspace(0, 8, 33):
                            xyz = at.positions.copy()
                            xyz[7:] += delta * axis
                            pred = _forward(model, head, data(at, xyz, charge), "cuda", mode)
                            force = pred["forces"].numpy()
                            curve.append(dict(system=sysname, frame=idx, arm=arm, charge_mode=mode,
                                R_A=float(sep+delta), delta_R_A=float(delta), E_eV=pred["energy"],
                                radial_force_eV_A=float(force[7:].sum(axis=0) @ axis),
                                radial_ref_eV_A=float(at.get_forces()[7:].sum(axis=0) @ axis) if delta == 0 else None,
                                charge_B_e=(charge*(len(at)-7)/len(at) if mode=="uniform"
                                            else float(pred["q"][7:].sum()))))
                        for row in curve:
                            row["E_minus_last_eV"] = row["E_eV"] - curve[-1]["E_eV"]
                        rows.extend(curve)
        with (a.out/f"{sysname}.csv").open("w", newline="") as fp:
            w = csv.DictWriter(fp, rows[0].keys()); w.writeheader(); w.writerows(rows)
        (a.out/f"{sysname}.json").write_text(json.dumps(dict(source_sha256=sha(trajectory),
            split_sha256=sha(split), checkpoint_sha256=digests, test_count=len(ids),
            eligible_for_rigid_pull=len(candidates), test_frame_ids=chosen,
            convention="E referenced to the same two fragments at +8 A, not isolated monomers. Force is net on B along A to B. Only delta=0 has a reference label."), indent=2))
        print(sysname, len(rows), chosen, flush=True)


if __name__ == "__main__":
    main()
