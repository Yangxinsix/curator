#!/usr/bin/env python
"""Generate supervised atomic charge labels from a fixed-parameter QEq solve.

The output trajectory stores the solved charges in ASE initial_charges, which
Curator already reads as the ``atomic_charge`` training target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch
from ase.data import atomic_numbers, chemical_symbols
from ase.io import read, write

from curator.data import AseDataReader, MatScipyNeighborList, properties
from curator.layer import EwaldSummation


DEFAULT_ELECTRONEGATIVITY = {
    "H": 4.528,
    "B": 4.07,
    "C": 5.343,
    "N": 6.899,
    "O": 8.741,
    "Na": 2.76,
    "Cl": 8.8,
}

DEFAULT_HARDNESS = {
    "H": 13.89,
    "B": 8.87,
    "C": 10.126,
    "N": 11.308,
    "O": 13.364,
    "Na": 6.43,
    "Cl": 10.35,
}


def _load_mapping(value: Optional[str], default: Mapping[str, float]) -> Dict[str, float]:
    if value is None:
        return dict(default)
    path = Path(value)
    if path.exists():
        with path.open() as handle:
            loaded = json.load(handle)
        return {str(k): float(v) for k, v in loaded.items()}
    loaded = json.loads(value)
    return {str(k): float(v) for k, v in loaded.items()}


def _to_z_mapping(values: Mapping[str, float]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for key, value in values.items():
        try:
            z = int(key)
        except ValueError:
            z = atomic_numbers[key]
        out[z] = float(value)
    return out


def _lookup_by_z(z_values: Iterable[int], values: Mapping[int, float], name: str) -> np.ndarray:
    missing = sorted({int(z) for z in z_values if int(z) not in values})
    if missing:
        symbols = [chemical_symbols[z] for z in missing]
        raise KeyError(f"Missing {name} values for atomic numbers {missing} / symbols {symbols}")
    return np.asarray([values[int(z)] for z in z_values], dtype=np.float64)


def _batch_to_device(batch: properties.Type, device: torch.device) -> properties.Type:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _solve_constrained_qeq(
    chi: np.ndarray,
    hardness: np.ndarray,
    kernel: np.ndarray,
    total_charge: float,
    ridge: float,
) -> np.ndarray:
    n_atoms = chi.shape[0]
    hessian = kernel + 2.0 * np.diag(hardness)
    if ridge > 0.0:
        hessian = hessian + ridge * np.eye(n_atoms)

    lhs = np.zeros((n_atoms + 1, n_atoms + 1), dtype=np.float64)
    rhs = np.zeros(n_atoms + 1, dtype=np.float64)
    lhs[:n_atoms, :n_atoms] = hessian
    lhs[:n_atoms, n_atoms] = 1.0
    lhs[n_atoms, :n_atoms] = 1.0
    rhs[:n_atoms] = -chi
    rhs[n_atoms] = total_charge
    solution = np.linalg.solve(lhs, rhs)
    return solution[:n_atoms]


def solve_atoms_qeq(
    atoms,
    electronegativity: Mapping[int, float],
    hardness: Mapping[int, float],
    reader: AseDataReader,
    ewald: Optional[EwaldSummation],
    device: torch.device,
    total_charge: Optional[float],
    ridge: float,
) -> np.ndarray:
    z = atoms.get_atomic_numbers()
    chi = _lookup_by_z(z, electronegativity, "electronegativity")
    eta = _lookup_by_z(z, hardness, "hardness")

    if ewald is None:
        kernel = np.zeros((len(atoms), len(atoms)), dtype=np.float64)
    else:
        batch = _batch_to_device(reader(atoms), device)
        kernel_tensors = ewald.get_ewald_kernel(
            batch[properties.cell],
            batch[properties.n_atoms],
            batch[properties.positions],
            batch[properties.edge_dist],
            batch[properties.edge_idx],
        )
        if len(kernel_tensors) != 1:
            raise ValueError("generate_qeq_charge_labels.py expects one structure at a time")
        kernel = kernel_tensors[0].detach().cpu().numpy().astype(np.float64)

    q_total = float(atoms.info.get("total_charge", 0.0) if total_charge is None else total_charge)
    q = _solve_constrained_qeq(chi, eta, kernel, q_total, ridge)
    # Remove tiny numerical drift from the equality constraint.
    q += (q_total - float(q.sum())) / len(q)
    return q


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input ASE trajectory/database")
    parser.add_argument("output", type=Path, help="Output trajectory with QEq charges in initial_charges")
    parser.add_argument("--index", default=":", help="ASE index expression, default ':'")
    parser.add_argument("--electronegativity", help="JSON string or JSON file. Default: train-12 values")
    parser.add_argument("--hardness", help="JSON string or JSON file. Default: train-12 values")
    parser.add_argument("--total-charge", type=float, default=None, help="Override total charge for every structure")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Neighbor cutoff used for real-space Ewald")
    parser.add_argument("--k-cutoff", type=float, default=None, help="Ewald reciprocal cutoff")
    parser.add_argument("--alpha", type=float, default=0.4, help="Ewald alpha")
    parser.add_argument("--acc-factor", type=float, default=12.0, help="Ewald accuracy factor")
    parser.add_argument("--no-ewald", action="store_true", help="Solve isolated diagonal QEq without Coulomb/Ewald kernel")
    parser.add_argument("--ridge", type=float, default=0.0, help="Optional diagonal ridge for numerical stability")
    parser.add_argument("--device", default="cpu", help="Torch device for Ewald kernel construction")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of structures")
    args = parser.parse_args()

    chi = _to_z_mapping(_load_mapping(args.electronegativity, DEFAULT_ELECTRONEGATIVITY))
    eta = _to_z_mapping(_load_mapping(args.hardness, DEFAULT_HARDNESS))
    device = torch.device(args.device)

    frames = read(args.input, args.index)
    if not isinstance(frames, list):
        frames = [frames]
    if args.limit is not None:
        frames = frames[: args.limit]

    reader = AseDataReader(
        cutoff=args.cutoff,
        compute_neighbor_list=False,
        transforms=[MatScipyNeighborList(cutoff=args.cutoff, return_distance=True, return_cell_displacements=True)],
        return_cell_displacements=True,
        default_dtype=torch.float64,
    )
    ewald = None if args.no_ewald else EwaldSummation(
        cutoff=args.cutoff,
        k_cutoff=args.k_cutoff,
        alpha=args.alpha,
        acc_factor=args.acc_factor,
    ).to(device)

    out_frames: List = []
    abs_means: List[float] = []
    rms_values: List[float] = []
    for idx, atoms in enumerate(frames):
        q = solve_atoms_qeq(
            atoms,
            electronegativity=chi,
            hardness=eta,
            reader=reader,
            ewald=ewald,
            device=device,
            total_charge=args.total_charge,
            ridge=args.ridge,
        )
        labeled = atoms.copy()
        labeled.set_initial_charges(q)
        labeled.info["qeq_label_abs_mean"] = float(np.mean(np.abs(q)))
        labeled.info["qeq_label_rms"] = float(np.sqrt(np.mean(q * q)))
        labeled.info["qeq_label_sum"] = float(np.sum(q))
        out_frames.append(labeled)
        abs_means.append(labeled.info["qeq_label_abs_mean"])
        rms_values.append(labeled.info["qeq_label_rms"])
        if idx < 5 or (idx + 1) % 100 == 0:
            print(
                f"{idx:6d} natoms={len(atoms):4d} "
                f"abs_mean={abs_means[-1]:.6g} rms={rms_values[-1]:.6g} "
                f"min={q.min():.6g} max={q.max():.6g} sum={q.sum():.6g}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, out_frames)
    print(
        f"Wrote {len(out_frames)} structures to {args.output}\n"
        f"overall_abs_mean={np.mean(abs_means):.6g} overall_rms={np.mean(rms_values):.6g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
