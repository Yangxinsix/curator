#!/usr/bin/env python3
"""Generate a tiny same-geometry, multi-charge periodic DFT oracle.

This is an interface and data-contract oracle, not a production electronic-
structure dataset.  Charged periodic energies depend on the recorded cell and
electrostatic convention and must not be interpreted as absolute voltages.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pyscf
from pyscf.pbc import dft, gto


HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--distances", type=float, nargs="+", default=[0.6, 0.7, 0.9])
    parser.add_argument("--charges", type=int, nargs="+", default=[-1, 0, 1])
    parser.add_argument("--cell-length", type=float, default=6.0)
    parser.add_argument("--mesh", type=int, default=41)
    parser.add_argument("--precision", type=float, default=1.0e-6)
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def calculate(distance, charge, args):
    center = 0.5 * args.cell_length
    positions = np.array(
        [[center, center, center - 0.5 * distance], [center, center, center + 0.5 * distance]]
    )
    cell = gto.Cell()
    cell.atom = [["H", position] for position in positions]
    cell.a = np.eye(3) * args.cell_length
    cell.unit = "A"
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.charge = charge
    cell.spin = int((2 - charge) % 2)
    cell.precision = args.precision
    cell.mesh = [args.mesh] * 3
    cell.verbose = 3
    cell.build()

    method = dft.RKS(cell) if cell.spin == 0 else dft.UKS(cell)
    method = method.multigrid_numint(mesh=[args.mesh] * 3)
    method.xc = args.xc
    method.conv_tol = 1.0e-9
    method.max_cycle = 100
    energy = method.kernel()
    if not method.converged:
        raise RuntimeError("SCF did not converge for distance=%g charge=%d" % (distance, charge))
    gradient = method.nuc_grad_method().kernel()

    mo_energy = method.mo_energy
    mo_occupancy = method.mo_occ
    if isinstance(mo_energy, np.ndarray) and mo_energy.ndim == 1:
        mo_energy = np.stack((mo_energy, np.full_like(mo_energy, np.nan)))
        mo_occupancy = np.stack((mo_occupancy, np.full_like(mo_occupancy, np.nan)))
    return {
        "atomic_numbers": np.array([1, 1], dtype=np.int32),
        "positions": positions,
        "cell": np.asarray(cell.a) * BOHR_TO_ANGSTROM,
        "pbc": np.ones(3, dtype=np.bool_),
        "energy": np.array([energy * HARTREE_TO_EV]),
        "forces": -np.asarray(gradient) * HARTREE_TO_EV / BOHR_TO_ANGSTROM,
        "total_charge": np.array([charge], dtype=np.float64),
        "electron_count": np.array([cell.nelectron], dtype=np.int32),
        "spin": np.array([cell.spin], dtype=np.int32),
        "external_field": np.zeros(3),
        "mo_energy": np.asarray(mo_energy) * HARTREE_TO_EV,
        "mo_occupancy": np.asarray(mo_occupancy),
        "solver": type(method).__name__,
    }


def write_dataset(handle, name, value, unit):
    dataset = handle.create_dataset(name, data=value)
    dataset.attrs["units"] = unit


def main():
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError("%s already exists; pass --overwrite to replace it" % args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    records = []
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema"] = "curator.electronic_states"
            handle.attrs["schema_version"] = "1.0"
            handle.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            handle.attrs["generator"] = "scripts/generate_acvr_pyscf_pbc.py"
            handle.attrs["pyscf_version"] = pyscf.__version__
            handle.attrs["electrostatic_convention"] = json.dumps(
                {
                    "dimension": 3,
                    "k_point": "Gamma",
                    "exchange_divergence": "ewald",
                    "charged_periodic_cell": True,
                    "warning": "Charged-state energies are cell- and convention-dependent.",
                },
                sort_keys=True,
            )
            geometries = handle.create_group("geometries")
            for geometry_index, distance in enumerate(args.distances):
                geometry_id = "h2-r%.6f" % distance
                center = 0.5 * args.cell_length
                positions = np.array(
                    [
                        [center, center, center - 0.5 * distance],
                        [center, center, center + 0.5 * distance],
                    ]
                )
                geometry = geometries.create_group(geometry_id)
                geometry.attrs["geometry_id"] = geometry_id
                geometry.attrs["geometry_index"] = geometry_index
                geometry.attrs["distance_angstrom"] = distance
                write_dataset(geometry, "atomic_numbers", np.array([1, 1], dtype=np.int32), "atomic_number")
                write_dataset(geometry, "positions", positions, "angstrom")
                write_dataset(geometry, "cell", np.eye(3) * args.cell_length, "angstrom")
                write_dataset(geometry, "pbc", np.ones(3, dtype=np.bool_), "dimensionless")
                conditions = geometry.create_group("conditions")
                for condition_index, charge in enumerate(args.charges):
                    state_id = "%s-q%+d" % (geometry_id, charge)
                    print("calculating", state_id, flush=True)
                    result = calculate(distance, charge, args)
                    group = conditions.create_group(state_id)
                    group.attrs["state_id"] = state_id
                    group.attrs["condition_index"] = condition_index
                    group.attrs["solver"] = result.pop("solver")
                    group.attrs["xc"] = args.xc
                    group.attrs["basis"] = "gth-szv"
                    group.attrs["pseudo"] = "gth-pade"
                    group.attrs["mesh"] = args.mesh
                    group.attrs["precision"] = args.precision
                    group.attrs["scf_converged"] = True
                    write_dataset(group, "energy", result["energy"], "eV")
                    write_dataset(group, "forces", result["forces"], "eV/angstrom")
                    write_dataset(group, "total_charge", result["total_charge"], "elementary_charge")
                    write_dataset(group, "electron_count", result["electron_count"], "electron")
                    write_dataset(group, "spin", result["spin"], "n_alpha_minus_n_beta")
                    write_dataset(group, "external_field", result["external_field"], "V/angstrom")
                    write_dataset(group, "mo_energy", result["mo_energy"], "eV")
                    write_dataset(group, "mo_occupancy", result["mo_occupancy"], "electron")
                    records.append(
                        {
                            "state_id": state_id,
                            "geometry_id": geometry_id,
                            "distance_angstrom": distance,
                            "total_charge": charge,
                            "electron_count": int(result["electron_count"][0]),
                            "spin": int(result["spin"][0]),
                        }
                    )
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "schema": "curator.electronic_states",
        "schema_version": "1.0",
        "artifact": args.output.name,
        "sha256": digest,
        "records": records,
        "capabilities": {
            "labels": [
                "energy",
                "forces",
                "total_charge",
                "electron_count",
                "spin",
                "external_field",
            ],
            "can_validate": ["same_geometry_cross_charge", "periodic_fixed_total_charge"],
            "cannot_validate": [
                "absolute_electrode_potential",
                "constant_electrode_potential",
                "electrode_capacitance",
            ],
        },
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", args.output)
    print("sha256", digest)


if __name__ == "__main__":
    main()
