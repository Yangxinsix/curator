"""Held-out charged-molecule dipole diagnostic at the ORCA center of mass."""

import csv
import gzip
from pathlib import Path

import torch
from ase.data import atomic_masses
from ase.units import Debye
from curator.data import properties as p


def references(fold):
    with gzip.open(
        Path(__file__).with_name("extension_sabs_dipoles.csv.gz"), "rt"
    ) as stream:
        result = {
            int(row["source_index"]): [float(row[f"DFT_{k}_D"]) for k in "xyz"]
            for row in csv.DictReader(stream)
            if row["split"] == fold
        }
    return result


def dipole(z, batch, idx, refs):
    positions = batch[p.positions].detach().cpu().double()
    numbers = batch[p.atomic_numbers].detach().cpu().tolist()
    mass = torch.tensor(atomic_masses[numbers], dtype=torch.float64)
    origin = (mass @ positions) / mass.sum()
    pred = z.detach().cpu().double().flatten() @ (positions - origin) / Debye
    return dict(prediction=pred.tolist(), DFT=refs[idx])


def aggregate(rows):
    x = torch.tensor([r["dipole_D"]["prediction"] for r in rows], dtype=torch.float64)
    y = torch.tensor([r["dipole_D"]["DFT"] for r in rows], dtype=torch.float64)
    r = torch.corrcoef(torch.stack((x.flatten(), y.flatten())))[0, 1]
    return dict(
        frames=len(x),
        origin="ORCA center of mass, ASE elemental masses",
        component_RMSE_D=(x - y).square().mean().sqrt().item(),
        mean_predicted_magnitude_D=x.norm(dim=1).mean().item(),
        mean_DFT_magnitude_D=y.norm(dim=1).mean().item(),
        signed_component_pearson_r=float(r) if torch.isfinite(r) else None,
    )
