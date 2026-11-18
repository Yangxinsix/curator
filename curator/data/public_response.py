"""One data interface for the audited public energy/response benchmarks.

Readers differ by storage format.  Samples, units, splits, statistics, and the
variational training contract are shared; response labels are never state teachers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from ase import units as ase_units
from ase.data import atomic_numbers, chemical_symbols
from ase.geometry import find_mic
from ase.io import read
from ase.stress import voigt_6_to_full_3x3_stress
from torch.utils.data import DataLoader, Dataset

from . import properties
from .atoms_data import AtomsData
from .collate_atoms_data import collate_atoms_data
from .variational_data import (
    VariationalCondition, VariationalData, canonical_boundary_hash, geometry_hash,
)


PUBLIC_RESPONSE_ROOT = "/workspace/data/datasets/acvr/public_response"
PUBLIC_RESPONSE_UNITS = {
    "condition_energy": "eV", "condition_forces": "eV/Angstrom",
    "dipole": "e*Angstrom", "polarizability": "e^2*Angstrom^2/eV",
    "condition_stress": "eV/Angstrom^3", "born_charge": "dimensionless",
    "dielectric": "dimensionless",
}
_FILES = {
    "field_schnet/ethanol_vacuum": ("sqlite", ("data/ethanol_vacuum.db",)),
    "field_schnet/ethanol_methanol_qmmm": ("sqlite", ("data/ethanol_methanol_qmmm.db",)),
    "allegro_pol/BaTiO3": ("extxyz", ("raw/BaTiO3.xyz",)),
    "mlearn/Si": ("json", ("raw/training.json", "raw/test.json")),
    "phonondb/MgO": ("yaml", ("data/phono3py_params.yaml",)),
}
_MLEARN_COMMIT = "10c427a5480c6281c15c64efaf869b03be04818f"
_FIELDSCHNET_COMMIT = "c82b534e41fe035790700cacc4a330287c95eefa"
_QM_FORCE_REASON = (
    "Point-charge reference forces require the spatial derivative of the local "
    "electric field. The database contains neither external charge positions nor "
    "field gradients. A derivative at fixed stored local fields is not this label."
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _dtype(value):
    if value in ("float64", torch.float64):
        return torch.float64
    if value in ("float32", torch.float32):
        return torch.float32
    raise ValueError("default_dtype must be float32 or float64")


def _record(numbers, positions, cell, pbc, labels, raw_labels, source_id, **meta):
    numbers = np.asarray(numbers, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    pbc = np.broadcast_to(np.asarray(pbc, dtype=bool), (3,)).copy()
    if positions.shape != (len(numbers), 3) or not len(numbers):
        raise ValueError(f"Invalid geometry for {source_id}")
    for key, value in {"positions": positions, "cell": cell, **labels, **raw_labels}.items():
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError(f"Nonfinite {key} in {source_id}")
    return {
        "numbers": numbers, "positions": positions, "cell": cell, "pbc": pbc,
        "labels": {key: np.asarray(value, dtype=np.float64) for key, value in labels.items()},
        "raw_labels": {key: np.asarray(value).copy() for key, value in raw_labels.items()},
        "source_id": str(source_id), "meta": meta,
        "external_field": np.zeros((1, 3), dtype=np.float64),
    }


def geometry_group_fingerprint(record, decimals=6):
    """Conservative grouping invariant to rotation, translation and atom order.

    Species-resolved pair-distance multisets also group any homometric structures;
    that can reduce split sizes, but cannot leak identical/symmetry-related inputs.
    Periodic distances use ASE's general minimum-image implementation.
    """
    numbers, positions = record["numbers"], record["positions"]
    i, j = np.triu_indices(len(numbers), 1)
    vectors = positions[j] - positions[i]
    if record["pbc"].any():
        _, distances = find_mic(vectors, record["cell"], pbc=record["pbc"])
        metric = np.sort(np.linalg.svd(record["cell"], compute_uv=False))
    else:
        distances = np.linalg.norm(vectors, axis=-1)
        metric = np.zeros(3)
    pairs = np.sort(np.stack((numbers[i], numbers[j]), axis=1), axis=1)
    quantized = np.round(distances, decimals)
    order = np.lexsort((quantized, pairs[:, 1], pairs[:, 0]))
    digest = hashlib.sha256()
    digest.update(np.sort(numbers).astype("<i8").tobytes())
    digest.update(np.round(metric, decimals).astype("<f8").tobytes())
    digest.update(pairs[order].astype("<i8").tobytes())
    digest.update(quantized[order].astype("<f8").tobytes())
    return digest.hexdigest()


def _read_sqlite(paths, dataset):
    from ase.db import connect

    qmmm = dataset.endswith("qmmm")
    factors = {
        "energy": ase_units.Hartree, "forces": ase_units.Hartree / ase_units.Bohr,
        "dipole_moment": ase_units.Bohr,
        "polarizability": ase_units.Bohr**2 / ase_units.Hartree,
        "electric_field": ase_units.Hartree / ase_units.Bohr,
    }
    mapping = {"energy": "condition_energy", "forces": "condition_forces",
               "dipole_moment": "dipole", "polarizability": "polarizability"}
    disabled = ({"forces": _QM_FORCE_REASON} if qmmm else {
        "shielding": "Magnetic response is outside this electric-response training contract.",
        "dielectric_constant": "This is the prescribed solvent permittivity (1 in vacuum), not a molecular response target.",
        "magnetic_field": "Zero magnetic boundary input; magnetic observables are not enabled.",
    })
    records = []
    for row in connect(str(paths[0])).select(sort="id"):
        raw = {key: np.asarray(value) for key, value in row.data.items()}
        labels = {target: raw[source] * factors[source] for source, target in mapping.items()
                  if source not in disabled}
        record = _record(row.numbers, row.positions * ase_units.Bohr,
                         np.asarray(row.cell) * ase_units.Bohr, row.pbc, labels, raw,
                         f"sqlite:{row.id}", source_row_id=row.id,
                         molecule_formula="".join(f"{chemical_symbols[z]}{n}" for z, n in sorted(Counter(row.numbers).items())),
                         force_semantics="disabled_missing_field_gradient" if qmmm else "isolated_cartesian_gradient")
        field = raw["electric_field"] * factors["electric_field"]
        if qmmm:
            if field.shape != (len(row.numbers), 3):
                raise ValueError("QM/MM requires a separate three-vector field at each atom")
            record["local_external_field"] = field
        else:
            if field.shape != (3,):
                raise ValueError("Uniform external field must have shape [3]")
            record["external_field"] = field.reshape(1, 3)
        records.append(record)
    audit = {
        "source_units": "Hartree atomic units, including database positions in Bohr",
        "conversion_factors": factors, "positions_factor": ase_units.Bohr,
        "disabled_labels": disabled, "field_semantics": "fixed_per_atom_field" if qmmm else "uniform_field",
        "dipole_origin": "source_cartesian_origin; neutral molecules; no recentering",
        "polarizability_definition": "response to a uniform field increment, with stored local fields held fixed",
        "force_evidence": [
            f"https://github.com/atomistic-machine-learning/field_schnet/blob/{_FIELDSCHNET_COMMIT}/src/field_schnet/atomistic/model.py",
            f"https://github.com/atomistic-machine-learning/field_schnet/blob/{_FIELDSCHNET_COMMIT}/src/scripts/qmmm/qmmm_server.py",
        ] if qmmm else [],
    }
    return records, audit


def _read_extxyz(paths, dataset):
    records = []
    for index, atoms in enumerate(read(paths[0], index=":", format="extxyz")):
        raw = {**atoms.info, **atoms.arrays, **atoms.calc.results}
        raw = {key: value for key, value in raw.items() if key not in {"numbers", "positions"}}
        labels = {
            "condition_energy": np.asarray(raw["total_energy"]).reshape(1),
            "condition_forces": raw["forces"], "dipole": raw["polarization"],
            "polarizability": np.asarray(raw["polarizability"]).reshape(3, 3),
            "born_charge": np.asarray(raw["born_charge"]).reshape(len(atoms), 3, 3),
            "condition_stress": voigt_6_to_full_3x3_stress(raw["stress"]),
        }
        records.append(_record(atoms.numbers, atoms.positions, atoms.cell, atoms.pbc,
                               labels, raw, f"extxyz:{index}",
                               original_dataset_index=int(raw["original_dataset_index"]),
                               force_semantics="zero_field_cartesian_gradient"))
    return records, {
        "source_units": "eV, Angstrom, elementary charge; upstream numerical values retained",
        "disabled_labels": {"original_dataset_index": "Provenance identifier, not a physical target."},
        "field_semantics": "zero_field_response_only",
        "dipole_origin": "periodic cell dipole; source Berry branch",
        "dipole_branch": "source parser folds fractional cell dipole into its unit branch",
        "born_axis_order": ["atom", "dipole_component", "position_component"],
        "born_unit_convention": "dimensionless numbers Z*/e; source numbers reported in elementary-charge units",
        "born_evidence": "parse_QE.py get_born_charges: q[n,i,j] = delta F_j / delta field_i",
        "polarizability_audit": "e*Angstrom^2/V equals e^2*Angstrom^2/eV; factor 1. The source polarizability annotation is dimensionally consistent.",
        "stress_audit": "Source comment eV/A**2 is inconsistent with Sunit = hartree2eV / bohr2A**3 * (-1). The actual conversion produces tension-positive eV/Angstrom^3; no second sign flip.",
        "evidence_file": "source/scripts/2.ParseQE/parse_QE.py",
    }


def _si_stress_audit(documents):
    """Validate the raw VASP kbar convention against source elastic E/cell data."""
    rows = [row for row in documents if re.search(r"mode 0, strain", row.get("description", ""))]
    strains = np.array([float(row["description"].split("strain ")[1]) for row in rows])
    energies = np.array([row["outputs"]["energy"] for row in rows])
    volumes = np.array([np.linalg.det(row["structure"]["lattice"]["matrix"]) for row in rows])
    pressure = np.array([row["outputs"]["virial_stress"][0] for row in rows])
    if len(rows) < 7:
        raise ValueError("Si stress audit requires the original elastic-strain samples")
    # The source strain is Green-Lagrange: lambda_x = sqrt(1 + 2*strain).
    coefficients = np.polynomial.polynomial.polyfit(strains, energies, 6)
    derivative = np.polynomial.polynomial.polyval(strains, np.polynomial.polynomial.polyder(coefficients))
    tension_from_energy = derivative * (1 + 2 * strains) / volumes
    conversion = -0.1 * ase_units.GPa
    tension_from_label = pressure * conversion
    relative_error = float(np.linalg.norm(tension_from_label - tension_from_energy) / np.linalg.norm(tension_from_energy))
    if relative_error > 0.06:
        raise ValueError(f"Source Si energy-strain audit rejects kbar/sign convention: {relative_error}")
    return {
        "raw_unit": "kbar", "raw_order": ["xx", "yy", "zz", "xy", "yz", "xz"],
        "raw_sign": "positive_compression", "canonical_sign": "positive_tension",
        "conversion_factor": conversion, "energy_strain_relative_error": relative_error,
        "energy_strain_audit": "degree-6 fit on source mode-0 Green-Lagrange strains; compare dE/dstrain*(1+2strain)/volume",
        "author_order_evidence": f"https://github.com/materialyzeai/mlearn/blob/{_MLEARN_COMMIT}/mlearn/potentials/mtp.py#L59-L60",
        "author_matrix_evidence": f"https://github.com/materialyzeai/mlearn/blob/{_MLEARN_COMMIT}/mlearn/potentials/gap.py#L62-L64",
        "caution": "Author backend conversion code is inconsistent across backends; raw JSON unit/sign were independently checked against its own E/cell strain series, not copied from a backend output convention.",
    }


def _read_json(paths, dataset):
    documents = [json.loads(path.read_text()) for path in paths]
    stress_audit = _si_stress_audit(documents[0])
    records = []
    for phase, rows in zip(("official_train", "official_test"), documents):
        for index, row in enumerate(rows):
            structure, raw = row["structure"], row["outputs"]
            numbers = [atomic_numbers[site["species"][0]["element"]] for site in structure["sites"]]
            xx, yy, zz, xy, yz, xz = raw["virial_stress"]
            stress = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]) * stress_audit["conversion_factor"]
            labels = {"condition_energy": np.array([raw["energy"]]),
                      "condition_forces": raw["forces"], "condition_stress": stress}
            records.append(_record(numbers, [site["xyz"] for site in structure["sites"]],
                                   structure["lattice"]["matrix"], True, labels, raw,
                                   f"{phase}:{index}", official_split=phase,
                                   group=row["group"], description=row.get("description", ""),
                                   force_semantics="zero_field_cartesian_gradient"))
    return records, {"source_units": "eV, Angstrom; raw VASP pressure in kbar",
                     "stress": stress_audit, "disabled_labels": {},
                     "field_semantics": "zero_field_mechanical_data"}


def _read_yaml(paths, dataset):
    import yaml

    source = yaml.safe_load(paths[0].read_text())
    records = []

    def add(cell_key, perturbations, entry, role):
        cell = source[cell_key]
        lattice = np.asarray(cell["lattice"])
        positions = np.asarray([point["coordinates"] for point in cell["points"]]) @ lattice
        numbers = [atomic_numbers[point["symbol"]] for point in cell["points"]]
        for displacement in perturbations:
            positions[int(displacement["atom"]) - 1] += displacement["displacement"]
        raw = {"forces": np.asarray(entry["forces"])}
        records.append(_record(numbers, positions, lattice, True,
                               {"condition_forces": raw["forces"]}, raw,
                               f"{role}:{entry['displacement_id']}",
                               role=role, displacement_id=int(entry["displacement_id"]),
                               perturbations=[{"atom": int(p["atom"]) - 1, "displacement": p["displacement"]} for p in perturbations],
                               force_semantics="DFT force at supplied displaced geometry"))

    for parent in source["displacement_pairs"]:
        add("supercell", [parent], parent, "single_displacement")
        for child in parent["paired_with"]:
            add("supercell", [parent, child], child, "paired_displacement")
    for entry in source["phonon_displacements"]:
        add("phonon_supercell", [entry], entry, "phonon_holdout")
    primitive = source["primitive_cell"]
    reference = {
        "cell": primitive["lattice"],
        "numbers": [atomic_numbers[p["symbol"]] for p in primitive["points"]],
        "positions": (np.asarray([p["coordinates"] for p in primitive["points"]]) @ primitive["lattice"]).tolist(),
        "born_charge": source["born_effective_charge"],
        "dielectric": source["dielectric_constant"],
        "units": {"positions": "Angstrom", "cell": "Angstrom",
                  "born_charge": "dimensionless (Z*/e)", "dielectric": "dimensionless"},
        "semantics": "single equilibrium primitive-cell reference only; not labels on displaced frames",
    }
    return records, {
        "source_units": "VASP/Phonopy Angstrom and eV/Angstrom; force labels only",
        "disabled_labels": {}, "field_semantics": "zero_field_displacement_data",
        "reference_observables": reference,
        "displacement_audit": "YAML atom indices are 1-based; paired frames add both parent and child displacements, including when both address the same atom.",
    }


_READERS = {"sqlite": _read_sqlite, "extxyz": _read_extxyz, "json": _read_json, "yaml": _read_yaml}


def _grouped_split(records, indices, ratios, seed, stratum_key=None):
    strata = defaultdict(list)
    for index in indices:
        strata[records[index]["meta"].get(stratum_key, "all")].append(index)
    result = {phase: [] for phase in ratios}
    generator = np.random.default_rng(seed)
    for stratum in sorted(strata):
        groups = defaultdict(list)
        for index in strata[stratum]:
            groups[records[index]["geometry_group"]].append(index)
        keys = sorted(groups)
        generator.shuffle(keys)
        target = {phase: ratio * len(strata[stratum]) for phase, ratio in ratios.items()}
        counts = dict.fromkeys(ratios, 0)
        for key in keys:
            # Whole groups minimize deviations from the requested sample quotas.
            phase = max(ratios, key=lambda p: target[p] - counts[p])
            result[phase].extend(groups[key])
            counts[phase] += len(groups[key])
    return {phase: sorted(values) for phase, values in result.items()}


def _make_splits(records, dataset, seed):
    excluded = []
    if dataset == "mlearn/Si":
        test = [i for i, row in enumerate(records) if row["meta"]["official_split"] == "official_test"]
        test_groups = {records[i]["geometry_group"] for i in test}
        candidates = []
        for i, row in enumerate(records):
            if row["meta"]["official_split"] == "official_train":
                (excluded if row["geometry_group"] in test_groups else candidates).append(i)
        splits = _grouped_split(records, candidates, {"train": 0.9, "val": 0.1}, seed, "group")
        splits["test"] = test
        rule = "official test unchanged; any symmetry-group overlap excluded from official train; train/val grouped within source categories"
    elif dataset == "allegro_pol/BaTiO3":
        splits = {"train": list(range(41)), "val": list(range(41, 50)), "test": list(range(50, len(records)))}
        rule = "contiguous source order blocks 41/9/9; no random temporal interleaving"
    else:
        holdout = [i for i, row in enumerate(records) if row["meta"].get("role") == "phonon_holdout"]
        candidates = [i for i in range(len(records)) if i not in holdout]
        splits = _grouped_split(records, candidates, {"train": 0.8, "val": 0.1, "test": 0.1}, seed,
                                "molecule_formula" if dataset.endswith("qmmm") else None)
        if holdout:
            splits["phonon_holdout"] = holdout
        rule = "seeded whole geometry groups; 80/10/10 sample quotas; molecule-stratified for QM/MM; separate 512-atom phonon holdout"
    group_owner = {}
    for phase, indices in splits.items():
        for index in indices:
            group = records[index]["geometry_group"]
            if group in group_owner and group_owner[group] != phase:
                raise ValueError(f"Symmetry-related geometry crosses {group_owner[group]} and {phase}")
            group_owner[group] = phase
    return splits, {"rule": rule, "seed": seed, "group_fingerprint": "species-resolved minimum-image pair-distance multiset plus cell singular values; rounded to 1e-6 Angstrom",
                    "excluded_official_train_indices": excluded,
                    "excluded_reason": "same/symmetry-related geometry as unchanged official test" if excluded else None}


def _training_statistics(records, indices):
    species = sorted({int(z) for index in indices for z in records[index]["numbers"]})
    energy_indices = [i for i in indices if "condition_energy" in records[i]["labels"]]
    refs = np.zeros(len(species))
    if energy_indices:
        composition = np.array([[np.count_nonzero(records[i]["numbers"] == z) for z in species] for i in energy_indices])
        energy = np.array([float(records[i]["labels"]["condition_energy"].item()) for i in energy_indices])
        refs = np.linalg.lstsq(composition, energy, rcond=None)[0]
    reference = dict(zip(species, refs))
    scales = {}
    for label in sorted({key for i in indices for key in records[i]["labels"]}):
        values = []
        for index in indices:
            if label in records[index]["labels"]:
                value = records[index]["labels"][label]
                if label == "condition_energy":
                    value = value - sum(reference[int(z)] for z in records[index]["numbers"])
                values.append(value.reshape(-1))
        values = np.concatenate(values)
        scales[label] = max(float(np.sqrt(np.mean(values**2))), 1.0e-8)
    return {
        "fit_split": "train", "fit_indices": list(indices), "energy_reference_kind": "composition_least_squares_on_selected_training_energies",
        "atomic_reference_energies_eV": {str(z): float(value) for z, value in reference.items()},
        "energy_target_definition": "source energy in eV minus sum of training-fitted atomic references; add that sum to restore source zero",
        "supervised_scales": scales, "scale_kind": "training component RMS after energy-reference subtraction, minimum 1e-8",
    }


@dataclass(frozen=True)
class LocalFieldAugmentation:
    """Fixed residual-only conditions, independent of worker and batch order."""

    conditions: int
    std_ratio: float
    seed: int
    training_rms: float

    def apply(self, condition, local_field, source_index, label_names):
        generated = []
        # Each geometry gets a reproducible local stream. No mutable worker RNG
        # or epoch counter is needed to resume the exact same training problem.
        digest = hashlib.sha256(f"{self.seed}:{source_index}".encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % (2**63))
        for index in range(self.conditions):
            noise = torch.randn(local_field.shape, dtype=local_field.dtype, generator=generator).to(local_field.device)
            generated.append(replace(
                condition, condition_uid=f"{condition.condition_uid}:local_field:{index}",
                inputs={**condition.inputs, "local_external_field": local_field + self.std_ratio * self.training_rms * noise},
                labels={}, label_units={}, label_metadata={}, responses=(),
                label_eligible={key: False for key in label_names},
                provenance={**condition.provenance, "residual_only": True,
                            "local_field_augmentation": {**asdict(self), "index": index}},
            ))
        return tuple(generated)


class PublicResponseDataset(Dataset):
    """A generic split of atom records and observable labels, with no state teacher."""

    def __init__(self, records, indices, metadata, split, default_dtype="float64",
                 state_kind="vector", state_unit="e*Angstrom", conjugate_unit="V/Angstrom",
                 local_field_augmentation=None):
        self.records, self.indices, self.metadata, self.split = records, list(indices), metadata, split
        self.default_dtype = _dtype(default_dtype)
        self.state_kind, self.state_unit, self.conjugate_unit = state_kind, state_unit, conjugate_unit
        self.species = metadata["species"]
        self.local_field_augmentation = local_field_augmentation if split == "train" else None

    def __len__(self):
        return len(self.indices)

    def get_raw_labels(self, index):
        """All unconverted original labels, including explicitly disabled ones."""
        return copy.deepcopy(self.records[self.indices[index]]["raw_labels"])

    def get_n_atoms(self, index):
        return len(self.records[self.indices[index]]["numbers"])

    def __getitem__(self, index):
        from curator.variational import BoundarySpec, GroupSpec, VariableSpec

        source_index = self.indices[index]
        record = self.records[source_index]
        tensor = lambda value: torch.as_tensor(np.asarray(value).copy(), dtype=self.default_dtype)
        count = len(record["numbers"])
        atoms = {
            properties.atomic_numbers: torch.tensor(record["numbers"], dtype=torch.long),
            properties.positions: tensor(record["positions"]),
            properties.n_atoms: torch.tensor([count]), properties.cell: tensor(record["cell"]),
            properties.pbc: torch.tensor(record["pbc"].reshape(1, 3)),
            "external_field": tensor(record["external_field"]),
        }
        if "local_external_field" in record:
            atoms["local_external_field"] = tensor(record["local_external_field"])
        refs = self.metadata["training_statistics"]["atomic_reference_energies_eV"]
        offset = sum(refs.get(str(int(z)), 0.0) for z in record["numbers"])
        # Subtract the large source zero before a possible float32 cast.
        labels = {key: tensor(value - offset if key == "condition_energy" else value)
                  for key, value in record["labels"].items()}
        if self.state_kind == "scalar":
            if np.any(record["external_field"]) or "local_external_field" in record:
                raise ValueError("Scalar internal states need an explicit field-to-state mapping for nonzero vector fields")
            atoms.pop("external_field")
        components = 3 if self.state_kind == "vector" else 1
        variable = VariableSpec("internal", self.state_kind, components,
                                self.state_unit, self.conjugate_unit)
        boundary = asdict(BoundarySpec(variable, groups=(GroupSpec.all(),)))
        uid = f"{self.metadata['dataset_id']}:{record['source_id']}"
        gauge = copy.deepcopy(self.metadata["gauge"])
        label_metadata = {key: {"source": record["source_id"]} for key in labels}
        if "dipole" in labels:
            label_metadata["dipole"].update({"origin": "source_cartesian_origin", "branch": gauge.get("dipole_branch", "not_periodic")})
        if "condition_energy" in labels:
            label_metadata["condition_energy"].update({"source_energy_offset_eV": offset, "energy_kind": "training_referenced_source_energy"})
        condition = VariationalCondition(uid + ":condition", None, None, None,
            boundary, canonical_boundary_hash(boundary), gauge, labels,
            {key: PUBLIC_RESPONSE_UNITS[key] for key in labels}, label_metadata,
            provenance={"source_index": source_index, "source_id": record["source_id"], "split": self.split})
        conditions = (condition,)
        if self.local_field_augmentation is not None:
            conditions += self.local_field_augmentation.apply(
                condition, atoms["local_external_field"], source_index,
                self.metadata["enabled_labels"],
            )
        payload = VariationalData("1", uid,
            geometry_hash(atoms[properties.atomic_numbers], atoms[properties.positions], atoms[properties.cell], atoms[properties.pbc]),
            asdict(variable), torch.tensor([0, count]), torch.arange(count), ("all",), ("system",), conditions,
            provenance={"geometry_group": record["geometry_group"], "split": self.split})
        payload.validate(count)
        return AtomsData(atoms, {}, "public_response", meta={
            **copy.deepcopy(record["meta"]), "dataset_id": self.metadata["dataset_id"],
            "source_index": source_index, "source_id": record["source_id"], "split": self.split,
            "geometry_group": record["geometry_group"], "source_energy_offset_eV": offset,
            "raw_label_names": sorted(record["raw_labels"]),
            "disabled_labels": self.metadata["unit_audit"]["disabled_labels"],
        }, variational=payload)


class PublicResponseDataModule(pl.LightningDataModule):
    """The common training/evaluation and replay interface for five public datasets."""

    def __init__(self, dataset, root=PUBLIC_RESPONSE_ROOT, batch_size=8, seed=17,
                 default_dtype="float64", num_workers=0, shuffle=True,
                 train_limit=None, val_limit=None, test_limit=None, heads=None,
                 state_kind="vector", state_unit="e*Angstrom", conjugate_unit="V/Angstrom",
                 split_file=None, verify_hashes=True, local_field_conditions=0,
                 local_field_std_ratio=0.1, local_field_seed=None):
        super().__init__()
        if dataset not in _FILES:
            raise ValueError(f"Unknown public dataset {dataset!r}; choose from {sorted(_FILES)}")
        if state_kind not in {"scalar", "vector"}:
            raise ValueError("state_kind must be scalar or vector")
        if type(batch_size) is not int or batch_size < 1 or type(num_workers) is not int or num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers nonnegative")
        self.dataset_id, self.root = dataset, str(Path(root).expanduser().resolve())
        self.datapath = str(Path(self.root) / dataset)
        self.train_path = self.datapath
        self.batch_size, self.seed, self.num_workers, self.shuffle = batch_size, int(seed), num_workers, bool(shuffle)
        self.default_dtype = _dtype(default_dtype)
        self.state_kind, self.state_unit, self.conjugate_unit = state_kind, state_unit, conjugate_unit
        self.limits = {"train": train_limit, "val": val_limit, "test": test_limit}
        for limit in self.limits.values():
            if limit is not None and (type(limit) is not int or limit < 1):
                raise ValueError("dataset limits must be positive integers or None")
        self.heads = heads or ["variational_prediction"]
        self.split_file, self.verify_hashes = split_file, bool(verify_hashes)
        self.generator = torch.Generator().manual_seed(self.seed)
        if type(local_field_conditions) is not int or local_field_conditions < 0:
            raise ValueError("local_field_conditions must be a nonnegative integer")
        if not np.isfinite(local_field_std_ratio) or local_field_std_ratio <= 0:
            raise ValueError("local_field_std_ratio must be finite and positive")
        if local_field_conditions and (state_kind != "vector" or not dataset.endswith("qmmm")):
            raise ValueError("local field augmentation requires the QM/MM vector task")
        self.local_field_conditions = local_field_conditions
        self.local_field_std_ratio = float(local_field_std_ratio)
        self.local_field_seed = self.seed if local_field_seed is None else int(local_field_seed)
        self.datasets = {}

    def setup(self, stage=None):
        if self.datasets:
            return
        root = Path(self.datapath)
        manifest = json.loads((root / "manifest.json").read_text())
        source_hashes = {}
        for item in manifest["files"]:
            path = (root / item["path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Source manifest path escapes dataset root")
            digest = _sha256(path)
            if self.verify_hashes and (digest != item["sha256"] or path.stat().st_size != item["bytes"]):
                raise ValueError(f"Public source hash/size mismatch: {item['path']}")
            source_hashes[item["path"]] = digest
        format_name, files = _FILES[self.dataset_id]
        records, audit = _READERS[format_name]([root / name for name in files], self.dataset_id)
        for record in records:
            record["geometry_group"] = geometry_group_fingerprint(record)
        full_splits, split_audit = _make_splits(records, self.dataset_id, self.seed)
        if self.split_file is not None:
            supplied = json.loads(Path(self.split_file).read_text())
            if supplied != full_splits:
                raise ValueError("Public split file differs from the reproducible grouped split")
        splits = {phase: values[:self.limits[phase]] if self.limits.get(phase) else list(values)
                  for phase, values in full_splits.items()}
        if any(not splits[phase] for phase in ("train", "val", "test")):
            raise ValueError("Public train/val/test splits must all be nonempty")
        statistics = _training_statistics(records, splits["train"])
        species = sorted({int(z) for record in records for z in record["numbers"]})
        self.metadata = {
            "schema": "public_response_v1", "dataset_id": self.dataset_id,
            "source_manifest_sha256": _sha256(root / "manifest.json"), "source_hashes": source_hashes,
            "source_format": format_name, "source_files": list(files),
            "units": dict(PUBLIC_RESPONSE_UNITS), "unit_audit": audit,
            "default_dtype": str(self.default_dtype).removeprefix("torch."),
            "state_kind": self.state_kind, "state_unit": self.state_unit, "conjugate_unit": self.conjugate_unit,
            "species": [chemical_symbols[z] for z in species], "atomic_numbers": species,
            "full_split_indices": full_splits, "splits": splits, "split_indices": splits,
            "split_audit": split_audit, "limits": dict(self.limits),
            "geometry_groups": [record["geometry_group"] for record in records],
            "source_ids": [record["source_id"] for record in records],
            "training_statistics": statistics,
            "enabled_labels": sorted({key for record in records for key in record["labels"]}),
            "raw_labels": sorted({key for record in records for key in record["raw_labels"]}),
            "gauge": {"boundary": "unconstrained_internal_state",
                      "field_semantics": audit["field_semantics"],
                      "energy_zero": "training_composition_reference",
                      "dipole_branch": audit.get("dipole_branch", "not_applicable")},
            "split_statistics": {phase: {"count": len(indices),
                "atom_counts": dict(Counter(str(len(records[i]["numbers"])) for i in indices)),
                "geometry_group_count": len({records[i]["geometry_group"] for i in indices})}
                for phase, indices in splits.items()},
        }
        if "reference_observables" in audit:
            self.metadata["reference_observables"] = copy.deepcopy(audit["reference_observables"])
        augmentation = None
        if self.local_field_conditions:
            train_fields = np.concatenate([records[index]["local_external_field"].reshape(-1)
                                           for index in splits["train"]])
            rms = float(np.sqrt(np.mean(train_fields**2)))
            if not np.isfinite(rms) or rms <= 0:
                raise ValueError("local field augmentation requires positive finite training field RMS")
            augmentation = LocalFieldAugmentation(
                self.local_field_conditions, self.local_field_std_ratio, self.local_field_seed, rms)
            self.metadata["local_field_augmentation"] = {
                **asdict(augmentation), "unit": self.conjugate_unit, "fit_split": "train",
                "fit_indices": list(splits["train"]), "standard_deviation": rms * self.local_field_std_ratio,
                "sampling": "fixed_per_source_index_sha256_seed; independent normal field components",
                "labels": "original condition only; augmented conditions residual-only",
                "conditions_per_geometry": self.local_field_conditions + 1,
            }
        self.records = records
        self.datasets = {phase: PublicResponseDataset(records, indices, self.metadata, phase,
                         self.default_dtype, self.state_kind, self.state_unit, self.conjugate_unit, augmentation)
                         for phase, indices in splits.items()}
        self.train_dataset, self.val_dataset, self.test_dataset = (self.datasets[phase] for phase in ("train", "val", "test"))
        self.dataset, self.species = self.train_dataset, self.metadata["species"]
        self.train_idx, self.val_idx, self.test_idx = (splits[phase] for phase in ("train", "val", "test"))
        self.phonon_holdout_dataset = self.datasets.get("phonon_holdout")

    def _loader(self, split):
        self.setup()
        generator = self.generator if split == "train" else torch.Generator().manual_seed(self.seed + 1)
        return DataLoader(self.datasets[split], batch_size=self.batch_size,
                          shuffle=self.shuffle and split == "train", generator=generator,
                          num_workers=self.num_workers, collate_fn=collate_atoms_data)

    def train_dataloader(self):
        return self._loader("train")

    def val_dataloader(self):
        return self._loader("val")

    def test_dataloader(self):
        return self._loader("test")

    def phonon_holdout_dataloader(self):
        return self._loader("phonon_holdout")

    def state_dict(self):
        self.setup()
        return {"metadata": copy.deepcopy(self.metadata), "training_seed": self.seed,
                "generator_state": self.generator.get_state()}

    def load_state_dict(self, state_dict):
        self.setup()
        if state_dict["metadata"] != self.metadata:
            raise ValueError("Public source, units, splits, limits, or training statistics differ from checkpoint")
        if int(state_dict["training_seed"]) != self.seed:
            raise ValueError("Public training seed differs from checkpoint")
        self.generator.set_state(state_dict["generator_state"])


__all__ = ["PUBLIC_RESPONSE_ROOT", "PUBLIC_RESPONSE_UNITS", "PublicResponseDataset", "PublicResponseDataModule",
           "LocalFieldAugmentation"]
