"""Geometry-paired reader for the public TRECI Cu/water dataset."""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ase.io import read

from . import properties
from ._neighborlist import NeighborListTransform, TorchNeighborList
from ._transform import Transform
from .atoms_data import AtomsData
from .capabilities import DatasetCapabilityManifest
from .variational_data import (
    VariationalCondition,
    VariationalData,
    canonical_boundary_hash,
)


_CONDITIONS = (
    ("PZC_label", None, "PZC_label/dataset_PZC_label_round0.xyz"),
    ("m05", -0.5, "m05/dataset_m05_round0.xyz"),
    ("m075", -0.75, "m075/dataset_m075_round0.xyz"),
    ("m1", -1.0, "m1/dataset_m1_round0.xyz"),
    ("m125", -1.25, "m125/dataset_m125_round0.xyz"),
    ("m15", -1.5, "m15/dataset_m15_round0.xyz"),
    ("m175", -1.75, "m175/dataset_m175_round0.xyz"),
    ("m2", -2.0, "m2/dataset_m2_round0.xyz"),
)


def treci_geometry_split(
    dataset: "TreciDataset",
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    salt: str = "curator-acvr-treci-v1",
) -> Dict[str, List[int]]:
    """Deterministically split geometry records, never individual conditions."""

    if not isinstance(dataset, TreciDataset):
        raise TypeError("dataset must be a TreciDataset")
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1):
        raise ValueError("validation and test fractions must lie strictly between zero and one")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than one")
    order = sorted(
        range(len(dataset)),
        key=lambda index: hashlib.sha256(
            (salt + ":" + dataset._geometry_uids[index]).encode("utf-8")
        ).digest(),
    )
    n_validation = max(1, round(validation_fraction * len(order)))
    n_test = max(1, round(test_fraction * len(order)))
    return {
        "train": order[n_validation + n_test :],
        "validation": order[:n_validation],
        "test": order[n_validation : n_validation + n_test],
    }


def _geometry_uid(atoms) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(atoms.numbers, dtype="<i4").tobytes())
    digest.update(np.asarray(atoms.positions, dtype="<f8").tobytes())
    return digest.hexdigest()


class TreciDataset(torch.utils.data.Dataset):
    """Pair PZC and seven target-potential labels by identical nuclei."""

    def __init__(
        self,
        datapath: str,
        cutoff: Optional[float] = None,
        compute_neighbor_list: bool = True,
        transforms: Optional[List[Transform]] = None,
        default_dtype: torch.dtype = torch.get_default_dtype(),
        task: str = "treci",
        weight: float = 1.0,
    ) -> None:
        self.datapath = str(Path(datapath).expanduser().resolve())
        root = Path(self.datapath)
        if not root.is_dir():
            raise FileNotFoundError(self.datapath)
        self.transforms = [] if transforms is None else list(transforms)
        if compute_neighbor_list:
            if cutoff is None:
                raise ValueError("cutoff is required when computing neighbor lists")
            if not any(isinstance(item, NeighborListTransform) for item in self.transforms):
                self.transforms.append(TorchNeighborList(cutoff=float(cutoff)))
        self.default_dtype = default_dtype
        self.task = task
        self.weight = float(weight)

        frames: Dict[str, Dict[str, object]] = {}
        for label, _, relative_path in _CONDITIONS:
            path = root / relative_path
            if not path.is_file():
                raise FileNotFoundError(path)
            mapped = {}
            for atoms in read(path, index=":", format="extxyz"):
                uid = _geometry_uid(atoms)
                if uid in mapped:
                    raise ValueError("TRECI file %s contains duplicate geometry %s" % (path, uid))
                mapped[uid] = atoms
            frames[label] = mapped
        common = set.intersection(*(set(mapped) for mapped in frames.values()))
        if not common:
            raise ValueError("TRECI files have no geometry common to every condition")
        self._geometry_uids = tuple(sorted(common))
        self._frames = frames

        fully_charge_labeled = sum(
            all(
                "NELECT" in frames[label][uid].info
                and "U_vs_SHE" in frames[label][uid].info
                and "initial_charges" in frames[label][uid].arrays
                for label, _, _ in _CONDITIONS
            )
            for uid in self._geometry_uids
        )
        self.capabilities = DatasetCapabilityManifest(
            name="TRECI Cu/water common-geometry potential matrix",
            version="git 24a0fe816e8aa93b6b604df9aa354806a9867cf8",
            source="https://github.com/michelegiovannibianchi/TRECI",
            license="MIT",
            labels=(
                "energy",
                "free_energy_unverified_semantics",
                "forces",
                "stress",
                "target_potential_vs_she",
                "electron_count_optional",
                "actual_potential_vs_she_optional",
                "bader_charge_optional",
            ),
            can_validate=(
                "same_geometry_cross_potential_energy_force",
                "fixed_potential_conditioning",
                "electron_count_response_on_fully_labeled_subset",
            ),
            cannot_validate=(
                "unambiguous_electrode_free_charge",
                "grand_potential_without_double_reference_audit",
                "constant_potential_kkt_without_control_mapping_audit",
            ),
            notes=(
                "%d geometries have all eight conditions." % len(self._geometry_uids),
                "%d geometries have NELECT, actual potential, and Bader charge in all conditions."
                % fully_charge_labeled,
                "PZC is a reference ensemble, not an eighth fixed numerical potential.",
                "Bader group charges are partition-dependent proxies.",
            ),
        )

    def __len__(self) -> int:
        return len(self._geometry_uids)

    def get_n_atoms(self, index: int) -> int:
        return len(self._frames["PZC_label"][self._geometry_uids[index]])

    def __getitem__(self, index: int) -> AtomsData:
        uid = self._geometry_uids[index]
        reference = self._frames["PZC_label"][uid]
        n_atoms = len(reference)
        atoms = {
            properties.atomic_numbers: torch.as_tensor(reference.numbers, dtype=torch.long),
            properties.positions: torch.as_tensor(reference.positions, dtype=self.default_dtype),
            properties.cell: torch.as_tensor(reference.cell.array, dtype=self.default_dtype),
            properties.pbc: torch.as_tensor(reference.pbc, dtype=torch.bool).reshape(1, 3),
            properties.n_atoms: torch.tensor([n_atoms], dtype=torch.long),
            properties.image_idx: torch.zeros(n_atoms, dtype=torch.long),
        }
        for transform in self.transforms:
            atoms = transform(atoms)

        conditions = []
        symbols = np.asarray(reference.get_chemical_symbols())
        copper = np.flatnonzero(symbols == "Cu")
        electrolyte = np.flatnonzero(symbols != "Cu")
        for label, target_potential, relative_path in _CONDITIONS:
            frame = self._frames[label][uid]
            if (
                len(frame) != n_atoms
                or not np.array_equal(frame.numbers, reference.numbers)
                or not np.allclose(frame.positions, reference.positions, rtol=0, atol=1.0e-10)
                or not np.allclose(frame.cell.array, reference.cell.array, rtol=0, atol=1.0e-7)
            ):
                raise ValueError("TRECI paired geometry %s is inconsistent across conditions" % uid)

            if target_potential is None:
                boundary = {
                    "schema_version": "1.0",
                    "ensemble": "potential_of_zero_charge_reference",
                    "variable": {"name": "charge", "kind": "charge", "components": 1, "unit": "e"},
                    "compile_status": "reference_only",
                }
            else:
                boundary = {
                    "schema_version": "1.0",
                    "ensemble": "electronically_grand_canonical",
                    "variable": {"name": "charge", "kind": "charge", "components": 1, "unit": "e"},
                    "controls": [
                        {
                            "observable": {"kind": "electron_reservoir", "group": "system"},
                            "target": target_potential,
                            "unit": "V_vs_SHE",
                            "reference": "SHE",
                        }
                    ],
                    "compile_status": "requires_double_reference_sign_and_offset_audit",
                }
            condition_labels = {
                properties.condition_energy: torch.tensor(
                    [frame.calc.results["energy"]], dtype=self.default_dtype
                ),
                properties.condition_free_energy: torch.tensor(
                    [frame.calc.results["free_energy"]], dtype=self.default_dtype
                ),
                properties.condition_forces: torch.as_tensor(
                    frame.calc.results["forces"], dtype=self.default_dtype
                ),
                properties.condition_stress: torch.as_tensor(
                    frame.calc.results["stress"], dtype=self.default_dtype
                ).reshape(-1),
            }
            units = {
                properties.condition_energy: "eV",
                properties.condition_free_energy: "eV",
                properties.condition_forces: "eV/angstrom",
                properties.condition_stress: "eV/angstrom^3",
            }
            metadata = {
                properties.condition_energy: {
                    "energy_kind": "TRECI source energy",
                    "reference": "Double Reference dataset",
                    "semantic_status": "requires thermodynamic audit",
                },
                properties.condition_free_energy: {
                    "energy_kind": "TRECI source free_energy",
                    "reference": "Double Reference dataset",
                    "semantic_status": "unverified; do not assume ACVR grand potential",
                },
                properties.condition_forces: {
                    "semantic_status": "source DFT force label"
                },
            }
            if "U_vs_SHE" in frame.info:
                condition_labels[properties.electrode_potential] = torch.tensor(
                    [frame.info["U_vs_SHE"]], dtype=self.default_dtype
                )
                units[properties.electrode_potential] = "V_vs_SHE"
                metadata[properties.electrode_potential] = {
                    "reference": "SHE",
                    "definition": "TRECI Double Reference output",
                }
            if "initial_charges" in frame.arrays:
                bader = torch.as_tensor(frame.arrays["initial_charges"], dtype=self.default_dtype)
                condition_labels[properties.variable_value] = bader.reshape(-1, 1)
                condition_labels[properties.group_charge] = torch.stack(
                    (bader.sum(), bader[copper].sum(), bader[electrolyte].sum())
                )
                units[properties.variable_value] = "e"
                units[properties.group_charge] = "e"
                metadata[properties.variable_value] = {"partition": "Bader", "role": "auxiliary"}
                metadata[properties.group_charge] = {"partition": "Bader", "role": "proxy"}
            electron_count = frame.info.get("NELECT")
            neutral_count = frame.info.get("NELECT0")
            net_charge = None
            if electron_count is not None and neutral_count is not None:
                net_charge = torch.tensor(neutral_count - electron_count, dtype=torch.float64)
            conditions.append(
                VariationalCondition(
                    condition_uid=label,
                    electron_count=None
                    if electron_count is None
                    else torch.tensor(electron_count, dtype=torch.float64),
                    electron_count_convention=None
                    if electron_count is None
                    else "VASP_valence_electron_count",
                    net_charge=net_charge,
                    boundary_spec=boundary,
                    boundary_hash=canonical_boundary_hash(boundary),
                    gauge={
                        "thermodynamic_control_gauge": {
                            "kind": "external_reference",
                            "reference": "SHE",
                        },
                        "electrostatic_potential_gauge": {
                            "kind": "double_reference_method",
                            "audit_status": "pending",
                        },
                        "dipole_polarization_gauge": {"kind": "not_labeled"},
                    },
                    labels=condition_labels,
                    label_units=units,
                    label_metadata=metadata,
                    provenance={
                        "source_file": relative_path,
                        "nominal_target_potential_v_vs_she": target_potential,
                        "neutral_electron_count": neutral_count,
                        "excess_charge_convention": "NELECT0_minus_NELECT",
                    },
                )
            )

        group_indices = np.concatenate((np.arange(n_atoms), copper, electrolyte))
        payload = VariationalData(
            schema_version="1.0",
            geometry_uid=uid,
            geometry_hash=uid,
            variable_spec={
                "name": "charge",
                "kind": "charge",
                "components": 1,
                "unit": "e",
                "conjugate_unit": "eV/e",
            },
            group_member_ptr=torch.tensor(
                [0, n_atoms, n_atoms + len(copper), n_atoms + len(copper) + len(electrolyte)],
                dtype=torch.long,
            ),
            group_atom_index=torch.as_tensor(group_indices, dtype=torch.long),
            group_names=("system", "electrode", "electrolyte"),
            group_roles=("system", "metal_electrode", "electrolyte"),
            conditions=tuple(conditions),
            provenance={
                "dataset": "TRECI",
                "git_commit": "24a0fe816e8aa93b6b604df9aa354806a9867cf8",
                "geometry_pairing": "exact atomic numbers and position bytes",
            },
        )
        payload.validate(n_atoms)
        return AtomsData(
            atoms=atoms,
            targets={},
            task=self.task,
            weight=self.weight,
            meta={"dataset": "TRECI", "geometry_uid": uid},
            variational=payload,
        )
