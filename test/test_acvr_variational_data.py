import json
import os

import h5py
import numpy as np
import pytest
import torch
import apsw
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from curator.data import (
    ElectronicStatesDataset,
    TreciDataset,
    treci_geometry_split,
    VariationalCondition,
    VariationalData,
    VariationalResponse,
    canonical_boundary_hash,
    collate_atoms_data,
    collate_variational_data,
    properties,
)
from curator.data.sql_database import QMDatabase, Sqlite3Dataset
from curator.data.treci import _CONDITIONS


def _dataset(path):
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = "curator.electronic_states"
        handle.attrs["schema_version"] = "1.0"
        handle.attrs["generator"] = "unit-test"
        handle.attrs["electrostatic_convention"] = json.dumps(
            {"dimension": 3, "k_point": "Gamma", "exchange_divergence": "ewald"}
        )
        geometries = handle.create_group("geometries")
        for geometry_index, charges in enumerate(((0, 1), (-1,))):
            geometry = geometries.create_group("geometry-%d" % geometry_index)
            geometry.attrs["geometry_id"] = "geometry-%d" % geometry_index
            for name, value, unit in (
                ("atomic_numbers", [1, 1], "atomic_number"),
                ("positions", [[0, 0, 0], [0, 0, 0.7 + 0.1 * geometry_index]], "angstrom"),
                ("cell", np.eye(3) * 6, "angstrom"),
                ("pbc", [True, True, True], "dimensionless"),
            ):
                dataset = geometry.create_dataset(name, data=value)
                dataset.attrs["units"] = unit
            conditions = geometry.create_group("conditions")
            for charge in charges:
                condition = conditions.create_group("q%+d" % charge)
                condition.attrs["scf_converged"] = True
                condition.attrs["solver"] = "RKS" if charge == 0 else "UKS"
                for name, value, unit in (
                    ("energy", [float(charge)], "eV"),
                    ("forces", np.zeros((2, 3)), "eV/angstrom"),
                    ("total_charge", [charge], "elementary_charge"),
                    ("electron_count", [2 - charge], "electron"),
                    ("spin", [abs(charge)], "n_alpha_minus_n_beta"),
                    ("external_field", [0, 0, 0], "V/angstrom"),
                ):
                    dataset = condition.create_dataset(name, data=value)
                    dataset.attrs["units"] = unit


def _condition(uid, labels, responses=()):
    boundary = {"variable": "charge", "equalities": [{"target": 0.0}]}
    return VariationalCondition(
        condition_uid=uid,
        electron_count=torch.tensor(2.0, dtype=torch.float64),
        electron_count_convention="all_electron",
        net_charge=torch.tensor(0.0, dtype=torch.float64),
        boundary_spec=boundary,
        boundary_hash=canonical_boundary_hash(boundary),
        gauge={"electrostatic_potential_gauge": {"kind": "differences_only"}},
        labels=labels,
        label_units={key: "eV" for key in labels},
        responses=responses,
    )


def _payload(uid, conditions, groups=((0, 1),)):
    members = [index for group in groups for index in group]
    ptr = [0]
    for group in groups:
        ptr.append(ptr[-1] + len(group))
    return VariationalData(
        schema_version="1.0",
        geometry_uid=uid,
        geometry_hash=uid + "-hash",
        variable_spec={"kind": "charge", "components": 1, "unit": "e"},
        group_member_ptr=torch.tensor(ptr, dtype=torch.long),
        group_atom_index=torch.tensor(members, dtype=torch.long),
        group_names=tuple("group-%d" % index for index in range(len(groups))),
        group_roles=tuple("system" for _ in groups),
        conditions=tuple(conditions),
    )


def _treci_dataset(path):
    for condition_index, (label, potential, relative_path) in enumerate(_CONDITIONS):
        frames = []
        for geometry_index in range(3):
            atoms = Atoms(
                ["Cu", "O"],
                positions=[[0, 0, 0], [0, 0, 1.8 + 0.1 * geometry_index]],
                cell=np.eye(3) * 8,
                pbc=True,
            )
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=float(condition_index + geometry_index),
                free_energy=float(condition_index + geometry_index) - 0.1,
                forces=np.zeros((2, 3)),
                stress=np.zeros(6),
            )
            atoms.info["head"] = label
            if geometry_index == 0:
                atoms.info["NELECT0"] = 20.0
                atoms.info["NELECT"] = 20.0 + 0.1 * condition_index
                atoms.info["U_vs_SHE"] = 0.1 if potential is None else potential + 0.01
                atoms.set_initial_charges([-0.1 * condition_index, 0.0])
            frames.append(atoms)
        output = path / relative_path
        output.parent.mkdir(parents=True, exist_ok=True)
        write(output, frames, format="extxyz")


def test_electronic_states_reader_groups_conditions_by_geometry(tmp_path):
    path = tmp_path / "states.h5"
    _dataset(path)
    dataset = ElectronicStatesDataset(str(path), compute_neighbor_list=False)

    assert len(dataset) == 2
    first = dataset[0]
    assert first.targets == {}
    assert first.variational.geometry_uid == "geometry-0"
    assert len(first.variational.conditions) == 2
    assert [float(item.net_charge) for item in first.variational.conditions] == [0.0, 1.0]
    dataset.capabilities.require_claims(["same_geometry_cross_charge"])
    with pytest.raises(ValueError, match="cannot validate"):
        dataset.capabilities.require_claims(["constant_electrode_potential"])


def test_variational_batch_keeps_geometry_once_and_flattens_conditions(tmp_path):
    path = tmp_path / "states.h5"
    _dataset(path)
    dataset = ElectronicStatesDataset(str(path), compute_neighbor_list=False)
    batch = collate_atoms_data([dataset[0], dataset[1]])
    variational = batch["variational"]

    assert batch[properties.positions].shape == (4, 3)
    assert variational["condition_ptr"].tolist() == [0, 2, 3]
    assert variational["condition_geometry_index"].tolist() == [0, 0, 1]
    assert variational["condition_atom_ptr"].tolist() == [0, 2, 4, 6]
    assert variational["condition_state_ptr"].tolist() == [0, 2, 4, 6]
    assert variational["condition_group_ptr"].tolist() == [0, 1, 2, 3]
    assert variational["geometry_group_ptr"].tolist() == [0, 1, 2]
    assert variational["group_atom_index"].tolist() == [0, 1, 2, 3]
    energy = variational["labels"][properties.condition_energy]
    assert energy["present"].tolist() == [True, True, True]
    assert energy["ptr"].tolist() == [0, 1, 2, 3]


def test_ragged_labels_responses_and_true_zero_are_distinct_from_missing():
    response = VariationalResponse(
        input_ref="control:left_potential",
        output_ref="group_charge:left",
        value=torch.tensor([[0.0], [2.0]], dtype=torch.float64),
        unit="e/V",
        method="finite_difference",
    )
    first = _condition("first", {"group_potential": torch.tensor([0.0])}, (response,))
    second = _condition("second", {})
    third = _condition("third", {"group_potential": torch.tensor([1.0, -1.0])})
    batch = collate_variational_data(
        [_payload("a", (first, second)), _payload("b", (third,), groups=((0,), (1,)))],
        [2, 2],
    )

    label = batch["labels"]["group_potential"]
    assert label["present"].tolist() == [True, False, True]
    assert label["ptr"].tolist() == [0, 1, 1, 3]
    assert label["values"].tolist() == [0.0, 1.0, -1.0]
    assert batch["condition_response_ptr"].tolist() == [0, 1, 1, 1]
    assert batch["response_value_ptr"].tolist() == [0, 2]
    assert batch["response_shape"].tolist() == [[2, 1]]


def test_invalid_boundary_hash_and_charged_dipole_without_origin_fail_closed():
    condition = _condition("bad-hash", {})
    object.__setattr__(condition, "boundary_hash", "not-the-hash")
    with pytest.raises(ValueError, match="boundary_hash"):
        condition.validate(2)

    boundary = {"variable": "charge", "equalities": [{"target": 1.0}]}
    charged = VariationalCondition(
        condition_uid="charged",
        electron_count=torch.tensor(1.0, dtype=torch.float64),
        electron_count_convention="all_electron",
        net_charge=torch.tensor(1.0, dtype=torch.float64),
        boundary_spec=boundary,
        boundary_hash=canonical_boundary_hash(boundary),
        gauge={},
        labels={"dipole": torch.zeros(3)},
        label_units={"dipole": "e*angstrom"},
    )
    with pytest.raises(ValueError, match="origin"):
        charged.validate(2)


def test_variational_sqlite_roundtrip_preserves_float64_and_legacy_reader(tmp_path):
    path = tmp_path / "states.db"
    db = QMDatabase(
        str(path), flags=apsw.SQLITE_OPEN_READWRITE | apsw.SQLITE_OPEN_CREATE
    )
    db.add_data(
        {
            properties.atomic_numbers: np.array([1, 1]),
            properties.positions: np.array([[0, 0, 0], [0, 0, 0.7]]),
            properties.pbc: True,
            properties.cell: np.eye(3) * 6,
        }
    )
    value = torch.tensor([np.pi, np.nextafter(np.pi, np.inf)], dtype=torch.float64)
    response = VariationalResponse(
        input_ref="control:potential",
        output_ref="group_charge:system",
        value=value.reshape(1, 2),
        unit="e/V",
        method="analytic",
    )
    condition = _condition("q0", {"exact64": value}, (response,))
    payload = _payload("sqlite-geometry", (condition,))

    db.ensure_acvr_schema()
    db.ensure_acvr_schema()
    db.add_variational_data(0, payload)
    loaded = db.get_variational_data(0)
    assert torch.equal(loaded.conditions[0].labels["exact64"], value)
    assert torch.equal(loaded.conditions[0].responses[0].value, response.value)
    assert loaded.conditions[0].boundary_hash == condition.boundary_hash
    with pytest.raises(ValueError, match="already exist"):
        db.add_variational_data(0, payload)
    assert db.get_variational_data(0).geometry_uid == "sqlite-geometry"

    dataset = Sqlite3Dataset(str(path), compute_neighbor_list=False, default_dtype=torch.float64)
    assert dataset[0].variational.geometry_uid == "sqlite-geometry"
    assert torch.equal(dataset[0].variational.conditions[0].labels["exact64"], value)


def test_treci_reader_pairs_potential_conditions_and_masks_optional_charge(tmp_path):
    _treci_dataset(tmp_path)
    dataset = TreciDataset(str(tmp_path), compute_neighbor_list=False, default_dtype=torch.float64)

    assert len(dataset) == 3
    samples = [dataset[index] for index in range(len(dataset))]
    labeled = next(
        sample for sample in samples if sample.variational.conditions[0].electron_count is not None
    )
    assert len(labeled.variational.conditions) == 8
    assert labeled.variational.conditions[0].boundary_spec["compile_status"] == "reference_only"
    assert labeled.variational.conditions[1].boundary_spec["controls"][0]["target"] == -0.5
    assert float(labeled.variational.conditions[-1].net_charge) == pytest.approx(-0.7)
    assert labeled.variational.group_names == ("system", "electrode", "electrolyte")

    batch = collate_atoms_data(samples)["variational"]
    assert batch["condition_ptr"].tolist() == [0, 8, 16, 24]
    variable = batch["labels"][properties.variable_value]
    assert sum(variable["present"].tolist()) == 8
    assert sum(batch["electron_count_present"].tolist()) == 8
    dataset.capabilities.require_claims(["same_geometry_cross_potential_energy_force"])
    with pytest.raises(ValueError, match="cannot validate"):
        dataset.capabilities.require_claims(["unambiguous_electrode_free_charge"])
    split = treci_geometry_split(dataset, validation_fraction=0.25, test_fraction=0.25)
    assert sorted(split["train"] + split["validation"] + split["test"]) == [0, 1, 2]
    assert not (set(split["train"]) & set(split["validation"]))


@pytest.mark.skipif(
    "CURATOR_ACVR_PYSCF_STATES" not in os.environ,
    reason="periodic PySCF multi-charge artifact was not provided",
)
def test_official_pyscf_multicharge_artifact():
    dataset = ElectronicStatesDataset(
        os.environ["CURATOR_ACVR_PYSCF_STATES"], compute_neighbor_list=False
    )
    assert len(dataset) == 3
    assert all(len(dataset[index].variational.conditions) == 3 for index in range(3))


@pytest.mark.skipif(
    "CURATOR_TRECI_DATASET" not in os.environ,
    reason="TRECI Constant_potential/Training/Dataset was not provided",
)
def test_official_treci_common_geometry_artifact():
    dataset = TreciDataset(os.environ["CURATOR_TRECI_DATASET"], compute_neighbor_list=False)
    assert len(dataset) == 236
    fully_labeled = sum(
        all(
            condition.electron_count is not None
            and properties.variable_value in condition.labels
            for condition in dataset[index].variational.conditions
        )
        for index in range(len(dataset))
    )
    assert fully_labeled == 15
    split = treci_geometry_split(dataset)
    assert {key: len(value) for key, value in split.items()} == {
        "train": 188,
        "validation": 24,
        "test": 24,
    }
