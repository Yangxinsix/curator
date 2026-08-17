"""Reader for geometry-grouped, multi-condition electronic-structure data."""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from . import properties
from ._neighborlist import NeighborListTransform, TorchNeighborList
from ._transform import Transform
from .atoms_data import AtomsData
from .capabilities import DatasetCapabilityManifest
from .variational_data import (
    VariationalCondition,
    VariationalData,
    canonical_boundary_hash,
    geometry_hash,
)


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _require_unit(dataset, expected: str) -> None:
    actual = _text(dataset.attrs.get("units", ""))
    if actual != expected:
        raise ValueError("%s uses unit %r; expected %r" % (dataset.name, actual, expected))


class ElectronicStatesDataset(torch.utils.data.Dataset):
    """Return one geometry with all of its electronic/boundary conditions."""

    def __init__(
        self,
        datapath: str,
        cutoff: Optional[float] = None,
        compute_neighbor_list: bool = True,
        transforms: Optional[List[Transform]] = None,
        default_dtype: torch.dtype = torch.get_default_dtype(),
        task: str = "electronic_states",
        weight: float = 1.0,
    ) -> None:
        try:
            import h5py
        except ImportError as error:
            raise ImportError("ElectronicStatesDataset requires h5py") from error
        self.datapath = str(Path(datapath).expanduser().resolve())
        if not Path(self.datapath).is_file():
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
        self._file = None

        with h5py.File(self.datapath, "r") as handle:
            if _text(handle.attrs.get("schema", "")) != "curator.electronic_states":
                raise ValueError("not a curator.electronic_states HDF5 file")
            if _text(handle.attrs.get("schema_version", "")) != "1.0":
                raise ValueError("unsupported electronic-states schema version")
            if "geometries" not in handle:
                raise ValueError("electronic-states file is missing /geometries")
            self._geometry_ids = tuple(sorted(handle["geometries"].keys()))
            if not self._geometry_ids:
                raise ValueError("electronic-states file contains no geometries")
            for geometry_id in self._geometry_ids:
                self._validate_geometry(handle["geometries"][geometry_id])
            source = _text(handle.attrs.get("generator", "unknown"))
            version = _text(handle.attrs["schema_version"])

        self.capabilities = DatasetCapabilityManifest(
            name="Curator multi-condition electronic states",
            version=version,
            source=source,
            license="generated locally; source-method license applies",
            labels=(
                "energy",
                "forces",
                "total_charge",
                "electron_count",
                "spin",
            ),
            can_validate=("same_geometry_cross_charge", "periodic_fixed_total_charge"),
            cannot_validate=(
                "absolute_electrode_potential",
                "constant_electrode_potential",
                "electrode_capacitance",
            ),
            notes=(
                "Samples are split by geometry; conditions never cross split boundaries.",
                "Charged periodic energies depend on the recorded cell and electrostatic convention.",
            ),
        )

    @staticmethod
    def _validate_geometry(group) -> None:
        required = {"atomic_numbers", "positions", "cell", "pbc", "conditions"}
        missing = sorted(required - set(group.keys()))
        if missing:
            raise ValueError("geometry %s is missing %s" % (group.name, missing))
        _require_unit(group["atomic_numbers"], "atomic_number")
        _require_unit(group["positions"], "angstrom")
        _require_unit(group["cell"], "angstrom")
        _require_unit(group["pbc"], "dimensionless")
        n_atoms = group["atomic_numbers"].shape[0]
        if group["positions"].shape != (n_atoms, 3) or group["cell"].shape != (3, 3):
            raise ValueError("geometry %s has inconsistent shapes" % group.name)
        if not group["conditions"]:
            raise ValueError("geometry %s has no conditions" % group.name)
        for condition in group["conditions"].values():
            required_condition = {
                "energy",
                "forces",
                "total_charge",
                "electron_count",
                "spin",
                "external_field",
            }
            missing = sorted(required_condition - set(condition.keys()))
            if missing:
                raise ValueError("condition %s is missing %s" % (condition.name, missing))
            for key, unit in {
                "energy": "eV",
                "forces": "eV/angstrom",
                "total_charge": "elementary_charge",
                "electron_count": "electron",
                "spin": "n_alpha_minus_n_beta",
                "external_field": "V/angstrom",
            }.items():
                _require_unit(condition[key], unit)
            if condition["forces"].shape != (n_atoms, 3):
                raise ValueError("condition %s has inconsistent force shape" % condition.name)
            if not bool(condition.attrs.get("scf_converged", False)):
                raise ValueError("condition %s is not marked SCF-converged" % condition.name)

    def _handle(self):
        if self._file is None:
            import h5py

            self._file = h5py.File(self.datapath, "r")
        return self._file

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __del__(self):
        handle = getattr(self, "_file", None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self._geometry_ids)

    def get_n_atoms(self, index: int) -> int:
        geometry_id = self._geometry_ids[index]
        return int(self._handle()["geometries"][geometry_id]["atomic_numbers"].shape[0])

    def __getitem__(self, index: int) -> AtomsData:
        geometry_id = self._geometry_ids[index]
        root = self._handle()
        group = root["geometries"][geometry_id]
        atomic_numbers = torch.as_tensor(np.asarray(group["atomic_numbers"]), dtype=torch.long)
        source_positions = torch.as_tensor(np.asarray(group["positions"]), dtype=torch.float64)
        source_cell = torch.as_tensor(np.asarray(group["cell"]), dtype=torch.float64)
        source_pbc = torch.as_tensor(np.asarray(group["pbc"]), dtype=torch.bool)
        n_atoms = atomic_numbers.numel()
        atoms = {
            properties.atomic_numbers: atomic_numbers,
            properties.positions: source_positions.to(self.default_dtype),
            properties.cell: source_cell.to(self.default_dtype),
            properties.pbc: source_pbc.reshape(1, 3),
            properties.n_atoms: torch.tensor([n_atoms], dtype=torch.long),
            properties.image_idx: torch.zeros(n_atoms, dtype=torch.long),
        }
        for transform in self.transforms:
            atoms = transform(atoms)

        conditions = []
        electrostatic_convention = json.loads(root.attrs["electrostatic_convention"])
        condition_ids = sorted(
            group["conditions"].keys(),
            key=lambda name: int(group["conditions"][name].attrs.get("condition_index", 0)),
        )
        for condition_id in condition_ids:
            condition = group["conditions"][condition_id]
            charge = float(np.asarray(condition["total_charge"]).reshape(-1)[0])
            boundary = {
                "schema_version": "1.0",
                "variable": {"name": "charge", "kind": "charge", "components": 1, "unit": "e"},
                "equalities": [
                    {
                        "observable": {"kind": "group_sum", "group": "system"},
                        "target": charge,
                        "unit": "e",
                    }
                ],
                "controls": [],
                "environments": [],
            }
            gauge = {
                "thermodynamic_control_gauge": {"kind": "fixed_total_charge"},
                "electrostatic_potential_gauge": {
                    "kind": "periodic_ewald",
                    "reference": "recorded PySCF convention",
                },
                "dipole_polarization_gauge": {"kind": "not_labeled"},
            }
            attrs = {key: _text(value) for key, value in condition.attrs.items()}
            attrs["electrostatic_convention"] = electrostatic_convention
            attrs["external_field_v_per_angstrom"] = np.asarray(
                condition["external_field"]
            ).tolist()
            attrs["energy_kind"] = "fixed-electron-number periodic DFT total energy"
            conditions.append(
                VariationalCondition(
                    condition_uid=condition_id,
                    electron_count=torch.tensor(
                        float(np.asarray(condition["electron_count"]).reshape(-1)[0]),
                        dtype=torch.float64,
                    ),
                    electron_count_convention="valence_electrons_gth_pseudopotential",
                    net_charge=torch.tensor(charge, dtype=torch.float64),
                    boundary_spec=boundary,
                    boundary_hash=canonical_boundary_hash(boundary),
                    gauge=gauge,
                    labels={
                        properties.condition_energy: torch.as_tensor(
                            np.asarray(condition["energy"]), dtype=self.default_dtype
                        ),
                        properties.condition_forces: torch.as_tensor(
                            np.asarray(condition["forces"]), dtype=self.default_dtype
                        ),
                    },
                    label_units={
                        properties.condition_energy: "eV",
                        properties.condition_forces: "eV/angstrom",
                    },
                    provenance=attrs,
                )
            )

        payload = VariationalData(
            schema_version="1.0",
            geometry_uid=geometry_id,
            geometry_hash=geometry_hash(atomic_numbers, source_positions, source_cell, source_pbc),
            variable_spec={
                "name": "charge",
                "kind": "charge",
                "components": 1,
                "unit": "e",
                "conjugate_unit": "eV/e",
            },
            group_member_ptr=torch.tensor([0, n_atoms], dtype=torch.long),
            group_atom_index=torch.arange(n_atoms, dtype=torch.long),
            group_names=("system",),
            group_roles=("system",),
            conditions=tuple(conditions),
            provenance={
                "dataset": "PySCF periodic H2 multi-charge oracle",
                "source_file": self.datapath,
            },
        )
        payload.validate(n_atoms)
        return AtomsData(
            atoms=atoms,
            targets={},
            task=self.task,
            weight=self.weight,
            meta={"geometry_id": geometry_id},
            variational=payload,
        )
