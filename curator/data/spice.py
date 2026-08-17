"""Lazy reader for official SPICE 2.x HDF5 downloads."""

import bisect
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from ase import units

from . import properties
from ._neighborlist import NeighborListTransform, TorchNeighborList
from ._transform import Transform
from .atoms_data import AtomsData, atoms_data_from_dict
from .capabilities import DatasetCapabilityManifest


_SUBSET_ALIASES = {
    "ion_pairs": "ion pairs",
    "dipeptides": "dipeptides",
    "water_clusters": "water clusters",
    "solvated_amino_acids": "solvated amino acids",
    "solvated_pubchem": "solvated pubchem",
}


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _normalized_subsets(subsets: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if subsets is None:
        return ()
    return tuple(
        _SUBSET_ALIASES.get(str(value).casefold(), str(value).casefold())
        for value in subsets
    )


def _normalized_unit(value) -> str:
    return _decode(value).strip().casefold().replace(" ", "_")


def _validate_unit(dataset, expected: Tuple[str, ...], group_name: str) -> None:
    """Reject explicit unit metadata that would make conversion unsafe.

    Some historical SPICE files omit unit attributes.  Those files follow the
    documented atomic-unit schema, so absence remains accepted.  An explicit
    incompatible attribute is never silently interpreted as an atomic unit.
    """

    if "units" not in dataset.attrs:
        return
    actual = _normalized_unit(dataset.attrs["units"])
    allowed = tuple(_normalized_unit(value) for value in expected)
    if actual not in allowed:
        raise ValueError(
            "SPICE group %s field %s has unsupported units %s; expected %s"
            % (group_name, dataset.name.rsplit("/", 1)[-1], actual, ", ".join(allowed))
        )


def _formal_charge_from_smiles(smiles: str) -> int:
    """Return the integer formal charge encoded in bracketed SMILES atoms."""

    charge = 0
    for atom in re.findall(r"\[([^]]+)\]", smiles):
        atom = atom.rsplit(":", 1)[0]
        matches = re.findall(r"([+-]+)(\d*)", atom)
        if not matches:
            continue
        signs, magnitude = matches[-1]
        if "+" in signs and "-" in signs:
            raise ValueError("invalid mixed-sign formal charge in SMILES")
        value = int(magnitude) if magnitude else len(signs)
        charge += value if signs[0] == "+" else -value
    return charge


def ion_pair_distance_split(
    dataset: "SpiceDataset",
    train_max_distance: float = 5.5,
    validation_max_distance: float = 6.5,
) -> Dict[str, List[int]]:
    """Split two-atom SPICE ion pairs by separation in angstrom.

    The returned indices address the flattened :class:`SpiceDataset`, so the
    result can be serialized directly as an ``AtomsDataModule.split_file``.
    Training contains distances ``<= train_max_distance``, validation contains
    the open/closed interval up to ``validation_max_distance``, and test holds
    the longer-range extrapolation configurations.
    """

    if not isinstance(dataset, SpiceDataset):
        raise TypeError("dataset must be a SpiceDataset")
    train_max_distance = float(train_max_distance)
    validation_max_distance = float(validation_max_distance)
    if not (
        np.isfinite(train_max_distance)
        and np.isfinite(validation_max_distance)
        and 0.0 < train_max_distance < validation_max_distance
    ):
        raise ValueError(
            "distance thresholds must be finite and satisfy "
            "0 < train_max_distance < validation_max_distance"
        )
    if np.any(dataset._atom_counts != 2):
        raise ValueError("ion-pair distance splitting requires exactly two atoms per group")

    split = {"train": [], "validation": [], "test": []}
    handle = dataset._handle()
    offset = 0
    for group_name, count in zip(dataset._group_names, dataset._counts):
        count = int(count)
        conformations = np.arange(count, dtype=np.int64) * dataset._stride
        positions = np.asarray(handle[group_name]["conformations"][conformations])
        distances = np.linalg.norm(positions[:, 1] - positions[:, 0], axis=1) * units.Bohr
        for local_index, distance in enumerate(distances):
            index = offset + local_index
            if distance <= train_max_distance:
                split["train"].append(index)
            elif distance <= validation_max_distance:
                split["validation"].append(index)
            else:
                split["test"].append(index)
        offset += count

    if any(not indices for indices in split.values()):
        raise ValueError("distance thresholds produced an empty train/validation/test split")
    return split


class SpiceDataset(torch.utils.data.Dataset):
    """Read selected SPICE molecules and conformations without loading all data.

    The official file stores positions in bohr, energies in hartree, and energy
    gradients in hartree/bohr.  Samples returned here use Curator units:
    angstrom, eV, and eV/angstrom, with forces equal to minus the gradient.

    Total molecular charge is read from formal charges in the mapped SMILES,
    without adding RDKit as a runtime dependency.  When MBIS charges exist,
    their rounded sum is cross-checked against that formal charge; the atomic
    MBIS values remain optional, partition-dependent auxiliary labels.
    """

    def __init__(
        self,
        datapath: str,
        cutoff: Optional[float] = None,
        compute_neighbor_list: bool = True,
        transforms: Optional[List[Transform]] = None,
        default_dtype: torch.dtype = torch.get_default_dtype(),
        task: str = "spice",
        weight: float = 1.0,
        subsets: Optional[Sequence[str]] = None,
        energy_key: str = "formation_energy",
        include_atomic_charges: bool = True,
        include_dipole: bool = True,
        require_forces: bool = True,
        require_total_charge: bool = True,
        charge_tolerance: float = 5.0e-3,
        max_conformations_per_molecule: Optional[int] = None,
        stride: int = 1,
        return_atoms_data: bool = True,
    ) -> None:
        super().__init__()
        if stride <= 0:
            raise ValueError("stride must be positive")
        if max_conformations_per_molecule is not None and max_conformations_per_molecule <= 0:
            raise ValueError("max_conformations_per_molecule must be positive")
        try:
            import h5py
        except ImportError as error:
            raise ImportError("SpiceDataset requires the optional h5py package") from error

        self.datapath = str(Path(datapath).expanduser().resolve())
        if not Path(self.datapath).is_file():
            raise FileNotFoundError(self.datapath)
        self.cutoff = cutoff
        self.compute_neighbor_list = compute_neighbor_list
        self.transforms = [] if transforms is None else list(transforms)
        if compute_neighbor_list:
            if cutoff is None:
                raise ValueError("cutoff is required when computing neighbor lists")
            if not any(isinstance(item, NeighborListTransform) for item in self.transforms):
                self.transforms.append(TorchNeighborList(cutoff=float(cutoff)))
        self.default_dtype = default_dtype
        self.task = task
        self.weight = float(weight)
        self.energy_key = energy_key
        self.include_atomic_charges = include_atomic_charges
        self.include_dipole = include_dipole
        self.require_forces = require_forces
        self.require_total_charge = require_total_charge
        self.charge_tolerance = float(charge_tolerance)
        self.return_atoms_data = bool(return_atoms_data)
        self._file = None

        filters = _normalized_subsets(subsets)
        group_names = []
        counts = []
        atom_counts = []
        common_fields = None
        with h5py.File(self.datapath, "r") as handle:
            for group_name in sorted(handle.keys()):
                group = handle[group_name]
                subset = _decode(group["subset"][0]).casefold()
                if filters and not any(value in subset for value in filters):
                    continue
                self._validate_group(group, group_name)
                count = int(group["conformations"].shape[0])
                count = (count + stride - 1) // stride
                if max_conformations_per_molecule is not None:
                    count = min(count, max_conformations_per_molecule)
                if count == 0:
                    continue
                group_names.append(group_name)
                counts.append(count)
                atom_counts.append(int(group["atomic_numbers"].shape[0]))
                fields = set(group.keys())
                common_fields = fields if common_fields is None else common_fields & fields
        if not group_names:
            raise ValueError("no SPICE groups matched the requested subset filters")
        self._group_names = tuple(group_names)
        self._counts = np.asarray(counts, dtype=np.int64)
        self._atom_counts = np.asarray(atom_counts, dtype=np.int64)
        self._cumulative = np.cumsum(self._counts)
        self._stride = stride
        self.capabilities = self._build_capabilities(common_fields or set())

    def _validate_group(self, group, name: str) -> None:
        required = {"subset", "smiles", "atomic_numbers", "conformations", self.energy_key}
        if self.require_forces:
            required.add("dft_total_gradient")
        missing = sorted(required - set(group.keys()))
        if missing:
            raise ValueError(
                "SPICE group %s is missing required fields: %s"
                % (name, ", ".join(missing))
            )
        n_conformations, n_atoms, dimensions = group["conformations"].shape
        if dimensions != 3 or group["atomic_numbers"].shape != (n_atoms,):
            raise ValueError("SPICE group %s has inconsistent geometry shapes" % name)
        if group[self.energy_key].shape[0] != n_conformations:
            raise ValueError("SPICE group %s has inconsistent energy count" % name)
        if "dft_total_gradient" in group and group["dft_total_gradient"].shape != (
            n_conformations,
            n_atoms,
            3,
        ):
            raise ValueError("SPICE group %s has inconsistent gradient shape" % name)
        _validate_unit(group["conformations"], ("bohr", "a0"), name)
        _validate_unit(group[self.energy_key], ("hartree", "hartrees", "eh"), name)
        if "dft_total_gradient" in group:
            _validate_unit(
                group["dft_total_gradient"],
                ("hartree/bohr", "hartrees/bohr", "eh/a0"),
                name,
            )
        if "mbis_charges" in group:
            _validate_unit(
                group["mbis_charges"],
                ("elementary_charge", "elementary charges", "e"),
                name,
            )
        if "scf_dipole" in group:
            _validate_unit(
                group["scf_dipole"],
                (
                    "elementary_charge*bohr",
                    "elementary_charge_bohr",
                    "e*bohr",
                    "e*a0",
                ),
                name,
            )

    def _build_capabilities(self, fields: set) -> DatasetCapabilityManifest:
        labels = ["energy", "atomic_numbers", "positions"]
        if "dft_total_gradient" in fields:
            labels.append("forces")
        if self.require_total_charge:
            labels.append("total_charge")
        if "mbis_charges" in fields and self.include_atomic_charges:
            labels.append("atomic_charge_mbis")
        if "scf_dipole" in fields and self.include_dipole:
            labels.append("dipole")
        can_validate = ["isolated_molecule_energy_force", "ion_pair_distance_asymptote"]
        if "total_charge" in labels:
            can_validate.append("fixed_total_charge_conditioning")
        if "atomic_charge_mbis" in labels:
            can_validate.append("partition_charge_auxiliary")
        return DatasetCapabilityManifest(
            name="SPICE selected subset",
            version="2.x HDF5 schema",
            source="https://github.com/openmm/spice-dataset",
            license="CC0",
            labels=tuple(labels),
            can_validate=tuple(can_validate),
            cannot_validate=(
                "constant_electrode_potential",
                "electrode_capacitance",
                "same_geometry_cross_voltage_response",
                "periodic_electrochemical_interface",
            ),
            notes=tuple(
                [
                    "Charged-molecule dipoles are origin dependent.",
                ]
                + (
                    ["MBIS atomic charges are partition-dependent auxiliary labels."]
                    if "atomic_charge_mbis" in labels
                    else ["This selection does not expose atomic partition charges."]
                )
            ),
        )

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
        return int(self._cumulative[-1])

    def _locate(self, index: int) -> Tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        group_index = bisect.bisect_right(self._cumulative, index)
        previous = 0 if group_index == 0 else int(self._cumulative[group_index - 1])
        conformation = (index - previous) * self._stride
        return group_index, conformation

    def get_n_atoms(self, index: int) -> int:
        group_index, _ = self._locate(index)
        return int(self._atom_counts[group_index])

    def __getitem__(self, index: int):
        group_index, conformation = self._locate(index)
        group_name = self._group_names[group_index]
        group = self._handle()[group_name]
        atomic_numbers = torch.as_tensor(
            np.asarray(group["atomic_numbers"]), dtype=torch.long
        )
        positions = torch.as_tensor(
            np.asarray(group["conformations"][conformation]) * units.Bohr,
            dtype=self.default_dtype,
        )
        n_atoms = atomic_numbers.numel()
        sample: Dict[str, torch.Tensor] = {
            properties.atomic_numbers: atomic_numbers,
            properties.positions: positions,
            properties.n_atoms: torch.tensor([n_atoms], dtype=torch.long),
            properties.image_idx: torch.zeros(n_atoms, dtype=torch.long),
            properties.pbc: torch.zeros((1, 3), dtype=torch.bool),
            properties.energy: torch.tensor(
                [float(group[self.energy_key][conformation]) * units.Hartree],
                dtype=self.default_dtype,
            ),
        }
        if "dft_total_gradient" in group:
            gradient = np.asarray(group["dft_total_gradient"][conformation])
            sample[properties.forces] = torch.as_tensor(
                -gradient * units.Hartree / units.Bohr,
                dtype=self.default_dtype,
            )
        formal_charge = torch.tensor(
            [float(_formal_charge_from_smiles(_decode(group["smiles"][0])))],
            dtype=self.default_dtype,
        )
        if "mbis_charges" in group:
            charge = torch.as_tensor(
                np.asarray(group["mbis_charges"][conformation]).reshape(n_atoms),
                dtype=self.default_dtype,
            )
            total = charge.sum()
            rounded = total.round()
            if abs(float((total - rounded).detach().cpu())) > self.charge_tolerance:
                raise ValueError(
                    "MBIS charges for %s conformation %d do not sum near an integer"
                    % (group_name, conformation)
                )
            if abs(float((rounded - formal_charge).detach().cpu())) > 0.5:
                raise ValueError(
                    "MBIS and SMILES total charges disagree for %s" % group_name
                )
            if self.include_atomic_charges:
                sample[properties.atomic_charge] = charge
        if self.require_total_charge:
            sample[properties.total_charge] = formal_charge
        if self.include_dipole and "scf_dipole" in group:
            sample[properties.dipole] = torch.as_tensor(
                np.asarray(group["scf_dipole"][conformation]).reshape(1, 3)
                * units.Bohr,
                dtype=self.default_dtype,
            )

        for transform in self.transforms:
            sample = transform(sample)
        if not self.return_atoms_data:
            return sample
        meta = {
            "dataset": "SPICE",
            "group": group_name,
            "conformation": conformation,
            "subset": _decode(group["subset"][0]),
            "smiles": _decode(group["smiles"][0]),
            "energy_key": self.energy_key,
        }
        return atoms_data_from_dict(
            sample,
            task=self.task,
            weight=self.weight,
            meta=meta,
        )
