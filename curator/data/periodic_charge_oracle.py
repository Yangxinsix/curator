"""Small periodic charge oracle for conq/conp amortized-solver comparisons.

This is a numerical benchmark, not electronic-structure reference data. Every
arm uses the same local quadratic energy and three-dimensional tin-foil Ewald
interaction. Geometry splits are fixed before conditions are generated.
"""

from dataclasses import asdict

import torch
from torch.utils.data import Dataset

from curator.layer._ewald import EwaldSummation
from curator.variational.backends.ewald import EwaldChargeOperator
from curator.variational.spec import (
    BoundarySpec, ConstraintSpec, ControlSpec, GaugeSpec, GroupSpec, VariableSpec,
)

from . import properties
from .atoms_data import AtomsData
from .variational_data import VariationalCondition, VariationalData, canonical_boundary_hash, geometry_hash
from .variational_oracle import VariationalOracleDataModule


CHARGE_VARIABLE = VariableSpec(name="charge", kind="atomic_charge", components=1,
                               unit="e", conjugate_unit="V")
CHARGE_GAUGE = {"electrostatic_potential_gauge": {"kind": "zero_mean_group_potential"},
                "spatial_boundary": "3D_periodic_tinfoil", "energy_zero": "periodic_charge_oracle_v1"}


def periodic_charge_coefficients(atoms):
    """Fixed teacher coefficients; depend only on species and local geometry."""
    numbers = atoms[properties.atomic_numbers].to(atoms[properties.positions])
    edges, distance = atoms[properties.edge_idx], atoms[properties.edge_dist]
    density = torch.zeros_like(numbers).index_add(
        0, edges[:, 0], torch.exp(-(distance / 2.8).square()))
    chi = 1.2 * (numbers - 1.5) + 0.3 * density
    hardness = 24.0 + 0.5 * numbers + 0.2 * density
    return chi, hardness, 0.01 * density.sum()


def periodic_charge_reference(hessian, chi, constant, constraint, target, drive):
    """Independent dense KKT reference, used only when creating labels/audits."""
    count, rank = hessian.shape[0], constraint.shape[0]
    kkt = hessian.new_zeros((count + rank, count + rank))
    kkt[:count, :count] = hessian
    kkt[:count, count:] = constraint.T
    kkt[count:, :count] = constraint
    solution = torch.linalg.solve(kkt, torch.cat((drive - chi, target)))
    charge, multiplier = solution[:count], solution[count:]
    energy = constant + (chi - drive) @ charge + 0.5 * charge @ hessian @ charge
    return charge, multiplier, energy


class PeriodicChargeOracleDataset(Dataset):
    """Two perturbed periodic planes with six conditions per geometry.

    Two conq and two conp conditions span random controls in the specified
    ranges. A third condition of each type repeats the first control with a
    zero-sum drive on two right-plane sites. Those paired conditions test a
    response in the left plane without changing its local boundary inputs.
    """

    @torch.inference_mode(False)
    def __init__(self, split="train", num_train=128, num_val=32, num_test=64, seed=17,
                 plane_side=4, spacing=2.8, cell_height=16.0, jitter=0.08,
                 cutoff=3.5, charge_extent=0.5, potential_extent=1.0,
                 local_drive=0.1, ewald_alpha=0.4, ewald_accuracy=8.0):
        from ase import Atoms
        from ase.neighborlist import neighbor_list

        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if min(num_train, num_val, num_test) < 1 or plane_side < 2:
            raise ValueError("splits must be nonempty and plane_side must be at least two")
        if (spacing <= 0 or cell_height <= 0 or jitter < 0 or cutoff <= 0
                or min(charge_extent, potential_extent, local_drive) <= 0):
            raise ValueError("geometry lengths and condition ranges must be positive")
        if cutoff >= cell_height / 2 - 2 * jitter:
            raise ValueError("local cutoff must leave the two perturbed planes disconnected")
        self.split = split
        self.n_atoms = 2 * plane_side ** 2
        order = torch.randperm(num_train + num_val + num_test,
                               generator=torch.Generator().manual_seed(seed)).tolist()
        splits = {"train": order[:num_train], "val": order[num_train:num_train + num_val],
                  "test": order[num_train + num_val:]}
        self.indices = splits[split]
        ewald = EwaldSummation(alpha=ewald_alpha, acc_factor=ewald_accuracy)
        self.metadata = {
            "schema": "periodic_charge_oracle_v1", "seed": seed, "splits": splits,
            "n_atoms": self.n_atoms, "plane_side": plane_side, "spacing": spacing,
            "cell_height": cell_height, "jitter": jitter, "cutoff": cutoff,
            "num_conditions": 6, "charge_extent": charge_extent,
            "potential_extent": potential_extent, "local_drive": local_drive,
            "ewald_alpha": ewald.alpha, "ewald_cutoff": ewald.cutoff,
            "ewald_k_cutoff": ewald.k_cutoff, "ewald_accuracy": ewald_accuracy,
            "variable_spec": asdict(CHARGE_VARIABLE), "gauge": CHARGE_GAUGE,
            "units": {"positions": "Angstrom", "condition_energy": "eV",
                      "variable_value": "e", "control_derivative": "e",
                      "constraint_derivative": "V", "local_external_field": "V"},
            "teacher": "fixed local coefficients v1; Ewald; independent dense KKT",
            "stability_policy": "fixed teacher; reject non-positive reduced curvature; no retuning",
        }
        side = plane_side ** 2
        xy = torch.cartesian_prod(torch.arange(plane_side), torch.arange(plane_side)).double()
        positions = torch.cat((
            torch.cat(((xy + 0.5) * spacing, torch.full((side, 1), cell_height / 4)), dim=1),
            torch.cat(((xy + 0.5) * spacing, torch.full((side, 1), 3 * cell_height / 4)), dim=1),
        ))
        cell = torch.diag(torch.tensor([plane_side * spacing, plane_side * spacing, cell_height],
                                      dtype=torch.float64))
        groups = (GroupSpec.from_indices("left", range(side)),
                  GroupSpec.from_indices("right", range(side, self.n_atoms)), GroupSpec.all())
        group_matrix = torch.zeros((2, self.n_atoms), dtype=torch.float64)
        group_matrix[0, :side], group_matrix[1, side:] = 1, 1
        self.samples = []
        for index in self.indices:
            generator = torch.Generator().manual_seed(seed + 104729 * (index + 1))
            geometry = positions + jitter * (2 * torch.rand(positions.shape, generator=generator,
                                                           dtype=torch.float64) - 1)
            numbers = 1 + torch.randint(0, 2, (self.n_atoms,), generator=generator)
            ase_atoms = Atoms(numbers=numbers.numpy(), positions=geometry.numpy(),
                              cell=cell.numpy(), pbc=True)
            center, neighbor, shifts = neighbor_list("ijS", ase_atoms, cutoff, self_interaction=False)
            edges = torch.stack((torch.as_tensor(center), torch.as_tensor(neighbor)), dim=1).long()
            displacements = torch.as_tensor(shifts, dtype=torch.float64) @ cell
            difference = geometry[edges[:, 1]] - geometry[edges[:, 0]] + displacements
            atoms = {
                properties.positions: geometry, properties.atomic_numbers: numbers,
                properties.cell: cell.clone(), properties.pbc: torch.ones((1, 3), dtype=torch.bool),
                properties.n_atoms: torch.tensor([self.n_atoms]),
                properties.edge_idx: edges, properties.edge_diff: difference,
                properties.edge_dist: torch.linalg.vector_norm(difference, dim=-1),
                properties.cell_displacements: displacements,
                properties.n_pairs: torch.tensor([len(edges)]),
            }
            kernel = EwaldChargeOperator(ewald).geometry_kernels(atoms)[0]
            chi, hardness, constant = periodic_charge_coefficients(atoms)
            hessian = kernel + torch.diag(hardness)
            conditions = []
            for kind, extent in (("conq", charge_extent), ("conp", potential_extent)):
                amplitudes = extent * (2 * torch.rand(2, generator=generator, dtype=torch.float64) - 1)
                for condition_index, amplitude in enumerate((*amplitudes, amplitudes[0])):
                    local = torch.zeros(self.n_atoms, dtype=torch.float64)
                    if condition_index == 2:
                        local[side], local[side + 1] = local_drive, -local_drive
                    if kind == "conq":
                        constraint = group_matrix
                        target = torch.stack((amplitude, -amplitude))
                        drive = local
                        equalities = tuple(ConstraintSpec.group_sum(name + "_charge", name,
                                            float(value), unit="e")
                                           for name, value in zip(("left", "right"), target))
                        controls = ()
                    else:
                        constraint = torch.ones((1, self.n_atoms), dtype=torch.float64)
                        target = torch.zeros(1, dtype=torch.float64)
                        values = torch.stack((amplitude, -amplitude))
                        drive = group_matrix.T @ values + local
                        equalities = (ConstraintSpec.group_sum("neutrality", "all", 0.0, unit="e"),)
                        controls = tuple(ControlSpec.group_potential(name + "_potential", name,
                                          float(value), unit="V")
                                         for name, value in zip(("left", "right"), values))
                    # Curvature is checked in the actual feasible tangent space.
                    null = torch.linalg.svd(constraint, full_matrices=True).Vh[constraint.shape[0]:].T
                    curvature = torch.linalg.eigvalsh(null.T @ hessian @ null).min()
                    if not bool(curvature > 1e-8):
                        raise ValueError("periodic teacher has non-positive constrained curvature")
                    charge, multiplier, energy = periodic_charge_reference(
                        hessian, chi, constant, constraint, target, drive)
                    boundary = asdict(BoundarySpec(variable=CHARGE_VARIABLE, groups=groups,
                                       equalities=equalities, controls=controls, gauge=GaugeSpec("zero_mean")))
                    labels = {"condition_energy": energy.reshape(1), "variable_value": charge[:, None],
                              "constraint_derivative": -multiplier}
                    units = {"condition_energy": "eV", "variable_value": "e", "constraint_derivative": "V"}
                    inputs = {"local_external_field": local[:, None], "constraint_target": target}
                    if kind == "conp":
                        inputs["control_value"] = values
                        labels["control_derivative"] = -group_matrix @ charge
                        units["control_derivative"] = "e"
                    conditions.append(VariationalCondition(
                        condition_uid=f"periodic-g{index:05d}-{kind}-{condition_index}",
                        electron_count=None, electron_count_convention=None,
                        net_charge=torch.zeros(1, dtype=torch.float64),
                        boundary_spec=boundary, boundary_hash=canonical_boundary_hash(boundary),
                        gauge=CHARGE_GAUGE, labels=labels, label_units=units,
                        label_metadata={"condition_energy": {"energy_kind": "boundary_legendre_potential",
                                                               "includes_local_drive": True}},
                        provenance={"split": split, "boundary_kind": kind, "amplitude": float(amplitude),
                                    "local_drive": condition_index == 2, "minimum_curvature": float(curvature)},
                        inputs=inputs, label_eligible={"control_derivative": kind == "conp"},
                    ))
            self.samples.append(AtomsData(
                atoms=atoms, targets={}, task="periodic_charge_oracle",
                meta={"split": split, "geometry_index": index},
                variational=VariationalData(
                    schema_version="1", geometry_uid=f"periodic-charge-g{index:05d}",
                    geometry_hash=geometry_hash(numbers, geometry, cell, atoms[properties.pbc]),
                    variable_spec=asdict(CHARGE_VARIABLE),
                    group_member_ptr=torch.tensor([0, side, self.n_atoms]),
                    group_atom_index=torch.arange(self.n_atoms), group_names=("left", "right"),
                    group_roles=("electrode", "electrode"), conditions=tuple(conditions),
                    provenance={"seed": seed, "split": split, "teacher": self.metadata["teacher"]},
                ),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def get_n_atoms(self, index):
        return self.n_atoms


class PeriodicChargeOracleDataModule(VariationalOracleDataModule):
    """Reuse the existing grouped-oracle loader and checkpointed shuffle state."""

    def __init__(self, batch_size=4, num_train=128, num_val=32, num_test=64, seed=17,
                 num_workers=0, shuffle=True, default_dtype="float64", datapath=None,
                 train_path=None, heads=None, **geometry_kwargs):
        super().__init__(batch_size=batch_size, num_train=num_train, num_val=num_val,
                         num_test=num_test, seed=seed, num_workers=num_workers,
                         shuffle=shuffle, default_dtype=default_dtype, datapath=datapath,
                         train_path=train_path, heads=heads)
        self.dataset_kwargs = dict(num_train=num_train, num_val=num_val, num_test=num_test,
                                   seed=seed, **geometry_kwargs)

    def setup(self, stage=None):
        if not self.datasets:
            self.datasets = {split: PeriodicChargeOracleDataset(split=split, **self.dataset_kwargs)
                             for split in ("train", "val", "test")}
            self.metadata = self.datasets["train"].metadata
            self.metadata["supervised_scales"] = {}
            for name in ("condition_energy", "control_derivative", "constraint_derivative"):
                values = torch.cat([condition.labels[name].flatten()
                                    for sample in self.datasets["train"]
                                    for condition in sample.variational.conditions
                                    if name in condition.labels])
                self.metadata["supervised_scales"][name] = max(float(values.square().mean().sqrt()), 1e-8)
            self.metadata["scale_kind"] = "training component RMS; minimum 1e-8; no reference subtraction"
            # All splits carry the same dataset contract and TRAIN statistics.
            # Per-condition curvature remains in each condition's provenance.
            for dataset in self.datasets.values():
                dataset.metadata = self.metadata
            self.dataset = self.datasets["train"]
            self.train_idx = self.metadata["splits"]["train"]
            self.val_idx = self.metadata["splits"]["val"]
            self.test_idx = self.metadata["splits"]["test"]
