"""Reproducible geometry-grouped data for a learned quadratic ACVR oracle.

This is synthetic reference data, not an electronic-structure dataset.  The two
entities each carry two scalar internal channels, with an absolute conjugate
field.  Full field-response labels make the four-dimensional Hessian observable.
"""

from dataclasses import asdict

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from curator.variational import BoundarySpec, ControlSpec, GroupSpec, VariableSpec

from . import properties
from .atoms_data import AtomsData
from .collate_atoms_data import collate_atoms_data
from .variational_data import (
    VariationalCondition,
    VariationalData,
    VariationalResponse,
    canonical_boundary_hash,
    geometry_hash,
)


ORACLE_VARIABLE = VariableSpec(
    name="internal", kind="scalar_channels", components=2,
    component_names=("a", "b"), unit="dimensionless", conjugate_unit="eV",
)
ORACLE_GAUGE = {"kind": "absolute_conjugate_field", "energy_zero": "analytic_oracle_v1"}
ORACLE_UNITS = {
    "positions": "Angstrom", "condition_energy": "eV",
    "condition_forces": "eV/Angstrom", "variable_value": "dimensionless",
    "field": "eV", "response": "1/eV",
}


def oracle_boundary(field):
    """The serialized fields are the only condition inputs to either network."""
    return BoundarySpec(
        variable=ORACLE_VARIABLE,
        groups=(GroupSpec.from_indices("left", [0]), GroupSpec.from_indices("right", [1])),
        controls=(
            ControlSpec.group_potential("left_field", "left", tuple(field[:2]), unit="eV"),
            ControlSpec.group_potential("right_field", "right", tuple(field[2:]), unit="eV"),
        ),
    )


def oracle_features(positions):
    """Translation/rotation-invariant two-entity distance features."""
    distance = torch.linalg.vector_norm(positions[..., 1, :] - positions[..., 0, :], dim=-1)
    scaled = (distance - 1.6) / 0.8
    return torch.stack((scaled, scaled.square()), dim=-1)


def oracle_coefficients(positions):
    """Known analytic coefficients, used only for labels and independent checks."""
    features = oracle_features(positions)
    t, t2 = features.unbind(-1)
    lower = positions.new_tensor([
        [1.3, 0.0, 0.0, 0.0], [0.12, 1.1, 0.0, 0.0],
        [-0.08, 0.07, 1.2, 0.0], [0.05, -0.09, 0.11, 1.4],
    ])
    slope = positions.new_tensor([
        [0.10, 0.0, 0.0, 0.0], [0.03, -0.08, 0.0, 0.0],
        [0.02, -0.02, 0.07, 0.0], [-0.03, 0.02, 0.04, -0.09],
    ])
    lower = lower + t[..., None, None] * slope
    hessian = lower @ lower.transpose(-1, -2) + 0.5 * torch.eye(4, dtype=positions.dtype, device=positions.device)
    linear = (
        positions.new_tensor([0.25, -0.15, 0.20, -0.30])
        + t[..., None] * positions.new_tensor([0.12, 0.08, -0.10, 0.06])
        + t2[..., None] * positions.new_tensor([0.02, -0.03, 0.01, 0.04])
    )
    constant = 0.15 + 0.20 * t + 0.18 * t2
    return hessian, linear, constant


class VariationalOracleDataset(Dataset):
    """Generate an exact, deterministic train/validation/test geometry split.

    Training contains the two distance endpoints. Validation/test distances are
    strictly inside that range; their scalar field amplitudes also lie inside
    the training range. All conditions belonging to one geometry stay together.
    """

    def __init__(self, split="train", num_train=64, num_val=16, num_test=16,
                 seed=17, num_conditions=5):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if num_train < 2 or num_val < 1 or num_test < 1 or num_conditions < 2:
            raise ValueError("oracle needs >=2 train geometries, >=1 val/test geometry and >=2 conditions")
        self.split = split
        total = num_train + num_val + num_test
        generator = torch.Generator().manual_seed(seed)
        interior = (torch.randperm(total - 2, generator=generator) + 1).tolist()
        splits = {
            "train": [0, total - 1] + interior[:num_train - 2],
            "val": interior[num_train - 2:num_train + num_val - 2],
            "test": interior[num_train + num_val - 2:],
        }
        self.indices = splits[split]
        self.metadata = {
            "schema": "learned_variational_oracle_v1", "seed": seed,
            "num_conditions": num_conditions, "splits": splits,
            "variable_spec": asdict(ORACLE_VARIABLE), "n_entities": 2,
            "flattening": "entity_major", "units": ORACLE_UNITS, "gauge": ORACLE_GAUGE,
            "distance_range": [0.8, 2.4], "train_field_amplitude_range": [-0.8, 0.8],
            "held_out_field_amplitude_range": [-0.6, 0.6],
            "teacher": "analytic_v1; direct torch.linalg.solve; no learned parameters",
        }
        distances = torch.linspace(0.8, 2.4, total, dtype=torch.float64)
        extent = 0.8 if split == "train" else 0.6
        amplitudes = torch.linspace(-extent, extent, num_conditions, dtype=torch.float64)
        fields = amplitudes[:, None] * torch.tensor([0.8, -0.5, 0.3, -0.6], dtype=torch.float64)
        self.samples = []
        # Dataset setup can run under Lightning's evaluation no-grad context.
        with torch.inference_mode(False), torch.enable_grad():
            for index in self.indices:
                positions = torch.zeros((2, 3), dtype=torch.float64)
                positions[1, 0] = distances[index]
                positions.requires_grad_(True)
                hessian, linear, constant = oracle_coefficients(positions)
                response = torch.linalg.inv(hessian)
                conditions = []
                for condition_index, field in enumerate(fields):
                    # Independently compute labels without the ACVR compiler/solver.
                    state = torch.linalg.solve(hessian, field - linear)
                    energy = 0.5 * state @ hessian @ state + (linear - field) @ state + constant
                    forces = -torch.autograd.grad(energy, positions, retain_graph=True)[0]
                    spec = asdict(oracle_boundary(field.tolist()))
                    conditions.append(VariationalCondition(
                        condition_uid=f"g{index:04d}-c{condition_index:02d}",
                        electron_count=None, electron_count_convention=None, net_charge=None,
                        boundary_spec=spec, boundary_hash=canonical_boundary_hash(spec),
                        gauge=dict(ORACLE_GAUGE),
                        labels={"condition_energy": energy.detach().reshape(1),
                                "condition_forces": forces.detach(),
                                "variable_value": state.detach().reshape(2, 2)},
                        label_units={key: ORACLE_UNITS[key] for key in (
                            "condition_energy", "condition_forces", "variable_value")},
                        label_metadata={"condition_energy": {"energy_kind": "field_legendre_potential"}},
                        responses=(VariationalResponse(
                            input_ref="field", output_ref="variable_value", value=response.detach(),
                            unit="1/eV", method="analytic_inverse_hessian", gauge=dict(ORACLE_GAUGE),
                        ),),
                        provenance={"split": split, "field_amplitude": float(amplitudes[condition_index])},
                    ))
                atoms = {
                    properties.positions: positions.detach(),
                    properties.atomic_numbers: torch.tensor([1, 2], dtype=torch.long),
                    properties.n_atoms: torch.tensor([2], dtype=torch.long),
                    properties.cell: torch.zeros((3, 3), dtype=torch.float64),
                    properties.pbc: torch.zeros((1, 3), dtype=torch.bool),
                }
                self.samples.append(AtomsData(
                    atoms=atoms, targets={}, task="variational_oracle",
                    meta={"split": split, "geometry_index": index},
                    variational=VariationalData(
                        schema_version="1", geometry_uid=f"oracle-g{index:04d}",
                        geometry_hash=geometry_hash(atoms[properties.atomic_numbers], positions,
                                                    atoms[properties.cell], atoms[properties.pbc]),
                        variable_spec=asdict(ORACLE_VARIABLE),
                        group_member_ptr=torch.tensor([0, 1, 2], dtype=torch.long),
                        group_atom_index=torch.tensor([0, 1], dtype=torch.long),
                        group_names=("left", "right"), group_roles=("entity", "entity"),
                        conditions=tuple(conditions), provenance={"seed": seed, "split": split},
                    ),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def get_n_atoms(self, index):
        return 2


class VariationalOracleDataModule(pl.LightningDataModule):
    """Use the normal Lightning lifecycle and Curator atom-data collation."""

    def __init__(self, batch_size=16, num_train=64, num_val=16, num_test=16,
                 seed=17, num_conditions=5, num_workers=0, shuffle=True,
                 default_dtype="float64", datapath=None, train_path=None, heads=None):
        super().__init__()
        if default_dtype not in {"float64", torch.float64}:
            raise ValueError("the reference oracle requires float64")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.default_dtype = torch.float64
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.datapath, self.train_path = datapath, train_path
        self.heads = heads or ["energy"]
        self.species = ["H", "He"]
        self.dataset_kwargs = dict(num_train=num_train, num_val=num_val, num_test=num_test,
                                   seed=seed, num_conditions=num_conditions)
        self.generator = torch.Generator().manual_seed(seed)
        self.datasets = {}

    def setup(self, stage=None):
        if not self.datasets:
            self.datasets = {split: VariationalOracleDataset(split=split, **self.dataset_kwargs)
                             for split in ("train", "val", "test")}
            self.metadata = self.datasets["train"].metadata
            self.dataset = self.datasets["train"]
            self.train_idx = self.metadata["splits"]["train"]
            self.val_idx = self.metadata["splits"]["val"]
            self.test_idx = self.metadata["splits"]["test"]

    def _loader(self, split):
        self.setup()
        # A resumed Trainer may run an extra validation sanity pass. Its
        # dataloader must not consume the training shuffle generator.
        generator = self.generator if split == "train" else torch.Generator().manual_seed(
            self.dataset_kwargs["seed"] + (1 if split == "val" else 2)
        )
        return DataLoader(self.datasets[split], batch_size=self.batch_size,
                          shuffle=self.shuffle and split == "train", generator=generator,
                          num_workers=self.num_workers, collate_fn=collate_atoms_data)

    def train_dataloader(self):
        return self._loader("train")

    def val_dataloader(self):
        return self._loader("val")

    def test_dataloader(self):
        return self._loader("test")

    def state_dict(self):
        self.setup()
        return {"metadata": self.metadata, "generator_state": self.generator.get_state()}

    def load_state_dict(self, state_dict):
        self.setup()
        if state_dict["metadata"] != self.metadata:
            raise ValueError("oracle dataset metadata/splits differ from the checkpoint")
        self.generator.set_state(state_dict["generator_state"])
