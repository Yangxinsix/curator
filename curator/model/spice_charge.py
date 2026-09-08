"""Isolated diatomic SPICE model with a learned fixed-charge functional.

The short-range reference uses the same local energy architecture. Its energy
has no distance dependence beyond the local cutoff. The ACVR branch adds the
complete isolated Gaussian Coulomb interaction, including its self energy.
"""

import copy
from dataclasses import asdict
import math

from ase.units import Bohr, Hartree
import torch
from torch import nn

from curator.data import properties
from curator.variational import (
    VariableLayout, compile_boundary, prepare_amortized_training_batch, solve_implicit,
)
from curator.variational.variables.charge import charge_variable_spec, fixed_total_charge_boundary


class SpiceChargeModel(nn.Module):
    """Element-shared local physical parameters and an independent proposal.

    Each input structure must have exactly two atoms and explicit total charge.
    Gaussian density widths are Cartesian standard deviations in Angstrom.
    Dipoles use the input coordinate origin, so translation by t changes them
    by Q*t for a charged molecule. The short-range variant uses uniform Q/2
    charges only as an unlearned dipole reference; it has no charge energy.
    """

    def __init__(self, stage="physical", variant="acvr", hidden_features=32,
                 cutoff=6.0, seed=4, heads=None, gaussian_sigma=0.8):
        super().__init__()
        if stage not in {"physical", "proposal"} or variant not in {"acvr", "short_range"}:
            raise ValueError("unsupported SPICE training stage or model variant")
        if variant == "short_range" and stage == "proposal":
            raise ValueError("the short-range baseline has no variational proposal stage")
        if hidden_features < 1 or cutoff <= 0 or gaussian_sigma <= 0:
            raise ValueError("hidden_features, cutoff and gaussian_sigma must be positive")
        if not math.isfinite(cutoff) or not math.isfinite(gaussian_sigma):
            raise ValueError("cutoff and gaussian_sigma must be finite")
        self.stage, self.variant = stage, variant
        self.hidden_features, self.cutoff, self.seed = hidden_features, float(cutoff), seed
        self.gaussian_sigma, self.heads = float(gaussian_sigma), heads
        self.model_outputs = [properties.energy, properties.forces, properties.dipole, properties.atomic_charge]
        if stage == "proposal":
            self.model_outputs.append("amortized_training")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.physical = nn.ModuleDict({
                "embedding": nn.Embedding(119, hidden_features, dtype=torch.float64),
                "local": nn.Sequential(
                    nn.Linear(2 * hidden_features + 8, hidden_features, dtype=torch.float64), nn.SiLU(),
                    nn.Linear(hidden_features, hidden_features, dtype=torch.float64), nn.SiLU(),
                    nn.Linear(hidden_features, 3, dtype=torch.float64),
                ),
            })
            self.proposal = nn.ModuleDict({
                "embedding": nn.Embedding(119, hidden_features, dtype=torch.float64),
                "local": nn.Sequential(
                    nn.Linear(2 * hidden_features + 3, hidden_features, dtype=torch.float64), nn.SiLU(),
                    nn.Linear(hidden_features, hidden_features, dtype=torch.float64), nn.SiLU(),
                    nn.Linear(hidden_features, 1, dtype=torch.float64),
                ),
            })
        self.physical.requires_grad_(stage == "physical")
        self.proposal.requires_grad_(stage == "proposal")
        self._metadata_snapshot = self._current_metadata()
        self._dataset_metadata = None

    def _current_metadata(self):
        return {
            "schema": "spice_charge_diatomic_v1", "variant": self.variant,
            "hidden_features": self.hidden_features, "cutoff": self.cutoff,
            "gaussian_sigma": self.gaussian_sigma, "coulomb_prefactor": Hartree * Bohr,
            "kernel": "isolated_gaussian_erf_r_over_2sigma_with_self",
            "hardness_floor": 0.1, "variable_spec": asdict(charge_variable_spec()),
            "units": {"positions": "Angstrom", "energy": "eV", "forces": "eV/Angstrom",
                      "dipole": "e*Angstrom", "total_charge": "e"},
            "gauge": {"boundary": "fixed_total_charge", "dipole_origin": "source_cartesian_origin",
                      "coordinate_transform": "none"},
            "teacher_policy": "rebuild_each_batch_from_current_detached_functional",
        }

    def _validate_static_metadata(self):
        if self._metadata_snapshot != self._current_metadata():
            raise ValueError("SPICE model static units/kernel/layout semantics changed")

    def _validate_dataset_contract(self, metadata):
        if not isinstance(metadata, dict) or metadata.get("schema") != "spice_acvr_ion_pairs_v1":
            raise ValueError("SPICE model requires the ion-pair dataset manifest")
        if not isinstance(metadata.get("source_sha256"), str) or len(metadata["source_sha256"]) != 64:
            raise ValueError("SPICE dataset requires a source SHA256 identity")
        for category in ("units", "gauge"):
            for key, value in self._metadata_snapshot[category].items():
                if metadata.get(category, {}).get(key) != value:
                    raise ValueError(f"SPICE dataset {category}.{key} differs from model semantics")
        splits = metadata.get("splits", {})
        if set(splits) != {"train", "val", "test"}:
            raise ValueError("SPICE dataset requires train/val/test geometry splits")
        indices = [index for split in splits.values() for index in split]
        if len(indices) != len(set(indices)):
            raise ValueError("SPICE dataset geometry splits overlap or contain duplicates")

    def get_extra_state(self):
        return dict(copy.deepcopy(self._metadata_snapshot), dataset_metadata=copy.deepcopy(self._dataset_metadata))

    def set_extra_state(self, state):
        self._validate_static_metadata()
        metadata = dict(state)
        if "dataset_metadata" not in metadata:
            raise ValueError("SPICE checkpoint is missing dataset metadata")
        dataset = metadata.pop("dataset_metadata")
        if metadata != self._metadata_snapshot:
            raise ValueError("SPICE checkpoint units/kernel/layout/variant are incompatible")
        if dataset is not None:
            self._validate_dataset_contract(dataset)
        if self._dataset_metadata is not None and dataset != self._dataset_metadata:
            raise ValueError("SPICE checkpoint dataset metadata/splits differ")
        self._dataset_metadata = copy.deepcopy(dataset)

    def setup_from_datamodule(self, datamodule):
        self._validate_static_metadata()
        self._validate_dataset_contract(datamodule.metadata)
        if self._dataset_metadata is not None and datamodule.metadata != self._dataset_metadata:
            raise ValueError("SPICE dataset metadata/splits differ from the checkpoint")
        self._dataset_metadata = copy.deepcopy(datamodule.metadata)

    def validate_dataset_metadata(self, metadata):
        self._validate_static_metadata()
        self._validate_dataset_contract(metadata)
        if self._dataset_metadata is None or metadata != self._dataset_metadata:
            raise ValueError("SPICE evaluation dataset metadata/splits differ from training")

    @staticmethod
    def _inputs(batch, positions=None):
        counts = batch[properties.n_atoms].reshape(-1)
        if counts.numel() == 0 or not bool((counts == 2).all()):
            raise ValueError("SpiceChargeModel requires exactly two atoms per structure")
        positions = batch[properties.positions] if positions is None else positions
        if positions.shape != (2 * counts.numel(), 3):
            raise ValueError("positions must have shape [2 * n_structures, 3]")
        numbers = batch[properties.atomic_numbers]
        if numbers.dtype != torch.long or numbers.shape != (2 * counts.numel(),):
            raise ValueError("atomic_numbers must be a torch.long vector with two atoms per structure")
        if bool(((numbers < 1) | (numbers > 118)).any()):
            raise ValueError("atomic numbers must lie between 1 and 118")
        charge = batch[properties.total_charge].reshape(-1).to(positions)
        if charge.shape != counts.shape or not bool(torch.isfinite(charge).all()):
            raise ValueError("total_charge must contain one finite input control per structure")
        if not bool(torch.isfinite(positions).all()):
            raise ValueError("positions must be finite")
        if properties.pbc in batch and bool(batch[properties.pbc].any()):
            raise ValueError("the SPICE Gaussian Coulomb kernel requires isolated structures")
        positions = positions.reshape(-1, 2, 3)
        distance = torch.linalg.vector_norm(positions[:, 1] - positions[:, 0], dim=-1)
        if bool((distance <= 1e-8).any()):
            raise ValueError("diatomic structures must have noncoincident atoms")
        return positions, numbers.reshape(-1, 2), charge, distance

    def _physical_call(self, module, values):
        if self.stage == "proposal":
            # Preserve coordinate derivatives for force evaluation while also
            # isolating physics from generic policies that unfreeze all weights.
            return torch.func.functional_call(
                module, {name: value.detach() for name, value in module.named_parameters()}, (values,),
            )
        return module(values)

    def coefficients(self, batch, positions=None):
        """Return per-structure chi[B,2], hardness[B,2], short energy[B]."""
        positions, numbers, _, distance = self._inputs(batch, positions)
        embedded = self._physical_call(self.physical["embedding"], numbers)
        scaled = distance / self.cutoff
        capped = scaled.clamp(max=1.0)
        envelope = (1 - capped).pow(3) * (1 + 3 * capped + 6 * capped.square())
        centers = torch.linspace(0, 1, 8, dtype=positions.dtype, device=positions.device)
        radial = torch.exp(-16 * (scaled[:, None] - centers).square()) * envelope[:, None]
        local = torch.cat((embedded, embedded.flip(1) * envelope[:, None, None],
                           radial[:, None, :].expand(-1, 2, -1)), dim=-1)
        parameters = self._physical_call(self.physical["local"], local)
        return parameters[..., 0], nn.functional.softplus(parameters[..., 1]) + 0.1, parameters[..., 2].sum(1)

    def gaussian_kernel(self, distance):
        """Full Coulomb matrix; self energy is 0.5 * q_i**2 * K_ii."""
        pair = (Hartree * Bohr) * torch.erf(distance / (2 * self.gaussian_sigma)) / distance
        self_term = torch.full_like(pair, (Hartree * Bohr) / (math.sqrt(math.pi) * self.gaussian_sigma))
        return torch.stack((self_term, pair, pair, self_term), dim=-1).reshape(-1, 2, 2)

    def problems(self, batch, positions=None):
        """Build exact current physical problems, including short energy in c."""
        if self.variant != "acvr":
            raise ValueError("the short-range baseline has no variational charge functional")
        coordinates, _, charge, distance = self._inputs(batch, positions)
        chi, hardness, short_energy = self.coefficients(batch, positions)
        hessian = torch.diag_embed(hardness) + self.gaussian_kernel(distance)
        variable = charge_variable_spec()
        layout = VariableLayout(variable, n_entities=2, dtype=coordinates.dtype, device=coordinates.device)
        problems = []
        for index, target in enumerate(charge):
            boundary = compile_boundary(fixed_total_charge_boundary(float(target.detach()), variable), layout)
            problems.append(boundary.to_problem(hessian[index], chi[index], short_energy[index]))
        return tuple(problems)

    def propose(self, batch):
        """Return raw charges[2B] using an independent geometry-and-Q network."""
        if self.variant != "acvr":
            raise ValueError("the short-range baseline has no variational charge proposal")
        _, numbers, charge, distance = self._inputs(batch)
        with torch.set_grad_enabled(self.stage == "proposal" and torch.is_grad_enabled()):
            embedded = self.proposal["embedding"](numbers)
            radial = torch.stack((distance / self.cutoff, 1 / (1 + distance), charge), dim=-1)
            features = torch.cat((embedded, embedded.flip(1), radial[:, None, :].expand(-1, 2, -1)), dim=-1)
            return self.proposal["local"](features).reshape(-1)

    def forward(self, data):
        if torch.is_inference_mode_enabled():
            raise RuntimeError("SPICE force evaluation requires Trainer(inference_mode=False)")
        with torch.enable_grad():
            positions = data[properties.positions].detach().clone().requires_grad_(True)
            _, _, total_charge, _ = self._inputs(data, positions)
            if self.variant == "acvr":
                problems = self.problems(data, positions)
                solved = [solve_implicit(problem) for problem in problems]
                if not all(result.reliable for result in solved):
                    raise RuntimeError("SPICE exact charge solve failed certification")
                charges = torch.stack([result.state for result in solved])
                energy = torch.stack([result.energy for result in solved])
            else:
                _, _, energy = self.coefficients(data, positions)
                charges = (total_charge[:, None] / 2).expand(-1, 2)
            forces = -torch.autograd.grad(energy.sum(), positions, retain_graph=True,
                                          create_graph=self.training and self.stage == "physical")[0]
            output = {properties.energy: energy, properties.forces: forces,
                      properties.dipole: (charges[..., None] * positions.reshape(-1, 2, 3)).sum(1),
                      properties.atomic_charge: charges.reshape(-1)}
            if self.stage == "proposal":
                output["amortized_training"] = prepare_amortized_training_batch(problems, self.propose(data))
            return output
