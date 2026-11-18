"""Prepared field-coverage data for the existing periodic charge teacher.

Geometry, teacher, group controls and the six legacy conditions are preserved.
Only external-field conditions and their independently solved labels are added.
"""

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from curator.layer._ewald import EwaldSummation
from curator.variational.backends.ewald import EwaldChargeOperator

from . import properties
from .periodic_charge_oracle import periodic_charge_coefficients
from .variational_oracle import VariationalOracleDataModule


def field_directions(side, seed, geometry_uid):
    """Full zero-sum bases on both groups, sparse pairs and mixed fields.

    Every vector has maximum absolute component one. The amplitude therefore
    means peak per-site potential (V), not an RMS or a vector norm. Independent
    deterministic streams give new directions on each geometry and probe set.
    """
    digest = hashlib.sha256(f"{seed}:{geometry_uid}".encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % 2**63)
    result = []
    for group, start in (("left", 0), ("right", side)):
        raw = torch.randn((side, side - 1), dtype=torch.float64, generator=generator)
        basis = torch.linalg.qr(raw - raw.mean(0), mode="reduced").Q
        basis = basis / basis.abs().amax(0)
        for index, vector in enumerate(basis.T):
            field = torch.zeros(2 * side, dtype=torch.float64)
            field[start:start + side] = vector
            result.append((group, "dense_zero_sum", index, field))
        pairs = torch.combinations(torch.arange(side), r=2)
        selected = torch.randperm(len(pairs), generator=generator)[:min(2, len(pairs))]
        for index, pair in enumerate(pairs[selected]):
            field = torch.zeros(2 * side, dtype=torch.float64)
            field[start + pair[0]], field[start + pair[1]] = 1, -1
            result.append((group, "sparse_pair", index, field))
    for index in range(2):
        field = torch.randn((2, side), dtype=torch.float64, generator=generator)
        field = field - field.mean(1, keepdim=True)
        field = field.flatten() / field.abs().max()
        result.append(("both", "mixed_zero_sum", index, field))
    return result


def teacher_matrices(sample, metadata):
    kernel = EwaldChargeOperator(EwaldSummation(
        alpha=metadata["ewald_alpha"], acc_factor=metadata["ewald_accuracy"]
    )).geometry_kernels(sample.atoms)[0]
    chi, hardness, constant = periodic_charge_coefficients(sample.atoms)
    return kernel + torch.diag(hardness), chi, constant


def expand_field_conditions(sample, metadata, amplitudes=(0.1, 0.3, 1.0), seed=20260914):
    """Solve all new conditions in batched, independent linear KKT systems."""
    amplitudes = tuple(float(value) for value in amplitudes)
    if (not amplitudes or len(set(amplitudes)) != len(amplitudes)
            or any(not 0 < value < float("inf") for value in amplitudes)):
        raise ValueError("amplitudes must be distinct finite positive voltages")
    n = sample.atoms[properties.positions].shape[0]
    side = n // 2
    directions = field_directions(side, seed, sample.variational.geometry_uid)
    hessian, chi, constant = teacher_matrices(sample, metadata)
    group_matrix = hessian.new_zeros((2, n))
    group_matrix[0, :side], group_matrix[1, side:] = 1, 1
    generated = list(sample.variational.conditions)
    for kind in ("conq", "conp"):
        bases = [condition for condition in sample.variational.conditions
                 if condition.provenance["boundary_kind"] == kind
                 and not condition.provenance["local_drive"]]
        constraint = group_matrix if kind == "conq" else hessian.new_ones((1, n))
        rank = constraint.shape[0]
        kkt = hessian.new_zeros((n + rank, n + rank))
        kkt[:n, :n], kkt[:n, n:], kkt[n:, :n] = hessian, constraint.T, constraint
        cases = []
        for index, (group, family, direction_index, direction) in enumerate(directions):
            base = bases[index % len(bases)]
            for amplitude in amplitudes:
                pair_uid = f"{base.condition_uid}:coverage:{seed}:{group}:{family}:{direction_index}:{amplitude:g}"
                for sign in (1, -1):
                    local = sign * amplitude * direction
                    drive = local.clone()
                    if kind == "conp":
                        drive += group_matrix.T @ base.inputs["control_value"]
                    cases.append((base, group, family, direction_index, amplitude, sign,
                                  pair_uid, local, drive))
        drives = torch.stack([case[-1] for case in cases], dim=1)
        targets = torch.stack([case[0].inputs["constraint_target"] for case in cases], dim=1)
        solutions = torch.linalg.solve(kkt, torch.cat((drives - chi[:, None], targets)))
        charges, multipliers = solutions[:n], solutions[n:]
        energies = constant + ((chi[:, None] - drives) * charges).sum(0)
        energies += 0.5 * (charges * (hessian @ charges)).sum(0)
        for index, case in enumerate(cases):
            base, group, family, direction_index, amplitude, sign, pair_uid, local, _ = case
            labels = {"condition_energy": energies[index].reshape(1).clone(),
                      "variable_value": charges[:, index, None].clone(),
                      "constraint_derivative": -multipliers[:, index].clone()}
            if kind == "conp":
                labels["control_derivative"] = -group_matrix @ charges[:, index]
            provenance = {**base.provenance,
                          "condition_design": "field_coverage_v2", "condition_seed": seed,
                          "perturbed_group": group, "direction_family": family,
                          "direction_index": direction_index, "field_peak_V": amplitude,
                          "field_rms_all_sites_V": float(local.square().mean().sqrt()),
                          "sign": sign, "pair_uid": pair_uid,
                          "base_condition_uid": base.condition_uid}
            # The historical evaluator interprets local_drive=True specifically
            # as its single right-pair intervention. Do not misclassify new
            # left/mixed/multiscale conditions as that historical metric.
            provenance.pop("local_drive", None)
            generated.append(replace(
                base, condition_uid=f"{pair_uid}:{sign:+d}", labels=labels,
                inputs={**base.inputs, "local_external_field": local[:, None].clone()},
                provenance=provenance,
            ))
    return replace(sample, variational=replace(sample.variational, conditions=tuple(generated)))


class PreparedChargeDataset(Dataset):
    """Frozen geometry groups; optional views retain all four unperturbed controls."""

    def __init__(self, samples, metadata, split, view="all", amplitude=None):
        if view not in {"all", "legacy", "coverage", "amplitude"}:
            raise ValueError("view must be all, legacy, coverage, or amplitude")
        if view == "amplitude" and amplitude not in metadata["field_design"]["amplitudes_V"]:
            raise ValueError("amplitude must be one of the prepared field amplitudes")
        self.metadata, self.split = metadata, split
        self.indices = metadata["splits"][split]
        self.n_atoms = metadata["n_atoms"]
        self.samples = []
        for sample in samples:
            conditions = []
            for condition in sample.variational.conditions:
                new = condition.provenance.get("condition_design") == "field_coverage_v2"
                base = condition.provenance.get("local_drive") is False
                if (view == "all" or view == "legacy" and not new
                        or view == "coverage" and (new or base)
                        or view == "amplitude" and (base or new and
                            condition.provenance["field_peak_V"] == amplitude)):
                    conditions.append(condition)
            self.samples.append(replace(sample, variational=replace(
                sample.variational, conditions=tuple(conditions))))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def get_n_atoms(self, index):
        return self.n_atoms


class PreparedChargeDataModule(VariationalOracleDataModule):
    """Load the prepared data without regenerating labels or changing models."""

    def __init__(self, archive_path, view="all", amplitude=None, verify_hashes=True, **kwargs):
        super().__init__(**kwargs)
        self.archive_path = Path(archive_path)
        self.view, self.amplitude = view, amplitude
        self.verify_hashes = verify_hashes

    def setup(self, stage=None):
        if self.datasets:
            return
        manifest = json.loads((self.archive_path / "manifest.json").read_text())
        self.metadata = copy.deepcopy(manifest["dataset_metadata"])
        self.metadata["prepared_view"] = {"view": self.view, "amplitude": self.amplitude}
        self.metadata["archive_sha256"] = {split: item["sha256"]
                                            for split, item in manifest["archives"].items()}
        # Preserve the manifest's shuffle seed rather than an unrelated default.
        self.dataset_kwargs["seed"] = self.metadata["seed"]
        self.generator.manual_seed(self.metadata["seed"])
        for split, item in manifest["archives"].items():
            path = self.archive_path / item["path"]
            if self.verify_hashes and hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"prepared {split} archive hash differs from manifest")
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved["split"] != split or saved["geometry_indices"] != self.metadata["splits"][split]:
                raise ValueError("prepared geometry split differs from manifest")
            self.datasets[split] = PreparedChargeDataset(
                saved["samples"], self.metadata, split, self.view, self.amplitude)
        scales = {}
        for name in ("condition_energy", "control_derivative", "constraint_derivative"):
            values = torch.cat([condition.labels[name].flatten()
                                for sample in self.datasets["train"]
                                for condition in sample.variational.conditions if name in condition.labels])
            scales[name] = max(float(values.square().mean().sqrt()), 1e-8)
        self.metadata["supervised_scales"] = scales
        self.metadata["scale_kind"] = "selected TRAIN view component RMS; minimum 1e-8"
        self.dataset = self.datasets["train"]
        self.train_idx, self.val_idx, self.test_idx = (
            self.metadata["splits"][split] for split in ("train", "val", "test"))
