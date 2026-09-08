"""Small learned quadratic functional and independent amortized proposal.

This representation is wrapped in the existing NeuralNetworkPotential, trained
by LitNNP, and reconstructed from ordinary Curator model configuration.
"""

from dataclasses import asdict
import copy

import torch
from torch import nn

from curator.data import properties
from curator.data.variational_data import canonical_boundary_hash
from curator.data.variational_oracle import (
    ORACLE_GAUGE, ORACLE_UNITS, ORACLE_VARIABLE, oracle_boundary, oracle_features,
)
from curator.variational import (
    VariableLayout, compile_boundary, prepare_amortized_training_batch,
    solve_implicit, solve_linear_response,
)


class LearnedVariationalOracle(nn.Module):
    """Learn H(R), g(R), c(R), then freeze them to learn a solution proposal.

    The physical head predicts the ten lower-triangular entries of L, four
    entries of g, and c from distance and squared-distance features. H=LL^T+0.5I
    is positive definite by construction. No analytic teacher coefficients are
    used by the model. Training stages share the same state-dict layout.
    """

    model_outputs = ["variational_prediction", "amortized_training"]

    def __init__(self, stage="physical", hidden_features=32, seed=4, data_seed=17, heads=None):
        super().__init__()
        if stage not in {"physical", "proposal"}:
            raise ValueError("stage must be physical or proposal")
        self.stage = stage
        self.hidden_features = hidden_features
        self.seed = seed
        self.data_seed = data_seed
        self.heads = heads
        self._metadata_snapshot = self._current_metadata()
        self._dataset_metadata = None
        # Local RNG isolation makes initialization reproducible without changing
        # the trainer's shuffle/dropout random stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.physical = nn.Linear(2, 15, dtype=torch.float64)
            self.proposal = nn.Sequential(
                nn.Linear(6, hidden_features, dtype=torch.float64), nn.Tanh(),
                nn.Linear(hidden_features, hidden_features, dtype=torch.float64), nn.Tanh(),
                nn.Linear(hidden_features, 4, dtype=torch.float64),
            )
            with torch.no_grad():
                self.physical.weight.mul_(0.08)
                self.physical.bias.zero_()
                self.physical.bias[torch.tensor([0, 2, 5, 9])] = 1.0
        self.physical.requires_grad_(stage == "physical")
        self.proposal.requires_grad_(stage == "proposal")

    def _current_metadata(self):
        # The training stage intentionally belongs to configuration, so a
        # physical checkpoint can initialize a fresh proposal-stage model.
        return copy.deepcopy({
            "schema": "learned_variational_oracle_v1", "units": ORACLE_UNITS,
            "gauge": ORACLE_GAUGE, "variable_spec": asdict(ORACLE_VARIABLE),
            "n_entities": 2, "flattening": "entity_major", "data_seed": self.data_seed,
            "teacher_policy": "rebuild_each_batch_from_current_detached_functional",
        })

    def _validate_static_metadata(self):
        if getattr(self, "_metadata_snapshot", None) != self._current_metadata():
            raise ValueError("oracle checkpoint schema/units/gauge/layout/data seed are incompatible")

    def get_extra_state(self):
        return dict(copy.deepcopy(self._metadata_snapshot),
                    dataset_metadata=copy.deepcopy(self._dataset_metadata))

    def set_extra_state(self, state):
        self._validate_static_metadata()
        metadata = dict(state)
        if "dataset_metadata" not in metadata:
            raise ValueError("oracle checkpoint is missing dataset metadata")
        dataset_metadata = metadata.pop("dataset_metadata")
        if metadata != self._metadata_snapshot:
            raise ValueError("oracle checkpoint schema/units/gauge/layout/data seed are incompatible")
        if dataset_metadata is not None:
            self._validate_dataset_contract(dataset_metadata)
        if self._dataset_metadata is not None and dataset_metadata != self._dataset_metadata:
            raise ValueError("oracle dataset metadata/splits differ from the checkpoint")
        self._dataset_metadata = copy.deepcopy(dataset_metadata)

    def _validate_dataset_contract(self, metadata):
        if not isinstance(metadata, dict) or not metadata.get("splits"):
            raise ValueError("oracle requires persisted training dataset metadata/splits")
        if metadata.get("seed") != self.data_seed:
            raise ValueError("oracle model data_seed must match the dataset seed")
        for key in ("schema", "units", "gauge", "variable_spec", "n_entities", "flattening"):
            if metadata.get(key) != self._metadata_snapshot[key]:
                raise ValueError(f"oracle dataset {key} is incompatible with the model")

    def setup_from_datamodule(self, datamodule):
        self._validate_static_metadata()
        self._validate_dataset_contract(datamodule.metadata)
        if self._dataset_metadata is not None and datamodule.metadata != self._dataset_metadata:
            raise ValueError("oracle dataset metadata/splits differ from the checkpoint")
        self._dataset_metadata = copy.deepcopy(datamodule.metadata)

    def validate_dataset_metadata(self, metadata):
        """Validate full-object checkpoints and independent evaluation splits."""
        self._validate_static_metadata()
        if getattr(self, "_dataset_metadata", None) is None:
            raise ValueError("oracle requires persisted training dataset metadata/splits")
        self._validate_dataset_contract(metadata)
        if metadata != self._dataset_metadata:
            raise ValueError("oracle evaluation dataset metadata/splits differ from training")

    def coefficients(self, positions):
        features = oracle_features(positions)
        if self.stage == "proposal":
            # Keep position derivatives for physical force evaluation, while
            # enforcing phase isolation even if a generic fine-tuning policy
            # changes parameter.requires_grad after model construction.
            raw = nn.functional.linear(features, self.physical.weight.detach(), self.physical.bias.detach())
        else:
            raw = self.physical(features)
        rows, columns = torch.tril_indices(4, 4, device=positions.device)
        lower = raw.new_zeros(raw.shape[:-1] + (4, 4))
        lower[..., rows, columns] = raw[..., :10]
        hessian = lower @ lower.transpose(-1, -2) + 0.5 * torch.eye(4, device=raw.device, dtype=raw.dtype)
        return hessian, raw[..., 10:14], raw[..., 14]

    def problems(self, batch, positions=None):
        """Build current physical problems; useful for certification/evaluation."""
        if positions is None:
            positions = batch[properties.positions].reshape(-1, 2, 3)
        hessian, linear, constant = self.coefficients(positions)
        layout = VariableLayout(ORACLE_VARIABLE, n_entities=2, dtype=positions.dtype, device=positions.device)
        fields, problems = [], []
        target = batch["variational"]
        for index, spec in enumerate(target["boundary_spec"]):
            field = [value for control in spec["controls"] for value in control["value"]]
            boundary = oracle_boundary(field)
            if canonical_boundary_hash(asdict(boundary)) != target["boundary_hash"][index]:
                raise ValueError("oracle received an unsupported or inconsistent boundary")
            compiled = compile_boundary(boundary, layout)
            geometry = int(target["condition_geometry_index"][index])
            problems.append(compiled.to_problem(hessian[geometry], linear[geometry], constant[geometry]))
            fields.append(positions.new_tensor(field))
        return tuple(problems), torch.stack(fields)

    def forward(self, data):
        if torch.is_inference_mode_enabled():
            raise RuntimeError("ACVR force evaluation requires Trainer(inference_mode=False)")
        # Force labels need position derivatives even under validation no_grad.
        with torch.enable_grad():
            positions = data[properties.positions].detach().clone().reshape(-1, 2, 3).requires_grad_(True)
            target = data["variational"]
            geometry_index = target["condition_geometry_index"]
            problems, fields = self.problems(data, positions)
            states, energies, forces, responses = [], [], [], []
            for index, problem in enumerate(problems):
                solved = solve_implicit(problem)
                states.append(solved.state)
                energies.append(solved.energy.reshape(1))
                # Full differentiation retains the mixed position/parameter
                # derivatives required by force training.
                forces.append(-torch.autograd.grad(
                    solved.energy, positions, create_graph=self.training and self.stage == "physical",
                    retain_graph=True,
                )[0][geometry_index[index]])
                responses.append(solve_linear_response(
                    problem, torch.eye(4, dtype=positions.dtype, device=positions.device)
                ).state)

            count = len(problems)
            predicted_labels = {}
            for key, values, shape, size in (
                ("condition_energy", energies, (1,), 1),
                ("condition_forces", forces, (2, 3), 6),
                ("variable_value", states, (2, 2), 4),
            ):
                predicted_labels[key] = {
                    "values": torch.cat([value.reshape(-1) for value in values]),
                    "ptr": torch.arange(count + 1, device=positions.device) * size,
                    "shape": (shape,) * count, "unit": ORACLE_UNITS[key],
                }
            prediction = {
                "value_space": "physical", "labels": predicted_labels,
                "responses": {
                    "values": torch.stack(responses).reshape(-1),
                    "ptr": torch.arange(count + 1, device=positions.device) * 16,
                    "condition_index": torch.arange(count, device=positions.device),
                    "input_ref": ("field",) * count, "output_ref": ("variable_value",) * count,
                    "shape": ((4, 4),) * count, "unit": ("1/eV",) * count,
                    "gauge": (ORACLE_GAUGE,) * count, "boundary_hash": target["boundary_hash"],
                },
            }
            features = torch.cat((oracle_features(positions.detach())[geometry_index], fields), dim=-1)
            with torch.set_grad_enabled(self.stage == "proposal"):
                proposal = self.proposal(features).reshape(-1)
            # Never cache teachers across parameter updates. The shared helper
            # detaches problem coefficients and exact labels before proposal loss.
            amortized = prepare_amortized_training_batch(problems, proposal)
        return {"variational_prediction": prediction, "amortized_training": amortized}
