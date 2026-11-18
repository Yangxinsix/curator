"""Joint variational energy and proposal heads on ordinary GNN features.

The physical energy keeps the complete coordinate/proposal graph. By default a
second, numerically identical proposal evaluation isolates residual gradients
to the proposal. The explicit ``joint`` policy reuses the live state and lets
stationarity also train the energy and representation. ``residual_evaluation=
"reuse"`` instead reuses the live energy/state and asks VariationalGradientTask
for proposal-only parameter gradients. Neither needs an inner solve or a dense
Hessian.
"""

from __future__ import annotations

import math
import copy
from dataclasses import asdict, fields, replace
from typing import Optional

import torch
from torch import nn

from curator.data import properties
from curator.data.variational_data import canonical_boundary_hash, condition_input
from curator.variational.compiler import compile_boundary
from curator.variational.functionals import FunctionalProblem, solve_functional_reference
from curator.variational.deployment import DeploymentFunctionalProblem
from curator.variational.observables import BatchedEnergyObservables, EnergyObservables, ObservableSpec
from curator.variational.spec import (
    BoundarySpec, ConstraintSpec, ControlSpec, EnvironmentSpec, GaugeSpec,
    GroupSpec, LinearTermSpec, VariableLayout, VariableSpec,
)

from ._force_output import ForceOutput


def _boundary_from_dict(payload):
    if isinstance(payload, BoundarySpec):
        return payload
    values = dict(payload)
    values["variable"] = VariableSpec(**values["variable"])
    values["groups"] = tuple(GroupSpec(**value) for value in values.get("groups", ()))
    for key, cls in (("equalities", ConstraintSpec), ("controls", ControlSpec),
                     ("environment", EnvironmentSpec)):
        entries = []
        for value in values.get(key, ()):
            value = dict(value)
            value["terms"] = tuple(LinearTermSpec(**term) for term in value["terms"])
            entries.append(cls(**value))
        values[key] = tuple(entries)
    if values.get("gauge") is not None:
        values["gauge"] = GaugeSpec(**values["gauge"])
    return BoundarySpec(**values)


def _condition_features(context, state_kind):
    scalar, field, origin = context["scalar"], context["network_field"], context["origin"]
    if state_kind == "scalar":
        return torch.cat((scalar, field, origin), dim=-1)
    invariants = torch.stack((field.square().sum(-1), origin.square().sum(-1),
                              (field * origin).sum(-1)), dim=-1)
    return torch.cat((scalar, invariants), dim=-1)


class VariationalStateHead(nn.Module):
    """Deterministic scalar or polar-vector proposal from shared GNN features."""

    def __init__(self, scalar_channels, vector_channels, components, state_kind, hidden_features):
        super().__init__()
        self.state_kind = state_kind
        self.components = components
        self.vector_channels = vector_channels
        extra = 2 * components if state_kind == "scalar" else 3
        output = components if state_kind == "scalar" else vector_channels + 2
        self.network = nn.Sequential(
            nn.Linear(scalar_channels + extra, hidden_features), nn.SiLU(),
            nn.Linear(hidden_features, hidden_features), nn.SiLU(),
            nn.Linear(hidden_features, output),
        )

    def forward(self, context):
        values = self.network(_condition_features(context, self.state_kind))
        if self.state_kind == "scalar":
            return values
        geometric = (context["vector"] * values[:, None, :self.vector_channels]).sum(-1)
        return (geometric + values[:, -2:-1] * context["network_field"]
                + values[:, -1:] * context["origin"])


class StableInternalEnergy(nn.Module):
    """A replaceable smooth reference energy, convex in the internal state.

    Positive quadratic/quartic local terms and positive graph-difference terms
    supply a stable starting implementation. This structure is not a requirement
    of the variational framework. A custom module can implement any smooth scalar
    ``forward(context, state[n_atoms, components])`` energy. Context contains
    scalar/vector features, positions, graph edges, field and feasible origin.
    ``network_field`` removes the constraint-normal gauge directions and is used
    by the default heads. ``field`` retains the full conjugate field for energy
    and response derivatives; custom energies using it must enforce their own
    gauge covariance.
    """

    def __init__(self, scalar_channels, vector_channels, components, state_kind,
                 hidden_features, curvature_floor=0.2):
        super().__init__()
        self.state_kind = state_kind
        self.vector_channels = vector_channels
        extra = 2 * components if state_kind == "scalar" else 3
        linear_channels = components if state_kind == "scalar" else vector_channels
        self.local = nn.Sequential(
            nn.Linear(scalar_channels + extra, hidden_features), nn.SiLU(),
            nn.Linear(hidden_features, 2 + linear_channels),
        )
        self.edge_strength = nn.Parameter(torch.tensor(-2.0))
        self.curvature_floor = float(curvature_floor)

    def forward(self, context, state):
        coefficients = self.local(_condition_features(context, self.state_kind))
        quadratic = nn.functional.softplus(coefficients[:, 0]) + self.curvature_floor
        quartic = 0.1 * nn.functional.softplus(coefficients[:, 1])
        if self.state_kind == "scalar":
            linear = coefficients[:, 2:]
        else:
            linear = (context["vector"] * coefficients[:, None, 2:]).sum(-1)
        norm_squared = state.square().sum(-1)
        energy = (0.5 * quadratic * norm_squared + 0.25 * quartic * norm_squared.square()
                  + (linear * state).sum(-1)).sum()
        edges = context["edge_index"]
        if edges.numel():
            differences = state[edges[:, 0]] - state[edges[:, 1]]
            scaled = (context["edge_distance"] / context["cutoff"]).clamp(max=1.0)
            envelope = (1 - scaled).pow(3) * (1 + 3 * scaled + 6 * scaled.square())
            weight = torch.exp(-context["edge_distance"].square()) * envelope
            energy = energy + 0.25 * nn.functional.softplus(self.edge_strength) * (
                weight * differences.square().sum(-1)
            ).sum()
        return energy


class VariationalEnergyOutput(ForceOutput):
    """Assemble predicted-state energies, conservative forces and routed residuals.

    Only structured condition outputs are public: one geometry can have several
    boundary conditions, so a condition energy must not masquerade as a geometry
    energy. The representation's ordinary energy readout supplies the baseline.
    ``state_mode='converged'`` is an explicit, small-system evaluation oracle;
    training always uses ``predicted`` and does not call an internal solver.

    ``energy_convention`` selects internal energy U or the boundary's
    thermodynamic potential G for the energy label. Dataset metadata can select
    this convention at setup. Forces, responses and stationarity always use G;
    in particular fixed-potential data may contain U and forces from G=U-VQ.

    ``batch_observables=True`` explicitly asserts that different geometries are
    independent throughout the GNN and energy graph (shared parameters are fine;
    cross-geometry pooling or training BatchNorm is not). It batches derivative
    execution in training and evaluation, and is disabled by default. Repeated
    conditions on the same geometry and unsupported derivative layouts retain
    the ordinary path.
    """

    requires_energy_grad = True
    joint_variational_output = True
    supports_state_modes = ("predicted", "converged")

    def __init__(self, state_kind="scalar", components=1, hidden_features=32,
                 internal_energy: Optional[nn.Module] = None, proposal: Optional[nn.Module] = None,
                 baseline_key=properties.energy, energy_unit="eV", force_unit="eV/Angstrom",
                 state_unit="dimensionless", conjugate_unit="eV", residual_scale=1.0,
                 compute_forces=True, state_mode="predicted", compute_responses=False,
                 retain_response_graph=False, response_unit=None, observables=(),
                 batch_observables=False, proposal_mode="absolute",
                 energy_convention="thermodynamic", state_semantics="variational_internal",
                 residual_gradient_policy="proposal_only", residual_evaluation="replay"):
        super().__init__()
        if state_kind not in {"scalar", "vector"}:
            raise ValueError("state_kind must be scalar or vector")
        if components < 1 or (state_kind == "vector" and components != 3):
            raise ValueError("vector states require components=3; scalar states need >=1 channel")
        if hidden_features < 1 or not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("hidden_features and fixed residual_scale must be positive")
        if state_mode not in self.supports_state_modes:
            raise ValueError("unsupported state_mode")
        if energy_convention not in ("internal", "thermodynamic"):
            raise ValueError("energy_convention must be internal or thermodynamic")
        self.energy_convention = energy_convention
        if state_semantics not in ("variational_internal", "predicted_auxiliary"):
            raise ValueError("unsupported state_semantics")
        self.state_semantics = state_semantics
        if residual_gradient_policy not in ("proposal_only", "joint"):
            raise ValueError("residual_gradient_policy must be proposal_only or joint")
        self.residual_gradient_policy = residual_gradient_policy
        if residual_evaluation not in ("replay", "reuse"):
            raise ValueError("residual_evaluation must be replay or reuse")
        self.residual_evaluation = residual_evaluation
        self.state_kind, self.components = state_kind, int(components)
        self.hidden_features = int(hidden_features)
        if isinstance(proposal, str):
            if proposal != "mlp":
                raise ValueError("use a BoundaryConditionedProposal module for communication proposals")
            proposal = None
        self.internal_energy, self.proposal = internal_energy, proposal
        self.baseline_key = baseline_key
        self.energy_unit, self.force_unit = energy_unit, force_unit
        self.state_unit, self.conjugate_unit = state_unit, conjugate_unit
        self.residual_scale = float(residual_scale)
        self.compute_forces = bool(compute_forces)
        self.produces_forces = self.compute_forces
        self.state_mode = state_mode
        if proposal_mode not in {"absolute", "correction"}:
            raise ValueError("proposal_mode must be absolute or correction")
        self.proposal_mode = proposal_mode
        self.compute_responses = bool(compute_responses)
        self.retain_response_graph = bool(retain_response_graph)
        self.batch_observables = bool(batch_observables)
        denominator = "(" + conjugate_unit + ")" if "/" in conjugate_unit else conjugate_unit
        numerator = "1" if state_unit == "dimensionless" else state_unit
        self.response_unit = response_unit or numerator + "/" + denominator
        self.observable_specs = tuple(
            value if isinstance(value, ObservableSpec) else ObservableSpec(**value)
            for value in observables
        )
        names = [value.name for value in self.observable_specs]
        if len(set(names)) != len(names):
            raise ValueError("observable names must be unique")
        reserved = {"condition_energy", "variable_value"}
        if self.compute_forces:
            reserved.add("condition_forces")
        if reserved.intersection(names):
            raise ValueError("configured observable duplicates an enabled built-in output")
        if any(value.unit is None for value in self.observable_specs):
            raise ValueError("configured observables require explicit physical units")
        self._dataset_metadata = None
        self.model_outputs = ["variational_prediction"]
        self._is_bound = False
        self.feature_spec = None
        self.cutoff = None

    def __setstate__(self, state):
        super().__setstate__(state)
        # Whole-model torch.load bypasses __init__. Models saved before
        # configured observables existed retain their original empty contract.
        if "observable_specs" not in self.__dict__:
            self.observable_specs = ()
        if "batch_observables" not in self.__dict__:
            self.batch_observables = False
        if "proposal_mode" not in self.__dict__:
            self.proposal_mode = "absolute"
        if "energy_convention" not in self.__dict__:
            self.energy_convention = "thermodynamic"
        if "state_semantics" not in self.__dict__:
            self.state_semantics = "variational_internal"
        if "residual_gradient_policy" not in self.__dict__:
            self.residual_gradient_policy = "proposal_only"
        if "residual_evaluation" not in self.__dict__:
            self.residual_evaluation = "replay"

    def configure_deployment(self, *, compute_residuals=False, energy_only=False,
                             reuse_constraint_projection=False, batch_forces=False):
        """Select an evaluation-only output policy without changing model weights.

        Residuals are omitted, never reported as zero. Energy-only evaluation
        also omits force/response/configured observable calculations. Gradients
        remain required for force outputs and automatic derivative backends.
        Analytic energy-only deployment also supports torch.no_grad(), but not
        torch.inference_mode(). Call reset_deployment() before training.
        """
        if self.training:
            raise RuntimeError("call eval() before configuring deployment")
        if self.state_mode != "predicted":
            raise RuntimeError("deployment requires predicted state_mode")
        if compute_residuals and self.state_semantics == "predicted_auxiliary":
            raise ValueError("latent states do not have stationarity residuals")
        if batch_forces and not self.batch_observables:
            raise ValueError("batch_forces requires batch_observables=True to assert independent geometries")
        if compute_residuals and reuse_constraint_projection:
            raise ValueError("projection reuse requires diagnostics-off deployment")
        self._deployment_options = {
            "compute_residuals": bool(compute_residuals),
            "energy_only": bool(energy_only),
            "reuse_constraint_projection": bool(reuse_constraint_projection),
            "batch_forces": bool(batch_forces),
        }
        return self

    @property
    def allows_no_grad_evaluation(self):
        """Explicit capability consumed by NeuralNetworkPotential."""
        options = getattr(self, "_deployment_options", None)
        return (
            not self.training and self.state_mode == "predicted"
            and options is not None and options["energy_only"]
            and not options["compute_residuals"]
            and getattr(self.internal_energy, "derivative_backend", "autograd") == "analytic"
        )

    def reset_deployment(self):
        """Restore the ordinary training/evaluation output contract."""
        self._deployment_options = None
        return self

    def _checkpoint_contract(self):
        contract = {
            "schema": "joint_variational_energy_v1", "training_mode": "joint",
            "energy_state": "predicted", "force_derivative": "full_proposal_path",
            "residual_gradient_policy": self.residual_gradient_policy, "state_kind": self.state_kind,
            "components": self.components, "baseline_key": self.baseline_key,
            "energy_unit": self.energy_unit, "force_unit": self.force_unit,
            "state_unit": self.state_unit, "conjugate_unit": self.conjugate_unit,
            "residual_scale": self.residual_scale, "response_unit": self.response_unit,
            "cutoff": self.cutoff,
        }
        if getattr(self, "proposal_mode", "absolute") == "correction":
            contract["proposal_mode"] = "initial_state_plus_correction"
            contract["expansion_state"] = "geometry_only_initial_state"
        if self.observable_specs:
            contract["observables"] = tuple(asdict(value) for value in self.observable_specs)
        if self.energy_convention != "thermodynamic":
            contract["energy_convention"] = self.energy_convention
        if self.state_semantics == "predicted_auxiliary":
            contract.update(state_semantics=self.state_semantics, training_mode="supervised_latent",
                            residual_gradient_policy="not_computed")
        if hasattr(self.proposal, "checkpoint_contract"):
            contract["proposal"] = self.proposal.checkpoint_contract()
        return contract

    def get_extra_state(self):
        return {"contract": self._checkpoint_contract(),
                "dataset_metadata": copy.deepcopy(self._dataset_metadata)}

    def set_extra_state(self, state):
        if state.get("contract") != self._checkpoint_contract():
            raise ValueError("joint variational checkpoint physical/gradient contract differs")
        self._dataset_metadata = copy.deepcopy(state.get("dataset_metadata"))

    def setup_from_datamodule(self, datamodule):
        metadata = getattr(datamodule, "metadata", None)
        if self._dataset_metadata is not None and metadata != self._dataset_metadata:
            raise ValueError("joint variational dataset metadata differs from checkpoint")
        convention = (metadata or {}).get("energy_convention")
        if convention is not None:
            if convention not in ("internal", "thermodynamic"):
                raise ValueError("energy_convention must be internal or thermodynamic")
            self.energy_convention = convention
        semantics = (metadata or {}).get("state_semantics", "variational_internal")
        if semantics not in ("variational_internal", "predicted_auxiliary"):
            raise ValueError("unsupported dataset state_semantics")
        self.state_semantics = semantics
        self._dataset_metadata = copy.deepcopy(metadata)

    def validate_dataset_metadata(self, metadata):
        if self._dataset_metadata is not None and metadata != self._dataset_metadata:
            raise ValueError("joint variational evaluation dataset differs from checkpoint")

    @property
    def is_bound(self):
        return self._is_bound

    def bind(self, spec, cutoff=None):
        from curator.model.features import IrrepsFeatureSpec, ScalarVectorFeatureSpec

        if self._is_bound:
            if self.feature_spec != spec or self.cutoff != cutoff:
                raise ValueError("variational head feature spec differs from representation")
            return
        if cutoff is None or not math.isfinite(float(cutoff)) or float(cutoff) <= 0:
            raise ValueError("variational output requires the representation's positive cutoff")
        self.cutoff = float(cutoff)
        self.feature_spec = spec
        if isinstance(spec, ScalarVectorFeatureSpec):
            self.scalar_channels, self.vector_channels = spec.scalar_channels, spec.vector_channels
        elif isinstance(spec, IrrepsFeatureSpec):
            self.scalar_channels = sum(mul for mul, ir in spec.irreps if str(ir) == "0e")
            self.vector_channels = sum(mul for mul, ir in spec.irreps if str(ir) == "1o")
        else:
            raise TypeError("unsupported variational feature specification")
        if not self.scalar_channels:
            raise ValueError("variational heads require invariant scalar features")
        if self.state_kind == "vector" and not self.vector_channels and self.proposal is None:
            raise ValueError("vector proposals require polar-vector GNN features")
        args = (self.scalar_channels, self.vector_channels, self.components,
                self.state_kind, self.hidden_features)
        if self.proposal is None:
            self.proposal = VariationalStateHead(*args)
        if self.internal_energy is None:
            self.internal_energy = StableInternalEnergy(*args)
        for module in (self.internal_energy, self.proposal):
            if hasattr(module, "bind"):
                module.bind(feature_spec=spec, scalar_channels=self.scalar_channels,
                            vector_channels=self.vector_channels, components=self.components,
                            state_kind=self.state_kind, cutoff=self.cutoff)
        from ._charge_internal_energy import EwaldChargeEnergy
        if getattr(self.proposal, "communication", None) == "ewald":
            if not isinstance(self.internal_energy, EwaldChargeEnergy):
                raise ValueError("Ewald proposal requires the same EwaldChargeEnergy used by the physical energy")
        if isinstance(self.internal_energy, EwaldChargeEnergy):
            if (self.state_unit, self.conjugate_unit, self.energy_unit) != ("e", "V", "eV"):
                raise ValueError("Ewald charge energy requires charge e, potential V and energy eV")
        for module in (self.proposal, self.internal_energy):
            if any(isinstance(child, (nn.modules.dropout._DropoutNd, nn.modules.batchnorm._BatchNorm))
                   for child in module.modules()):
                raise ValueError("joint residual routing requires deterministic heads without dropout or batch norm")
        self._is_bound = True

    def _features(self, data):
        from curator.model.features import scalar_vector_features

        features = self.feature_spec.flatten(data)
        scalar, vector = scalar_vector_features(features, self.feature_spec.irreps)
        return scalar, vector, features

    def _conditions(self, data):
        n_atoms = data[properties.n_atoms].reshape(-1)
        variational = data.get("variational")
        if variational is not None:
            indices = variational["condition_geometry_index"]
            boundaries = tuple(_boundary_from_dict(value) for value in variational["boundary_spec"])
            if len(boundaries) != indices.numel():
                raise ValueError("boundary conditions and geometry indices differ")
            for payload, expected in zip(variational["boundary_spec"], variational["boundary_hash"]):
                if canonical_boundary_hash(payload) != expected:
                    raise ValueError("variational boundary hash mismatch")
            return indices, boundaries
        # Ordinary atomistic batches may use an unconstrained state, total scalar
        # charge, or a uniform field. More complex boundaries use VariationalData.
        boundaries = []
        for index in range(n_atoms.numel()):
            variable = VariableSpec("internal", self.state_kind, self.components,
                                    self.state_unit, self.conjugate_unit, semantics=self.state_semantics)
            constraints, controls = (), ()
            if properties.total_charge in data:
                if self.state_kind != "scalar" or self.components != 1:
                    raise ValueError("total_charge shortcut requires one scalar state per atom")
                constraints = (ConstraintSpec.group_sum("total", "all",
                               float(data[properties.total_charge].reshape(-1)[index]), unit=self.state_unit),)
            # external_field is added as a live tensor below, not serialized via
            # tolist: energy/response derivatives must reach the caller's field.
            boundaries.append(BoundarySpec(variable, groups=(GroupSpec.all(),),
                                            equalities=constraints, controls=controls))
        return torch.arange(n_atoms.numel(), device=n_atoms.device), tuple(boundaries)

    def _context(self, data, scalar, vector, geometry, boundary, atom_ptr, condition=0, *,
                 projection_cache=None):
        start, stop = int(atom_ptr[geometry]), int(atom_ptr[geometry + 1])
        n_atoms = stop - start
        if boundary.variable.semantics != self.state_semantics:
            raise ValueError("condition state semantics differ; model requires " + self.state_semantics)
        if boundary.variable.components != self.components:
            raise ValueError("condition variable components differ from model")
        if boundary.variable.unit != self.state_unit or boundary.variable.conjugate_unit != self.conjugate_unit:
            raise ValueError("condition variable units differ from model")
        fields = {key: value[start:stop] for key, value in data.items()
                  if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == scalar.shape[0]}
        layout = VariableLayout(boundary.variable, n_atoms, fields,
                                dtype=scalar.dtype, device=scalar.device)
        compiled = compile_boundary(boundary, layout)
        overrides = {}
        for name in ("control_value", "constraint_target"):
            value = condition_input(data.get("variational", {}), name, condition)
            value = getattr(compiled, name) if value is None else value.to(scalar)
            if value.shape != getattr(compiled, name).shape:
                raise ValueError(name + " input shape differs from compiled boundary")
            if any(name in spec.derivatives for spec in self.observable_specs) and not value.requires_grad:
                value = value.detach().requires_grad_(True)
            overrides[name] = value
        compiled = replace(compiled, **overrides)
        n_state = n_atoms * self.components
        zeros = scalar.new_zeros(n_state)
        problem_type = FunctionalProblem if projection_cache is None else DeploymentFunctionalProblem
        constraint_problem = problem_type(lambda x: x.sum() * 0, n_state,
                                          compiled.constraint_matrix, compiled.constraint_target)
        if projection_cache is not None:
            projection_cache["problem"] = constraint_problem
        origin = constraint_problem.project_state(zeros)
        field = compiled.control_matrix.T @ compiled.control_value
        if compiled.environment_matrix.shape[0]:
            field = field + compiled.environment_matrix.T @ (
                compiled.environment_stiffness @ compiled.environment_target + compiled.environment_field)
        reference_field = field.reshape(n_atoms, self.components)
        uniform_input = condition_input(data.get("variational", {}), "external_field", condition)
        if uniform_input is not None:
            uniform_field = uniform_input.to(scalar)
        elif "_observable_fields" in data:
            uniform_field = data["_observable_fields"][geometry].to(scalar)
        elif "external_field" in data:
            uniform_field = data["external_field"].reshape(-1, self.components)[geometry].to(scalar)
        else:
            uniform_field = scalar.new_zeros(self.components)
        if uniform_field.shape != (self.components,):
            raise ValueError("uniform conjugate field and state component counts differ")
        if any("external_field" in spec.derivatives for spec in self.observable_specs):
            if not uniform_field.requires_grad:
                uniform_field = uniform_field.detach().requires_grad_(True)
        field = reference_field + uniform_field
        local_input = condition_input(data.get("variational", {}), "local_external_field", condition)
        if local_input is None:
            local_input = (data["local_external_field"][start:stop] if "local_external_field" in data
                           else scalar.new_zeros((n_atoms, self.components)))
        local_field = local_input.to(scalar)
        if local_field.shape != (n_atoms, self.components):
            raise ValueError("local conjugate field must have one row per atom")
        if any("local_external_field" in spec.derivatives for spec in self.observable_specs) and not local_field.requires_grad:
            local_field = local_field.detach().requires_grad_(True)
        field = field + local_field
        # A local field perturbation must have its own autograd node even when
        # canonical control weights depend on geometry. It must not alias the
        # reference field in the compensating linear-energy term below.
        field = field.clone()
        if self.compute_responses and not field.requires_grad:
            field = field.detach().requires_grad_(True)
        # A field in range(A.T) changes only the constrained energy zero:
        # -delta_field.T @ x = -lambda.T @ b. It cannot change the state or
        # learned state-dependent coefficients. Keep the original live field
        # for the boundary energy and differentiate through this projection.
        matrix = compiled.constraint_matrix
        network_field = field.reshape(-1)
        if projection_cache is not None:
            network_field = constraint_problem.project_tangent(network_field)
        elif matrix.shape[0]:
            network_field = network_field - matrix.T @ torch.linalg.solve(
                matrix @ matrix.T, matrix @ network_field)
        edges = data[properties.edge_idx]
        selected = ((edges[:, 0] >= start) & (edges[:, 0] < stop)
                    & (edges[:, 1] >= start) & (edges[:, 1] < stop))
        context = {
            "scalar": scalar[start:stop], "vector": vector[start:stop],
            "positions": data[properties.positions][start:stop],
            "edge_index": edges[selected] - start,
            "edge_distance": data[properties.edge_dist][selected],
            "edge_diff": data[properties.edge_diff][selected],
            "cell": (data[properties.cell].reshape(-1, 3, 3)[geometry] if properties.cell in data
                     else scalar.new_zeros((3, 3))),
            "pbc": (data[properties.pbc].reshape(-1, 3)[geometry] if properties.pbc in data
                    else torch.zeros(3, dtype=torch.bool, device=scalar.device)),
            "cutoff": scalar.new_tensor(self.cutoff),
            "field": field,
            "uniform_field": uniform_field,
            "local_external_field": local_field,
            "network_field": network_field.reshape(n_atoms, self.components),
            "reference_field": reference_field,
            "origin": origin.reshape(n_atoms, self.components),
            "constraint_matrix": compiled.constraint_matrix,
            "constraint_target": compiled.constraint_target,
            "control_matrix": compiled.control_matrix,
            "control_value": compiled.control_value,
            "environment_matrix": compiled.environment_matrix,
            "environment_stiffness": compiled.environment_stiffness,
        }
        return context, compiled, start, stop

    def _problem(self, context, compiled, baseline, *, detached=False, projection_cache=None):
        n_atoms = context["scalar"].shape[0]
        if detached:
            context = {key: value.detach() for key, value in context.items()}
            compiled = replace(compiled, **{
                field.name: getattr(compiled, field.name).detach()
                for field in fields(compiled)
                if isinstance(getattr(compiled, field.name), torch.Tensor)
            })
            weights = {name: value.detach() for name, value in self.internal_energy.named_parameters()}
            buffers = {name: value.detach().clone() for name, value in self.internal_energy.named_buffers()}
            # Baseline is independent of x. Omitting it avoids even a zero-gradient
            # path to physical parameters in a residual-only backward/AdamW step.
            def energy(state):
                return torch.func.functional_call(self.internal_energy, (weights, buffers),
                                                  (context, state.reshape(n_atoms, self.components))) + compiled.energy(state) - (
                    (context["field"] - context["reference_field"]).reshape(-1) @ state)
        else:
            def energy(state):
                return baseline + self.internal_energy(context, state.reshape(n_atoms, self.components)) + compiled.energy(state) - (
                    (context["field"] - context["reference_field"]).reshape(-1) @ state)
        if projection_cache is not None:
            if detached:
                raise ValueError("cannot reuse a live projection for detached residual training")
            return projection_cache["problem"].with_energy(energy)
        return FunctionalProblem(energy, n_atoms * self.components,
                                 compiled.constraint_matrix, compiled.constraint_target)

    @staticmethod
    def _pack(values, shapes, unit):
        ptr = [0]
        for value in values:
            ptr.append(ptr[-1] + value.numel())
        return {"values": torch.cat([value.reshape(-1) for value in values]),
                "ptr": torch.tensor(ptr, dtype=torch.long, device=values[0].device),
                "shape": tuple(shapes), "unit": unit}

    def _field_response(self, problem, state, field):
        """Optional complete state/field Jacobian; never used by residual training."""

        create_graph = self.training or self.retain_response_graph
        if self.state_mode == "predicted":
            outputs = state
        else:
            point = state.detach().requires_grad_(True)
            outputs = problem.state_gradient(point, create_graph=True)
        rows = []
        for value in outputs:
            gradient = torch.autograd.grad(value, field, create_graph=create_graph,
                                           retain_graph=True, allow_unused=True)[0]
            rows.append(torch.zeros_like(field).reshape(-1) if gradient is None else gradient.reshape(-1))
        jacobian = torch.stack(rows)
        if self.state_mode == "predicted":
            return jacobian
        from curator.variational.functional_response import solve_functional_linear_response
        response = solve_functional_linear_response(problem, state.detach(), -jacobian,
                                                    create_graph=create_graph)
        if not response.converged:
            raise RuntimeError("internal field-response solve did not converge")
        return response.state

    @staticmethod
    def _observable_inputs(data, context, geometry):
        inputs = {
            "external_field": context["uniform_field"],
            "local_external_field": context["local_external_field"],
            "control_value": context["control_value"],
            "constraint_target": context["constraint_target"],
        }
        if "_observable_positions" in data:
            inputs["positions"] = data["_observable_positions"][geometry]
        if "_observable_strains" in data:
            inputs["strain"] = data["_observable_strains"][geometry]
        normalizers = {}
        if "_observable_volumes" in data:
            normalizers["volume"] = data["_observable_volumes"][geometry]
        return inputs, normalizers

    def _observables(self, energy, data, context, geometry, *, problem=None, state=None, multipliers=None):
        """Execute derivative specifications without dispatching on property names."""
        if not self.observable_specs:
            return {}
        if any("positions" in spec.derivatives or "strain" in spec.derivatives for spec in self.observable_specs):
            if "_observable_positions" not in data:
                raise ValueError("coordinate observables require DifferentiableInputs before the GNN")
        inputs, normalizers = self._observable_inputs(data, context, geometry)
        if self.state_mode == "converged":
            from curator.variational.stationary_observables import StationaryEnergyObservables

            executor = StationaryEnergyObservables(problem, state, multipliers, inputs, normalizers=normalizers)
        else:
            executor = EnergyObservables(
                energy, inputs, create_graph=self.training or self.retain_response_graph,
                state_mode=self.state_mode, normalizers=normalizers,
            )
        return executor.evaluate(self.observable_specs)

    def _batched_observables(self, records, data):
        instances = [self._observable_inputs(data, context, geometry)
                     for _, context, geometry in records]
        inputs = {name: tuple(value[0][name] for value in instances) for name in instances[0][0]}
        normalizers = {name: tuple(value[1][name] for value in instances) for name in instances[0][1]}
        # Ragged coordinate Hessians and other large first-input axes keep the
        # original execution policy, rather than silently creating dense blocks.
        for spec in self.observable_specs:
            if len(spec.derivatives) == 2:
                first = spec.derivatives[0]
                sizes = [value.numel() for value in inputs[first]]
                if len(set(sizes)) != 1 or sizes[0] > 9:
                    outputs = [self._observables(energy, data, context, geometry)
                               for energy, context, geometry in records]
                    return {spec.name: tuple(output[spec.name] for output in outputs)
                            for spec in self.observable_specs}
        return BatchedEnergyObservables(
            tuple(energy for energy, _, _ in records), inputs, normalizers=normalizers,
            create_graph=self.training or self.retain_response_graph,
            state_mode=self.state_mode,
        ).evaluate(self.observable_specs)

    def forward(self, data):
        # Linear algebra and higher derivatives run in the model's master dtype.
        # The GNN may still run under mixed-precision autocast before this head.
        with torch.autocast(device_type=data[properties.positions].device.type, enabled=False):
            return self._forward(data)

    def _forward(self, data):
        deployment = getattr(self, "_deployment_options", None)
        if (getattr(self.internal_energy, "requires_shared_context_reuse", False)
                and self.residual_evaluation != "reuse"):
            raise ValueError("shared proposal context requires residual_evaluation='reuse'")
        if deployment is not None and (self.training or self.state_mode != "predicted"):
            raise RuntimeError("deployment is predicted evaluation only; call reset_deployment() before training")
        energy_only = deployment is not None and deployment["energy_only"]
        compute_residuals = (self.state_semantics == "variational_internal"
                             and (deployment is None or deployment["compute_residuals"]))
        compute_forces = self.compute_forces and not energy_only
        compute_responses = self.compute_responses and not energy_only
        observable_specs = () if energy_only else self.observable_specs
        if not self._is_bound:
            raise RuntimeError("VariationalEnergyOutput must bind to GNN features before training")
        if self.training and self.state_mode != "predicted":
            raise RuntimeError("joint training requires predicted state_mode; converged is evaluation only")
        if self.state_mode == "converged" and self.retain_response_graph and compute_forces:
            raise RuntimeError("converged forces support first derivatives only; use the linear-response/HVP "
                               "path for relaxed second derivatives, or compute_responses=True for field response")
        if self.state_mode not in self.supports_state_modes:
            raise ValueError("unsupported state_mode")
        if self.state_semantics == "predicted_auxiliary" and self.state_mode != "predicted":
            raise ValueError("latent states require predicted state_mode; no variational solve is defined")
        no_grad_energy = (
            energy_only and not compute_residuals
            and getattr(self.internal_energy, "derivative_backend", "autograd") == "analytic"
        )
        if torch.is_inference_mode_enabled() or (not torch.is_grad_enabled() and not no_grad_energy):
            raise RuntimeError("variational GNN evaluation requires gradients and inference_mode=False; "
                               "no_grad is supported only for analytic energy-only deployment")
        scalar, vector, features = self._features(data)
        reference = next(self.proposal.parameters(), data[properties.positions])
        scalar, vector, features = scalar.to(reference), vector.to(reference), features.to(reference)
        positions = data[properties.positions]
        if compute_forces and not positions.requires_grad:
            raise RuntimeError("enable PairwiseDistance(compute_distance_from_R=True, compute_forces=True)")
        counts = data[properties.n_atoms].reshape(-1)
        atom_ptr = nn.functional.pad(counts.cumsum(0), (1, 0))
        baselines = data[self.baseline_key].reshape(-1).to(reference)
        if baselines.numel() != counts.numel():
            raise ValueError("baseline readout must provide one scalar energy per geometry")
        indices, boundaries = self._conditions(data)
        batch_observables = (
            self.batch_observables and bool(observable_specs) and self.state_mode == "predicted"
            and indices.numel() > 1 and indices.unique().numel() == indices.numel()
        )
        batch_forces = (
            (deployment is None or deployment.get("batch_forces", False))
            and self.batch_observables and compute_forces
            and self.state_mode == "predicted" and indices.numel() > 1
            and indices.unique().numel() == indices.numel()
        )
        force_slices, force_energies = [], []
        observable_records = []
        energies, forces, states, residuals, diagnostics, shapes, responses = [], [], [], [], [], [], []
        feasibility = []
        initial_states, proposal_deltas = [], []
        correction_mode = getattr(self, "proposal_mode", "absolute") == "correction"
        if correction_mode and not callable(getattr(self.internal_energy, "initial_state", None)):
            raise ValueError("correction proposal requires internal_energy.initial_state(context)")
        observable_values = {spec.name: [] for spec in observable_specs}
        with torch.profiler.record_function("acvr.geometry_context"):
            geometry_contexts = (self.internal_energy.geometry_context(data)
                                 if hasattr(self.internal_energy, "geometry_context") else None)
        for condition, (geometry_tensor, boundary) in enumerate(zip(indices, boundaries)):
            geometry = int(geometry_tensor)
            projection_cache = ({} if deployment is not None
                                and deployment.get("reuse_constraint_projection", False) else None)
            with torch.profiler.record_function("acvr.boundary_context"):
                context, compiled, start, stop = self._context(
                    data, scalar, vector, geometry, boundary, atom_ptr, condition,
                    projection_cache=projection_cache)
            context["features"] = features[start:stop]
            if geometry_contexts is not None:
                context.update(geometry_contexts[geometry])
            if hasattr(self.internal_energy, "coefficients"):
                context.update(self.internal_energy.coefficients(context))
            if correction_mode:
                initial = self.internal_energy.initial_state(context)
                if (not isinstance(initial, torch.Tensor)
                        or initial.shape != (stop - start, self.components)
                        or initial.dtype != scalar.dtype or initial.device != scalar.device
                        or not torch.isfinite(initial).all()):
                    raise ValueError("initial_state must be a finite per-atom state tensor matching feature dtype/device")
                # Raw geometry-only anchor stays independent of boundary controls.
                context["initial_state"] = initial
                context["expansion_state"] = initial
            prepare = getattr(self.internal_energy, "prepare_proposal_context", None)
            if prepare is not None:
                context = prepare(context, self.proposal)
            problem = self._problem(context, compiled, baselines[geometry],
                                    projection_cache=projection_cache)
            if correction_mode:
                # Let the unchanged proposal see a feasible seed in its origin slot.
                context["origin"] = problem.project_state(initial.reshape(-1)).reshape_as(initial)
            with torch.profiler.record_function("acvr.proposal"):
                delta = self.proposal(context).reshape(-1)
                candidate = initial.reshape(-1) + delta if correction_mode else delta
                predicted = problem.project_state(candidate)
            if correction_mode:
                initial_states.append(initial.reshape(-1))
                proposal_deltas.append(delta)
            state = predicted
            if self.state_mode == "converged":
                with torch.profiler.record_function("acvr.reference_solve"):
                    solved = solve_functional_reference(problem, predicted.detach())
                if not solved.converged or not solved.stable:
                    raise RuntimeError("reference internal solve failed: " + solved.status)
                state = solved.state.detach()
                diagnostics.append({"status": solved.status, "converged": solved.converged,
                                    "stable": solved.stable, "iterations": solved.iterations,
                                    "residual": float(solved.residuals.stationarity_norm)})
            if (compute_residuals and self.residual_evaluation == "reuse"
                    and self.state_mode == "predicted" and not state.requires_grad):
                state = state.detach().requires_grad_(True)
            energy = problem.energy(state)
            derivative_energy = energy
            if self.state_mode == "converged":
                derivative_energy = energy + solved.multipliers.detach() @ problem.constraints(state)
            if batch_observables:
                observable_records.append((derivative_energy, context, geometry))
            elif observable_specs:
                for name, value in self._observables(
                    derivative_energy, data, context, geometry, problem=problem, state=state,
                    multipliers=solved.multipliers if self.state_mode == "converged" else None,
                ).items():
                    observable_values[name].append(value)
            label_energy = energy
            if self.energy_convention == "internal":
                # Remove the same boundary work used in _problem without
                # evaluating the expensive internal-energy network a second time.
                label_energy = energy - compiled.energy(state) + (
                    (context["field"] - context["reference_field"]).reshape(-1) @ state)
            energies.append(label_energy.reshape(1))
            states.append(state)
            feasibility.append(problem.constraints(state).detach())
            shapes.append((stop - start, self.components))
            if compute_responses:
                responses.append(self._field_response(problem, state, context["field"]))
            if batch_forces:
                force_slices.append((start, stop))
                force_energies.append(energy.reshape(1))
            elif compute_forces:
                force_energy = energy
                if self.state_mode == "converged":
                    force_energy = force_energy + solved.multipliers.detach() @ problem.constraints(state)
                gradient = torch.autograd.grad(force_energy, positions,
                                               create_graph=self.training or self.retain_response_graph,
                                               retain_graph=True, allow_unused=True)[0]
                forces.append(-(torch.zeros_like(positions) if gradient is None else gradient)[start:stop])
            if compute_residuals:
                reuse = self.residual_evaluation == "reuse" and self.state_mode == "predicted"
                if (reuse or self.residual_gradient_policy == "joint"
                        or (not self.training and self.residual_evaluation == "reuse")):
                    residual_problem, residual_state = problem, state
                else:
                    detached_context = {key: value.detach() for key, value in context.items()}
                    residual_problem = self._problem(detached_context, compiled, baselines[geometry], detached=True)
                    if self.state_mode == "predicted":
                        residual_candidate = self.proposal(detached_context).reshape(-1)
                        if correction_mode:
                            residual_candidate = residual_candidate + detached_context["initial_state"].reshape(-1)
                        residual_state = residual_problem.project_state(residual_candidate)
                        # Both branches must evaluate the identical physical state.
                        if not torch.allclose(residual_state.detach(), predicted.detach(), rtol=0, atol=0):
                            raise RuntimeError("proposal branches disagree; deterministic state head required")
                    else:
                        residual_state = state.detach().requires_grad_(True)
                residuals.append(residual_problem.stationarity(
                    residual_state, create_graph=self.training,
                    energy=energy if reuse else None).stationarity / self.residual_scale)
        if batch_forces:
            # Every energy has a disjoint geometry graph by the explicit
            # batch_observables contract. Different conditions of the same
            # geometry must keep the ordinary per-condition derivative path.
            gradient = torch.autograd.grad(
                torch.cat(force_energies).sum(), positions,
                create_graph=self.training or self.retain_response_graph,
                retain_graph=True, allow_unused=True,
            )[0]
            all_forces = -(torch.zeros_like(positions) if gradient is None else gradient)
            forces = [all_forces[start:stop] for start, stop in force_slices]
        if observable_records:
            observable_values = self._batched_observables(observable_records, data)
        labels = {
            "condition_energy": self._pack(energies, [(1,)] * len(energies), self.energy_unit),
            "variable_value": self._pack(states, shapes, self.state_unit),
        }
        if correction_mode:
            labels["initial_variable_value"] = self._pack(initial_states, shapes, self.state_unit)
            labels["proposal_delta"] = self._pack(proposal_deltas, shapes, self.state_unit)
        if compute_forces:
            labels["condition_forces"] = self._pack(forces, [(shape[0], 3) for shape in shapes], self.force_unit)
        for spec in observable_specs:
            values = observable_values[spec.name]
            labels[spec.name] = self._pack(values, [tuple(value.shape) for value in values], spec.unit)
        state_ptr = labels["variable_value"]["ptr"]
        if "variational" in data and not torch.equal(state_ptr, data["variational"]["condition_state_ptr"].to(state_ptr)):
            raise ValueError("condition state layout differs from variational batch")
        prediction = {
            "value_space": "physical", "labels": labels,
            "residuals": {"stationarity": {
                "values": torch.cat(residuals), "ptr": state_ptr,
                "source": "variational_energy_gradient", "gradient_policy": self.residual_gradient_policy,
                "requires_parameter_routing": (self.training and self.residual_evaluation == "reuse"
                                               and self.residual_gradient_policy == "proposal_only"),
                "normalization": {"kind": "fixed_physical_scale", "scale": self.residual_scale,
                                  "unit": self.conjugate_unit},
            }} if compute_residuals else {},
            "metadata": {"training_mode": "joint", "energy_state": self.state_mode,
                         "energy_convention": self.energy_convention,
                         "force_potential": "thermodynamic",
                         "batched_force_derivative": batch_forces,
                         "force_derivative": ("full_proposal_path" if self.state_mode == "predicted" else "stationary_envelope"),
                         "derivative_order": (2 if self.retain_response_graph and self.state_mode == "predicted" else 1),
                         "residual_gradient_policy": self.residual_gradient_policy, "state_kind": self.state_kind,
                         "components": self.components},
            "diagnostics": diagnostics,
            "constraint_violation": self._pack(feasibility, [tuple(value.shape) for value in feasibility], self.state_unit),
            "condition_geometry_index": indices,
        }
        if self.state_semantics == "predicted_auxiliary":
            prediction["metadata"].update(
                state_semantics=self.state_semantics, training_mode="supervised_latent",
                residuals_computed=False, residual_gradient_policy="not_computed")
        if deployment is not None:
            prediction["metadata"]["deployment"] = True
            prediction["metadata"]["residuals_computed"] = compute_residuals
            prediction["metadata"]["energy_only"] = energy_only
            if not compute_residuals:
                prediction["metadata"]["residual_gradient_policy"] = "not_computed"
            if energy_only:
                prediction["metadata"]["force_derivative"] = "not_computed"
                prediction["metadata"]["derivative_order"] = 0
        if correction_mode:
            prediction["metadata"]["proposal_mode"] = "initial_state_plus_correction"
            prediction["metadata"]["initial_variable_value"] = "raw_geometry_state"
            prediction["metadata"]["proposal_delta"] = "unprojected_correction"
        if observable_specs:
            prediction["metadata"]["observables"] = tuple(asdict(spec) for spec in observable_specs)
            prediction["metadata"]["observable_derivative"] = (
                "full_proposal_path" if self.state_mode == "predicted" else "implicit_stationary_response")
            prediction["metadata"]["derivative_order"] = max(
                prediction["metadata"]["derivative_order"],
                max(len(spec.derivatives) for spec in observable_specs),
            )
        if responses:
            packed = self._pack(responses, [tuple(value.shape) for value in responses], self.response_unit)
            variational = data.get("variational", {})
            prediction["responses"] = {
                "values": packed["values"], "ptr": packed["ptr"],
                "shape": packed["shape"], "unit": (self.response_unit,) * len(responses),
                "condition_index": torch.arange(len(responses), device=scalar.device),
                "input_ref": ("field",) * len(responses), "output_ref": ("variable_value",) * len(responses),
                "gauge": variational.get("gauge", ({"kind": "absolute_conjugate_field"},) * len(responses)),
                "boundary_hash": variational.get("boundary_hash", tuple(canonical_boundary_hash(asdict(value)) for value in boundaries)),
            }
            prediction["metadata"]["response_derivative"] = (
                "proposal_jacobian" if self.state_mode == "predicted" else "implicit_stationary_response")
        data["variational_prediction"] = prediction
        # Keep c available for internal diagnostics, but never expose it as the
        # public total-energy prediction through NeuralNetworkPotential.
        data["variational_baseline_energy"] = baselines
        return data


__all__ = ["VariationalEnergyOutput", "VariationalStateHead", "StableInternalEnergy"]
