"""Boundary-conditioned charge runtime with energy-consistent derivatives."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from curator.data import properties
from curator.layer._ewald import EwaldSummation

from ..amortized import solve_amortized
from ..backends.ewald import EwaldChargeOperator
from ..compiler import CompiledBoundary
from ..forces import ForceMode, differentiate_variational_energy
from ..implicit import solve_implicit
from ..validation import minimum_reduced_curvature
from .charge import ChargeFunctional


@dataclass(frozen=True)
class ChargeRuntimeResult:
    """Ragged states, thermodynamic energies, derivatives, and KKT certificates."""

    state: torch.Tensor
    condition_state_ptr: torch.Tensor
    multipliers: torch.Tensor
    multiplier_ptr: torch.Tensor
    thermodynamic_energy: torch.Tensor
    base_energy: torch.Tensor
    onsite_energy: torch.Tensor
    interaction_energy: torch.Tensor
    boundary_energy: torch.Tensor
    chemical_potential: torch.Tensor
    stationarity_residual: torch.Tensor
    primal_residual: torch.Tensor
    primal_ptr: torch.Tensor
    normalized_kkt_residual: torch.Tensor
    kkt_ptr: torch.Tensor
    minimum_reduced_curvature: torch.Tensor
    condition_geometry_index: torch.Tensor
    reliable: torch.Tensor
    status: Tuple[str, ...]
    condition_forces: Optional[torch.Tensor]
    condition_force_ptr: Optional[torch.Tensor]
    condition_virial: Optional[torch.Tensor]
    condition_stress: Optional[torch.Tensor]
    conservative_for_reported_energy: torch.Tensor
    force_mode: Optional[str]

    def as_variational_prediction(self, energy_label: Optional[str] = None) -> Dict:
        """Return the physical-unit prediction contract consumed by joint loss.

        Energy semantics are ensemble-dependent, so callers must explicitly
        choose ``condition_energy`` or ``condition_free_energy`` to expose it.
        With the default ``None`` only charge/force labels and KKT certificates
        are returned.
        """

        n_conditions = self.thermodynamic_energy.numel()
        state_counts = torch.diff(self.condition_state_ptr).detach().cpu().tolist()
        labels = {
            properties.variable_value: {
                "values": self.state,
                "ptr": self.condition_state_ptr,
                "shape": tuple((count, 1) for count in state_counts),
                "unit": "e",
            },
        }
        if energy_label is not None:
            if energy_label not in {
                properties.condition_energy,
                properties.condition_free_energy,
            }:
                raise ValueError("energy_label must be condition_energy or condition_free_energy")
            labels[energy_label] = {
                "values": self.thermodynamic_energy,
                "ptr": torch.arange(
                    n_conditions + 1,
                    dtype=torch.long,
                    device=self.thermodynamic_energy.device,
                ),
                "shape": tuple((1,) for _ in range(n_conditions)),
                "unit": "eV",
            }
        if self.condition_forces is not None:
            labels[properties.condition_forces] = {
                "values": self.condition_forces,
                "ptr": self.condition_force_ptr,
                "shape": tuple((count, 3) for count in state_counts),
                "unit": "eV/angstrom",
            }
        return {
            "value_space": "physical",
            "labels": labels,
            "certificates": {
                "kkt": {
                    "values": self.normalized_kkt_residual,
                    "ptr": self.kkt_ptr,
                    "eligible": torch.ones(
                        n_conditions,
                        dtype=torch.bool,
                        device=self.state.device,
                    ),
                    "source": "compiled_variational_operator",
                    "normalization": "per-condition KKT backward-error scales",
                }
            },
        }


class ACVRChargeRuntime(nn.Module):
    """Solve one or more charge boundaries for each geometry in a batch.

    ``condition_geometry_index`` allows many charge/voltage conditions to reuse
    one geometry, one local model evaluation, and one Ewald kernel.  The runtime
    reports the boundary-conditioned thermodynamic potential, not merely the
    base QEq energy.

    The legacy :class:`ChargeEquilibration` layer is intentionally untouched.
    This runtime is a new physical path whose force includes derivatives of
    electronegativity, hardness, Ewald, and every compiled boundary term.
    """

    def __init__(
        self,
        interaction,
        *,
        solver: str = "implicit",
        force_mode: Optional[ForceMode] = "envelope",
        residual_tolerance: float = 1.0e-8,
        feasibility_tolerance: float = 1.0e-10,
        max_correction_steps: int = 1,
        fallback: bool = True,
        projection_metric: str = "hardness",
    ) -> None:
        super().__init__()
        if isinstance(interaction, EwaldSummation):
            self.ewald = interaction
            interaction = EwaldChargeOperator(self.ewald)
        if not hasattr(interaction, "kernels"):
            raise TypeError("interaction must provide differentiable per-geometry kernels(data)")
        if solver not in {"implicit", "amortized"}:
            raise ValueError("solver must be 'implicit' or 'amortized'")
        if force_mode not in {None, "envelope", "full_differentiable", "adjoint_corrected"}:
            raise ValueError("unsupported force_mode")
        if projection_metric not in {"hardness", "identity"}:
            raise ValueError("projection_metric must be 'hardness' or 'identity'")
        if not math.isfinite(residual_tolerance) or residual_tolerance < 0:
            raise ValueError("residual_tolerance must be finite and nonnegative")
        if not math.isfinite(feasibility_tolerance) or feasibility_tolerance < 0:
            raise ValueError("feasibility_tolerance must be finite and nonnegative")
        if (
            isinstance(max_correction_steps, bool)
            or not isinstance(max_correction_steps, int)
            or max_correction_steps < 0
        ):
            raise ValueError("max_correction_steps must be a nonnegative integer")
        if not isinstance(fallback, bool):
            raise TypeError("fallback must be boolean")
        self.interaction = interaction
        self.solver = solver
        self.force_mode = force_mode
        self.residual_tolerance = float(residual_tolerance)
        self.feasibility_tolerance = float(feasibility_tolerance)
        self.max_correction_steps = max_correction_steps
        self.fallback = fallback
        self.projection_metric = projection_metric

    @staticmethod
    def _validate_parameters(data, electronegativity, hardness):
        counts = data[properties.n_atoms]
        if counts.dtype != torch.long or counts.ndim != 1 or torch.any(counts <= 0):
            raise ValueError("n_atoms must be a positive torch.long vector")
        if counts.device != electronegativity.device:
            raise ValueError("n_atoms and charge parameters must use the same device")
        n_atoms = int(counts.sum().item())
        if electronegativity.shape != (n_atoms,) or hardness.shape != (n_atoms,):
            raise ValueError("electronegativity and hardness must have one value per atom")
        if (
            not electronegativity.is_floating_point()
            or hardness.dtype != electronegativity.dtype
            or hardness.device != electronegativity.device
        ):
            raise ValueError("charge parameters must share a floating dtype and device")
        if not torch.isfinite(electronegativity).all() or not torch.isfinite(hardness).all():
            raise ValueError("charge parameters must be finite")
        if torch.any(hardness <= 0):
            raise ValueError("hardness must be positive")
        return counts

    @staticmethod
    def _condition_index(boundaries, counts, condition_geometry_index, device):
        n_conditions = len(boundaries)
        if not n_conditions:
            raise ValueError("at least one compiled boundary is required")
        if condition_geometry_index is None:
            if n_conditions != counts.numel():
                raise ValueError(
                    "condition_geometry_index is required unless there is one condition per geometry"
                )
            condition_geometry_index = torch.arange(
                n_conditions, dtype=torch.long, device=device
            )
        if (
            not isinstance(condition_geometry_index, torch.Tensor)
            or condition_geometry_index.dtype != torch.long
            or condition_geometry_index.shape != (n_conditions,)
        ):
            raise TypeError("condition_geometry_index must be torch.long with one value per boundary")
        condition_geometry_index = condition_geometry_index.to(device=device)
        if int(condition_geometry_index.min()) < 0 or int(condition_geometry_index.max()) >= counts.numel():
            raise ValueError("condition_geometry_index is out of range")
        return condition_geometry_index

    @staticmethod
    def _offsets(counts):
        return torch.nn.functional.pad(torch.cumsum(counts, dim=0), (1, 0))

    def _solve(self, problem, proposal, hardness):
        if self.solver == "implicit":
            result = solve_implicit(
                problem,
                residual_tolerance=self.residual_tolerance,
            )
            status = "implicit_exact" if result.reliable else "implicit_uncertified"
            return result, status
        if proposal is None:
            raise ValueError("amortized charge solving requires one proposal per condition")
        metric = torch.diag(hardness) if self.projection_metric == "hardness" else None
        result = solve_amortized(
            problem,
            proposal,
            projection_metric=metric,
            residual_tolerance=self.residual_tolerance,
            feasibility_tolerance=self.feasibility_tolerance,
            max_correction_steps=self.max_correction_steps,
            fallback=self.fallback,
        )
        return result, result.status

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        electronegativity: torch.Tensor,
        hardness: torch.Tensor,
        boundaries: Sequence[CompiledBoundary],
        *,
        condition_geometry_index: Optional[torch.Tensor] = None,
        proposals: Optional[torch.Tensor] = None,
        create_graph: Optional[bool] = None,
    ) -> ChargeRuntimeResult:
        create_graph = self.training if create_graph is None else bool(create_graph)
        if any(not isinstance(boundary, CompiledBoundary) for boundary in boundaries):
            raise TypeError("boundaries must contain CompiledBoundary objects")
        counts = self._validate_parameters(data, electronegativity, hardness)
        condition_geometry_index = self._condition_index(
            boundaries, counts, condition_geometry_index, electronegativity.device
        )
        geometry_offsets = self._offsets(counts)
        state_counts = counts[condition_geometry_index]
        state_ptr = self._offsets(state_counts)
        if proposals is not None and proposals.shape != (int(state_ptr[-1].item()),):
            raise ValueError("proposals must contain one ragged state per condition")
        if proposals is not None and (
            proposals.dtype != electronegativity.dtype
            or proposals.device != electronegativity.device
        ):
            raise ValueError("proposals must match charge parameter dtype and device")
        if self.solver == "implicit" and proposals is not None:
            raise ValueError("implicit solving does not consume amortized proposals")
        if self.solver == "amortized" and proposals is None:
            raise ValueError("amortized charge solving requires proposals")

        kernels = self.interaction.kernels(data)
        if len(kernels) != counts.numel():
            raise ValueError("interaction kernel count does not match n_atoms")

        states = []
        multipliers = []
        multiplier_ptr = [0]
        onsite_energies = []
        interaction_energies = []
        total_energies = []
        chemical_potentials = []
        stationarity = []
        primal = []
        primal_ptr = [0]
        normalized_kkt = []
        kkt_ptr = [0]
        curvatures = []
        reliable = []
        statuses = []
        problems = []
        solver_results = []

        for condition, boundary in enumerate(boundaries):
            geometry = int(condition_geometry_index[condition].item())
            start = int(geometry_offsets[geometry].item())
            stop = int(geometry_offsets[geometry + 1].item())
            chi = electronegativity[start:stop]
            eta = hardness[start:stop]
            kernel = kernels[geometry]
            if kernel.shape != (stop - start, stop - start):
                raise ValueError("interaction kernel shape does not match its geometry")
            functional = ChargeFunctional(chi, eta, kernel)
            problem = functional.problem(boundary)
            proposal = None
            if proposals is not None:
                proposal_start = int(state_ptr[condition].item())
                proposal_stop = int(state_ptr[condition + 1].item())
                proposal = proposals[proposal_start:proposal_stop]
            result, status = self._solve(problem, proposal, eta)
            state = result.state
            base_mu = functional.chemical_potential(state)
            onsite = chi @ state + 0.5 * eta @ state.square()
            interaction_energy = 0.5 * state @ kernel @ state

            states.append(state)
            multipliers.append(result.multipliers)
            multiplier_ptr.append(multiplier_ptr[-1] + result.multipliers.numel())
            onsite_energies.append(onsite)
            interaction_energies.append(interaction_energy)
            total_energies.append(problem.energy(state))
            chemical_potentials.append(base_mu)
            stationarity.append(result.residuals.stationarity)
            primal.append(result.residuals.primal)
            primal_ptr.append(primal_ptr[-1] + result.residuals.primal.numel())
            normalized = torch.cat(
                (
                    result.residuals.stationarity
                    / (1.0 + result.residuals.stationarity_scale),
                    result.residuals.primal / (1.0 + result.residuals.primal_scale),
                )
            )
            normalized_kkt.append(normalized)
            kkt_ptr.append(kkt_ptr[-1] + normalized.numel())
            curvatures.append(
                result.minimum_reduced_curvature
                if hasattr(result, "minimum_reduced_curvature")
                else minimum_reduced_curvature(problem)
            )
            reliable.append(result.reliable)
            statuses.append(status)
            problems.append(problem)
            solver_results.append(result)

        condition_forces = None
        force_ptr = None
        condition_virial = None
        condition_stress = None
        conservative = [False] * len(boundaries)
        if self.force_mode is not None:
            coordinates = {"positions": data[properties.positions]}
            if not coordinates["positions"].requires_grad:
                raise ValueError("positions must require gradients when force_mode is enabled")
            if properties.strain in data:
                coordinates["strain"] = data[properties.strain]
            forces = []
            virials = []
            stresses = []
            cell = data[properties.cell].reshape(-1, 3, 3)
            volumes = torch.abs(torch.linalg.det(cell))
            for condition, (problem, result) in enumerate(zip(problems, solver_results)):
                derivative = differentiate_variational_energy(
                    objective_fn=problem.energy,
                    constraint_fn=lambda state, current=problem: (
                        current.constraint_matrix @ state - current.constraint_target
                    ),
                    coordinates=coordinates,
                    state=result.state,
                    multipliers=result.multipliers,
                    mode=self.force_mode,
                    residual_tolerance=self.residual_tolerance,
                    feasibility_tolerance=self.feasibility_tolerance,
                    create_graph=create_graph,
                )
                geometry = int(condition_geometry_index[condition].item())
                start = int(geometry_offsets[geometry].item())
                stop = int(geometry_offsets[geometry + 1].item())
                forces.append(derivative.forces[start:stop])
                conservative[condition] = derivative.conservative_for_reported_energy
                if "strain" in derivative.gradients:
                    strain_gradient = derivative.gradients["strain"][geometry]
                    order = (0, 4, 8, 5, 2, 1)
                    virials.append((-strain_gradient).reshape(-1)[list(order)])
                    stresses.append(
                        (strain_gradient / volumes[geometry]).reshape(-1)[list(order)]
                    )
            condition_forces = torch.cat(forces)
            force_ptr = 3 * state_ptr
            if virials:
                condition_virial = torch.stack(virials)
                condition_stress = torch.stack(stresses)

        state = torch.cat(states)
        total = torch.stack(total_energies)
        onsite = torch.stack(onsite_energies)
        interaction_energy = torch.stack(interaction_energies)
        device = state.device
        return ChargeRuntimeResult(
            state=state,
            condition_state_ptr=state_ptr,
            multipliers=torch.cat(multipliers) if multipliers else state.new_zeros(0),
            multiplier_ptr=torch.tensor(multiplier_ptr, dtype=torch.long, device=device),
            thermodynamic_energy=total,
            base_energy=onsite + interaction_energy,
            onsite_energy=onsite,
            interaction_energy=interaction_energy,
            boundary_energy=total - onsite - interaction_energy,
            chemical_potential=torch.cat(chemical_potentials),
            stationarity_residual=torch.cat(stationarity),
            primal_residual=torch.cat(primal) if primal else state.new_zeros(0),
            primal_ptr=torch.tensor(primal_ptr, dtype=torch.long, device=device),
            normalized_kkt_residual=torch.cat(normalized_kkt),
            kkt_ptr=torch.tensor(kkt_ptr, dtype=torch.long, device=device),
            minimum_reduced_curvature=torch.stack(curvatures),
            condition_geometry_index=condition_geometry_index,
            reliable=torch.tensor(reliable, dtype=torch.bool, device=device),
            status=tuple(statuses),
            condition_forces=condition_forces,
            condition_force_ptr=force_ptr,
            condition_virial=condition_virial,
            condition_stress=condition_stress,
            conservative_for_reported_energy=torch.tensor(
                conservative, dtype=torch.bool, device=device
            ),
            force_mode=self.force_mode,
        )


__all__ = ["ACVRChargeRuntime", "ChargeRuntimeResult"]
