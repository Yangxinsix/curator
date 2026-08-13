from functools import partial
from typing import List, Optional, Type, Union

import torch
from torch import nn

from curator.data import properties

from ._atomwise_nn import AtomwiseNN
from ._ewald import EwaldSummation

try:
    from torch_scatter import scatter_add
except ImportError:
    from curator.utils import scatter_add


class ChargeEquilibration(nn.Module):
    """Evaluate physical QEq quantities and enforce the total charge.

    The network predicts charges directly. Training can penalize
    ``chemical_potential_residual`` to make those charges approach the
    constrained minimum of the QEq energy without solving a linear system.

    Every energy, force, and chemical potential produced here is in physical
    units.  This module deliberately does not add those quantities to the
    normalized MLIP ``energy`` or ``forces`` outputs; that composition belongs
    to the output normalization boundary.
    """

    def __init__(
        self,
        num_features: Optional[int] = None,
        cutoff: Optional[float] = None,
        k_cutoff: Optional[float] = None,
        alpha: Optional[float] = 0.4,
        acc_factor: float = 12.0,
        electronegativity_mlp: Union[AtomwiseNN, Type[AtomwiseNN], partial] = AtomwiseNN,
        hardness_mlp: Union[AtomwiseNN, Type[AtomwiseNN], partial] = AtomwiseNN,
        ewald: Union[EwaldSummation, Type[EwaldSummation], partial] = EwaldSummation,
        compute_forces: bool = True,
        constant_potential: bool = False,
        min_hardness: float = 1.0e-6,
        model_outputs: Optional[List[str]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.electronegativity_mlp = self._build_readout(
            electronegativity_mlp, num_features
        )
        self.hardness_mlp = self._build_readout(hardness_mlp, num_features)
        self.ewald = (
            ewald
            if isinstance(ewald, EwaldSummation)
            else ewald(
                cutoff=cutoff,
                k_cutoff=k_cutoff,
                alpha=alpha,
                acc_factor=acc_factor,
            )
        )
        self.compute_forces = compute_forces
        self.constant_potential = constant_potential
        self.min_hardness = min_hardness
        self.model_outputs = (
            model_outputs
            if model_outputs is not None
            else [
                properties.chemical_potential_residual,
                properties.atomic_charge,
            ]
        )

    @staticmethod
    def _build_readout(readout, num_features: Optional[int]) -> AtomwiseNN:
        if isinstance(readout, AtomwiseNN):
            return readout
        if num_features is None:
            raise ValueError("num_features is required when constructing QEq readouts")
        return readout(
            in_features=num_features,
            n_hidden=num_features // 2,
            n_hidden_layers=1,
            out_features=1,
            use_e3nn=False,
            activation="silu",
        )

    @staticmethod
    def _conserve_total_charge(
        raw_charge: torch.Tensor,
        hardness: torch.Tensor,
        image_idx: torch.Tensor,
        total_charge: torch.Tensor,
    ) -> torch.Tensor:
        """Project charges with inverse-hardness (charge-susceptibility) weights."""
        predicted_total = scatter_add(raw_charge, image_idx, dim=0)
        charge_error = total_charge.reshape(-1) - predicted_total
        inverse_hardness = hardness.reciprocal()
        normalization = scatter_add(inverse_hardness, image_idx, dim=0)
        weights = inverse_hardness / normalization[image_idx]
        return raw_charge + weights * charge_error[image_idx]

    @staticmethod
    def _remove_structure_mean(
        values: torch.Tensor,
        image_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Project an atomwise scalar onto the fixed-total-charge subspace."""
        totals = scatter_add(values, image_idx, dim=0)
        counts = scatter_add(torch.ones_like(values), image_idx, dim=0)
        return values - (totals / counts)[image_idx]

    def forward(
        self,
        data: properties.Type,
        training: Optional[bool] = None,
    ) -> properties.Type:
        training = self.training if training is None else training
        data = data.copy()

        node_embedding = data[properties.node_embedding]
        raw_chi = self.electronegativity_mlp._compute(node_embedding).squeeze(-1)
        raw_hardness = self.hardness_mlp._compute(node_embedding).squeeze(-1)

        # Preserve the existing checkpoint parameterization:
        #   chi_raw^2 q + h_raw^2 q^2
        # while naming the physical QEq curvature explicitly as eta = 2 h_raw^2.
        chi = raw_chi.square()
        hardness = 2.0 * raw_hardness.square() + self.min_hardness

        raw_charge = data[properties.atomic_charge]
        predicted_total = scatter_add(raw_charge, data[properties.image_idx], dim=0)
        total_charge = data.get(
            properties.total_charge,
            torch.zeros_like(predicted_total),
        )
        charge = self._conserve_total_charge(
            raw_charge,
            hardness,
            data[properties.image_idx],
            total_charge,
        )
        data[properties.atomic_charge] = charge

        atomic_onsite_energy = chi * charge + 0.5 * hardness * charge.square()
        onsite_energy = scatter_add(
            atomic_onsite_energy,
            data[properties.image_idx],
            dim=0,
        )
        ewald_energy = self.ewald(data)
        qeq_energy = onsite_energy + ewald_energy

        data[properties.onsite_energy] = onsite_energy
        data[properties.ewald_energy] = ewald_energy
        data[properties.qeq_energy] = qeq_energy

        chemical_potential = torch.autograd.grad(
            qeq_energy.sum(),
            charge,
            create_graph=training,
            retain_graph=True,
        )[0]
        chemical_potential_residual = self._remove_structure_mean(
            chemical_potential,
            data[properties.image_idx],
        )
        data[properties.chemical_potential] = chemical_potential
        data[properties.chemical_potential_residual] = (
            chemical_potential_residual
        )

        # Backward-compatible diagnostic. This is a single vector-Jacobian
        # product, not a second relaxed Ewald evaluation.
        if properties.residual_forces in self.model_outputs:
            residual_forces = torch.autograd.grad(
                charge,
                data[properties.positions],
                grad_outputs=chemical_potential_residual,
                create_graph=training,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if residual_forces is None:
                residual_forces = torch.zeros_like(data[properties.positions])
            data[properties.residual_forces] = residual_forces

        if self.compute_forces:
            fixed_charge_data = data.copy()
            fixed_charge_data[properties.atomic_charge] = charge.detach()
            fixed_charge_energy = self.ewald(fixed_charge_data)
            gradient_inputs = [data[properties.positions]]
            compute_cell_response = properties.strain in data
            if compute_cell_response:
                gradient_inputs.append(data[properties.strain])
            gradients = torch.autograd.grad(
                fixed_charge_energy.sum(),
                gradient_inputs,
                create_graph=training,
                retain_graph=training or compute_cell_response,
                allow_unused=compute_cell_response,
            )
            position_gradient = gradients[0]
            if position_gradient is None:
                position_gradient = torch.zeros_like(data[properties.positions])
            data[properties.ewald_forces] = -position_gradient

            if compute_cell_response:
                strain_gradient = gradients[1]
                if strain_gradient is None:
                    strain_gradient = torch.zeros_like(data[properties.strain])
                cell = data[properties.cell].reshape(-1, 3, 3)
                volumes = torch.abs(torch.linalg.det(cell))
                data[properties.ewald_virial] = (-strain_gradient).reshape(
                    -1, 9
                )[:, [0, 4, 8, 5, 2, 1]]
                data[properties.ewald_stress] = (
                    strain_gradient / volumes[:, None, None]
                ).reshape(-1, 9)[:, [0, 4, 8, 5, 2, 1]]

        return data
