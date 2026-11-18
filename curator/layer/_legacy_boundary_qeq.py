"""Historical direct-q QEq with explicit isolated/periodic electrostatics.

Only the kernel's boundary handling changes from the archived handoff: FFF
uses complete isolated Coulomb; TTT uses complete 3D Ewald. The historical
chi/k heads, uniform charge projection, fixed-charge force and force-response
penalty are inherited unchanged. No extra learned coupling or charge labels.
"""
import torch
from curator.data import properties
from curator.layer._legacy_charge_equilibration import LegacyChargeEquilibration
from curator.layer._legacy_ewald import EwaldSummation as HistoricalEwald
from curator.layer._ewald import EwaldSummation
from curator.variational.backends.ewald import EwaldChargeOperator


class BoundaryAwareLegacyEwald(HistoricalEwald):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.physical_ewald = EwaldSummation(cutoff=self.cutoff,
            k_cutoff=self.k_cutoff, alpha=self.alpha, acc_factor=self.acc_factor)

    def forward(self, data, ewald_kernel=None):
        kernels = EwaldChargeOperator(self.physical_ewald).geometry_kernels(data)
        chunks = data[properties.atomic_charge].split(data[properties.n_atoms].tolist())
        return torch.stack([0.5 * q @ kernel @ q for q, kernel in zip(chunks, kernels)])


class BoundaryAwareLegacyQEq(LegacyChargeEquilibration):
    def __init__(self, num_features=128, **kwargs):
        super().__init__(num_features=num_features,
            ewald=BoundaryAwareLegacyEwald(), **kwargs)


class ConservativeLegacyQEqControl(BoundaryAwareLegacyQEq):
    """Explicit control: same old parameterization, full dE/dR and KKT loss.

    This is NOT the historical force/objective baseline. It isolates those
    choices without changing chi/k/q heads or adding an exact QEq solver.
    """
    def forward(self, data, training=None):
        actual_training=self.training if training is None else training
        # Retain charge paths needed by the differentiable KKT loss and force.
        result=super().forward(data, training=True)
        result[properties.forces]=result[properties.forces]-result[properties.residual_forces]*self.ewald_weight
        charge=result[properties.atomic_charge]
        potential=torch.autograd.grad((result['residual_energy']+result[properties.ewald_energy]).sum(),
            charge,create_graph=actual_training,retain_graph=True)[0]
        sizes=result[properties.n_atoms].tolist()
        result[properties.chemical_potential_residual]=torch.cat([p-p.mean() for p in potential.split(sizes)])
        result["legacy_force_semantics"]="conservative_full_dE_dR"
        return result
