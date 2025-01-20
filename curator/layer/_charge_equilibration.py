from ._atomwise_nn import AtomwiseNN
from ._ewald import EwaldSummation
from typing import Union, Type, Optional
from functools import partial
import torch
from torch import nn

class ChargeEquilibration(nn.Module):
    """This class implements charge equilibration scheme by calculating Ewald summation and residual energy.
    The residual forces that contributed by electronegativity and hardness energies should be zero under strict charge equilibration.
    This class should be used in combination with training that minimize the residual forces.
    Args:
        num_features: dimensionality of node features
    """
    def __init__(
        self,
        num_features: Optional[int] = None,
        cutoff: Optional[float] = None,
        k_cutoff: Optional[float] = None,
        alpha: Optional[float] = 0.4,     # this value is obtained from the mean value of over 600k entries in Materials Project data
        acc_factor: float = 12.0,
        electronegativity_mlp: Union[AtomwiseNN, Type[AtomwiseNN], partial] = AtomwiseNN,
        hardness_mlp: Union[AtomwiseNN, Type[AtomwiseNN], partial] = AtomwiseNN,
        ewald: Union[EwaldSummation, Type[EwaldSummation], partial] = EwaldSummation,
        compute_forces: bool = True,
        constant_potential: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if isinstance(electronegativity_mlp, AtomwiseNN):
            self.electronegativity_mlp = electronegativity_mlp
        else:
            self.electronegativity_mlp = electronegativity_mlp(
                in_features=num_features,
                n_hidden=num_features // 2,
                n_hidden_layers=1,
                out_features=1,
                use_e3nn=False,
                activation='silu',
            )
        if isinstance(hardness_mlp, AtomwiseNN):
            self.hardness_mlp = hardness_mlp
        else:
            self.hardness_mlp = hardness_mlp(
                in_features=num_features,
                n_hidden=num_features // 2,
                n_hidden_layers=1,
                out_features=1,
                use_e3nn=False,
                activation='silu',
            )
        if isinstance(ewald, EwaldSummation):
            self.ewald = ewald
        else:
            self.ewald = ewald(cutoff=cutoff, k_cutoff=k_cutoff, alpha=alpha, acc_factor=acc_factor)
        
        self.compute_forces = compute_forces
        self.constant_potential = constant_potential

    def forward(self, data: properties.Type, training: bool=True) -> properties.Type:
        chi = self.electronegativity_mlp._compute(data[properties.node_embedding]).squeeze()
        hardness = self.hardness_mlp._compute(data[properties.node_embedding]).squeeze()

        # processing charges to make sure sum_i {q_i} = q_total
        # consider adding an upper bound for charge predictions here
        sum_charge = scatter_add(data[properties.atomic_charge], data[properties.image_idx], dim=0)
        total_charge = torch.zeros(1, dtype=sum_charge.dtype, device=sum_charge.device) if properties.total_charge not in data else data[properties.total_charge]
        diff_charge = (total_charge - sum_charge) / data[properties.n_atoms]
        data[properties.atomic_charge] = data[properties.atomic_charge] + torch.gather(diff_charge, 0, data[properties.image_idx])

        # calculate residual energy
        residual_energy =  chi ** 2 * data[properties.atomic_charge] + hardness ** 2 * data[properties.atomic_charge] ** 2    # use square to ensure that both values are positive
        residual_energy = scatter_add(residual_energy, data[properties.image_idx], dim=0)

        # calculate ewald energy, total energy = local + ewald + residual
        ewald_energy = self.ewald(data)
        data[properties.energy] += ewald_energy + residual_energy

        # calculate residual forces, total force = local + residual, residual forces should be zero under strict charge equilibration scheme
        if self.compute_forces:
            ewald_forces = torch.autograd.grad(
                ewald_energy,
                data[properties.positions],
                retain_graph=training,
                create_graph=training,
            )
            residual_forces = torch.autograd.grad(
                residual_energy,
                data[properties.positions],
                retain_graph=training,
                create_graph=training,
            )
            data[properties.forces] += ewald_forces[0]
            data[properties.residual_forces] = residual_forces[0]
        
        return data