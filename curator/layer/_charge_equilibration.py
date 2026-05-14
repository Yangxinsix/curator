from ._atomwise_nn import AtomwiseNN
from ._ewald import EwaldSummation
from curator.data import properties
from typing import Union, Type, Optional, List, Dict
from functools import partial
from ase.data import atomic_numbers, chemical_symbols
import torch
from torch import nn
try:
    from torch_scatter import scatter_add
except ImportError:
    from curator.utils import scatter_add

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
        model_outputs: List[str] = ['residual_forces', 'atomic_charge'],
        ewald_weight = 1.0,
        ewald_weight_trainable: bool = False,
        compute_residual_forces_mode: str = "training",
        reference_electronegativity: Union[Dict[int, float], Dict[str, float], None, str] = None,
        reference_hardness: Union[Dict[int, float], Dict[str, float], None, str] = None,
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
        # Giving electronegativity and hardness a reference value, preventing it becoming too small
        if reference_electronegativity == "auto":
            ref_chi_dict = torch.ones((119,), dtype=torch.float)
            self.register_buffer("ref_chi_dict", ref_chi_dict)
        elif isinstance(reference_electronegativity, Dict):
            ref_chi_dict = torch.zeros((119,), dtype=torch.float)
            if reference_electronegativity is not None:
                # convert chemical symbols to atomic numbers
                for k, v in reference_electronegativity.items():
                    if isinstance(k, str):
                        ref_chi_dict[atomic_numbers[k]] = v
                    else:
                        ref_chi_dict[k] = v
            self.register_buffer("ref_chi_dict", ref_chi_dict)
        else:
            ref_chi_dict = torch.zeros((119,), dtype=torch.float)
            self.register_buffer("ref_chi_dict", ref_chi_dict)

        if reference_hardness == "auto":
            ref_hardness_dict = torch.ones((119,), dtype=torch.float)
            self.register_buffer("ref_hardness_dict", ref_hardness_dict)
        elif isinstance(reference_hardness, Dict):
            ref_hardness_dict = torch.zeros((119,), dtype=torch.float)
            if reference_hardness is not None:
                # convert chemical symbols to atomic numbers
                for k, v in reference_hardness.items():
                    if isinstance(k, str):
                        ref_hardness_dict[atomic_numbers[k]] = v
                    else:
                        ref_hardness_dict[k] = v
            self.register_buffer("ref_hardness_dict", ref_hardness_dict)
        else:
            ref_hardness_dict = torch.zeros((119,), dtype=torch.float)
            self.register_buffer("ref_hardness_dict", ref_hardness_dict)

        if isinstance(ewald, EwaldSummation):
            self.ewald = ewald
        else:
            self.ewald = ewald(cutoff=cutoff, k_cutoff=k_cutoff, alpha=alpha, acc_factor=acc_factor)

        self.compute_forces = compute_forces
        if compute_residual_forces_mode not in {"training", "always", "never"}:
            raise ValueError(
                "compute_residual_forces_mode must be one of "
                "'training', 'always', or 'never'"
            )
        self.compute_residual_forces_mode = compute_residual_forces_mode
        self.constant_potential = constant_potential
        self.model_outputs = model_outputs
        if ewald_weight_trainable:
            self.ewald_weight = torch.nn.Parameter(torch.tensor(ewald_weight))
        else:
            self.register_buffer("ewald_weight", torch.tensor(ewald_weight))
        self.ewald_weight_trainable = ewald_weight_trainable

    def forward(self, data: properties.Type, training: Optional[bool] = None) -> properties.Type:
        if training is None:
            training = self.training

        data = data.copy()
        chi = self.electronegativity_mlp._compute(data[properties.node_embedding]).squeeze()
        hardness = self.hardness_mlp._compute(data[properties.node_embedding]).squeeze()

        chi0 = self.ref_chi_dict[data[properties.Z]]
        hardness0 = self.ref_hardness_dict[data[properties.Z]]

        # processing charges to make sure sum_i {q_i} = q_total
        # consider adding an upper bound for charge predictions here
        sum_charge = scatter_add(data[properties.atomic_charge], data[properties.image_idx], dim=0)
        total_charge = torch.zeros(1, dtype=sum_charge.dtype, device=sum_charge.device) if properties.total_charge not in data else data[properties.total_charge]
        diff_charge = (total_charge - sum_charge) / data[properties.n_atoms]
        data[properties.atomic_charge] = data[properties.atomic_charge] + torch.gather(diff_charge, 0, data[properties.image_idx])

        # calculate residual energy (per-atom first, then aggregate)
        atomic_residual_energy = (chi ** 2 + chi0) * data[properties.atomic_charge] + (hardness ** 2 + hardness0) * data[properties.atomic_charge] ** 2    # use square to ensure that both values are positive
        residual_energy = scatter_add(atomic_residual_energy, data[properties.image_idx], dim=0)

        # In LAMMPS mode with kspace, skip Ewald calculation (LAMMPS handles it)
        # Detect LAMMPS mode by presence of lammps_data key
        use_lammps_kspace = properties.lammps_data in data
        if use_lammps_kspace:
            # Return zero Ewald energy - LAMMPS kspace will compute the actual value
            ewald_energy = torch.zeros_like(residual_energy)
        else:
            # calculate ewald energy, total energy = local + ewald + residual
            ewald_energy = self.ewald(data)

        data[properties.short_energy] = data[properties.energy]
        data[properties.atomic_residual_energy] = atomic_residual_energy  # per-atom, needed for LAMMPS MLIAP
        data[properties.residual_energy] = residual_energy  # per-structure (sum of atomic_residual_energy)
        data[properties.ewald_energy] = ewald_energy
        data[properties.energy] = data[properties.energy] + (ewald_energy + residual_energy) * self.ewald_weight
        # data[properties.electrostatic_energy] = ewald_energy + residual_energy

        # calculate Ewald forces.  The physical QEq force is the fixed-charge
        # Ewald force, because the charge-response terms from residual and
        # Ewald energies cancel at charge equilibrium.
        if self.compute_forces:
            grad_outputs : List[Optional[torch.Tensor]] = [torch.ones_like(ewald_energy)]
            compute_residual_forces_mode = getattr(self, "compute_residual_forces_mode", "training")
            compute_residual_forces = (
                compute_residual_forces_mode == "always"
                or (compute_residual_forces_mode == "training" and training)
            )
            force_data = data.copy()
            force_data[properties.atomic_charge] = data[properties.atomic_charge].detach()
            fixed_charge_ewald_energy = torch.zeros_like(ewald_energy) if use_lammps_kspace else self.ewald(force_data)
            ewald_grad = torch.autograd.grad(
                fixed_charge_ewald_energy,
                data[properties.positions],
                grad_outputs=grad_outputs,
                retain_graph=training or compute_residual_forces,
                create_graph=training,
                allow_unused=True,
            )
            ewald_forces = torch.zeros_like(data[properties.positions]) if ewald_grad[0] is None else -ewald_grad[0]   # for torch.jit.script
            assert ewald_forces is not None

            if compute_residual_forces:
                # residual_forces is the charge-response derivative residual that
                # should vanish at strict QEq equilibrium and is used as a
                # zero-target loss term:
                #
                #   dE_res/dR + (dE_ewald/dR - ∂E_ewald/∂R|q)
                #
                # Here ∂E_res/∂R|q = 0 for coordinate-independent chi/eta, but
                # dE_res/dR includes charge response.  The Ewald charge-response
                # term is obtained as the difference between relaxed-q and
                # fixed-q Ewald derivatives.
                residual_grad = torch.autograd.grad(
                    residual_energy,
                    data[properties.positions],
                    grad_outputs=grad_outputs,
                    retain_graph=True,
                    create_graph=training,
                    allow_unused=True,
                )
                residual_grad = torch.zeros_like(data[properties.positions]) if residual_grad[0] is None else residual_grad[0]

                relaxed_ewald_energy = torch.zeros_like(ewald_energy) if use_lammps_kspace else self.ewald(data)
                relaxed_ewald_grad = torch.autograd.grad(
                    relaxed_ewald_energy,
                    data[properties.positions],
                    grad_outputs=grad_outputs,
                    retain_graph=training,
                    create_graph=training,
                    allow_unused=True,
                )
                relaxed_ewald_grad = torch.zeros_like(data[properties.positions]) if relaxed_ewald_grad[0] is None else relaxed_ewald_grad[0]
                fixed_ewald_grad = torch.zeros_like(data[properties.positions]) if ewald_grad[0] is None else ewald_grad[0]
                residual_forces = residual_grad + relaxed_ewald_grad - fixed_ewald_grad
            else:
                residual_forces = torch.zeros_like(data[properties.positions])

            assert residual_forces is not None

            data[properties.ewald_forces] = ewald_forces
            data[properties.forces] = data[properties.forces] + ewald_forces * self.ewald_weight
            data[properties.residual_forces] = residual_forces
        
        return data