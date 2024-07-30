import torch
from typing import Tuple, Optional, List, Callable, Union
from curator.data import properties
from ase.stress import full_3x3_to_voigt_6_stress
        
class GradientOutput(torch.nn.Module):
    def __init__(
        self,
        grad_on_edge_diff: bool = True,
        grad_on_positions: bool = False,
        model_outputs: List[str] = ['forces'],       # properties that need to be calculated, can be forces, stress, virial, etc.
        update_callback: Optional[Callable] = None,  # Add a callback parameter
    ) -> None:
        # TODO: define a set for allowed model outputs
        super().__init__()
        self.grad_on_edge_diff = grad_on_edge_diff
        self.grad_on_positions = grad_on_positions
        self.update_callback = update_callback
        self.model_outputs = model_outputs
        self.check_outputs()

    @torch.jit.ignore
    def update_model_outputs(self, outputs: Union[List[str], str]):
        if isinstance(outputs, str):
            self.model_outputs.append(outputs)
        else:
            self.model_outputs.extend(outputs)
        # update parent model
        if self.update_callback:
            self.update_callback()
        self.check_outputs()

    @torch.jit.ignore
    def check_outputs(self):
        self.compute_forces = True if 'forces' in self.model_outputs else False
        self.compute_stress = True if 'stress' in self.model_outputs else False

    def forward(
        self,
        data: properties.Type,
        training: bool=True,
    ) -> properties.Type:
        
        if self.grad_on_edge_diff:
            energy = data[properties.energy]
            edge_diff = data[properties.edge_diff]
            forces_dim = int(torch.sum(data[properties.n_atoms]))
            edge_idx = data[properties.edge_idx]
            if self.compute_forces:
                grad_outputs : List[Optional[torch.Tensor]] = [torch.ones_like(energy)]    # for model deploy
                dE_ddiff = torch.autograd.grad(
                    [energy,],
                    [edge_diff,],
                    grad_outputs=grad_outputs,
                    retain_graph=training,
                    create_graph=training,
                )
                dE_ddiff = torch.zeros_like(data[properties.positions]) if dE_ddiff is None else dE_ddiff[0]   # for torch.jit.script
                assert dE_ddiff is not None
                
                # diff = R_j - R_i, so -dE/dR_j = -dE/ddiff, -dE/R_i = dE/ddiff
                i_forces = torch.zeros((forces_dim, 3), device=edge_diff.device, dtype=edge_diff.dtype)
                j_forces = torch.zeros_like(i_forces)
                i_forces.index_add_(0, edge_idx[:, 0], dE_ddiff)
                j_forces.index_add_(0, edge_idx[:, 1], -dE_ddiff)
                forces = i_forces + j_forces
                data[properties.forces] = forces

                # Reference: https://en.wikipedia.org/wiki/Virial_stress
                # This method calculates virials by giving pair-wise force components
                
                if self.compute_stress:
                    if properties.cell in data:
                        image_idx = data[properties.image_idx]
                        atomic_virial = torch.einsum("ij, ik -> ijk", edge_diff, dE_ddiff)           # I'm quite not sure if a negative sign should be added before dE_ddiff, but I think it should be right
                        cell = data[properties.cell].view(-1, 3, 3)
                        volumes = torch.sum(cell[:, 0] * cell[:, 1].cross(cell[:, 2], dim=-1), dim=1)
                        # stress = torch.zeros_like(cell).index_add(0, , atomic_stress)
                        atomic_virial = torch.zeros(
                            (forces_dim, 3, 3),                                         
                            dtype=forces.dtype,
                            # it seens like the calculation is not very right... because f_ij is not absolutely right here. Maybe we need to do something like in force calculation
                            # add i_stress and j_stress together then it is the total stress. need verification
                            device=forces.device).index_add(0, edge_idx[:, 0], atomic_virial)
                        # j_stress = torch.zeros_like(i_stress).index_add(0, edge_idx[:, 1], -atomic_stress)
                        # atomic_stress = i_stress + j_stress          
                        virial = torch.zeros_like(cell).index_add(0, image_idx, atomic_virial)  # don't need to divide by two
                        stress = - virial / volumes[:, None, None]
                        data[properties.virial] = virial
                        data[properties.stress] = stress
            
        elif self.grad_on_positions:
            energy = data[properties.energy]
            grad_outputs : List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
            if self.compute_forces:
                grad_inputs = [data[properties.positions]]
                if self.compute_stress:
                    grad_inputs.append(data[properties.strain])
                grads = torch.autograd.grad(
                    [energy,],
                    grad_inputs,
                    grad_outputs=grad_outputs,
                    retain_graph=training,
                    create_graph=training,
                )
                dEdR = grads[0]
                if dEdR is None:
                    dEdR = torch.zeros_like(data[properties.positions])
                data[properties.forces] = -dEdR
                    
                if self.compute_stress:
                    if properties.cell in data:
                        stress = grads[1]
                        if stress is None:
                            stress = torch.zeros_like(data[properties.cell])
                        cell = data[properties.cell].view(-1, 3, 3)
                        volumes = torch.sum(cell[:, 0] * cell[:, 1].cross(cell[:, 2], dim=-1), dim=1)
                        data[properties.stress] = stress / volumes[:, None, None]
        
        else:
            raise ValueError("Gradients must be calculated with respect to positions or R_ij. Nothing is given!")
                    
        return data