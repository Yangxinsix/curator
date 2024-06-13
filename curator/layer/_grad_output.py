import torch
from typing import Tuple, Optional, List
from curator.data import properties
from ase.stress import full_3x3_to_voigt_6_stress
        
class GradientOutput(torch.nn.Module):
    def __init__(
        self,
        grad_on_edge_diff: bool = True,
        grad_on_positions: bool = False,
        compute_forces: bool = True,
        compute_stress: bool = False,
    ) -> None:
        super().__init__()
        self.grad_on_edge_diff = grad_on_edge_diff
        self.grad_on_positions = grad_on_positions
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self.model_outputs = []
        self.update_model_outputs()
    
    @property
    def compute_forces(self):
        return self._compute_forces

    @compute_forces.setter
    def compute_forces(self, value):
        self._compute_forces = value
        self.update_model_outputs()

    @property
    def compute_stress(self):
        return self._compute_stress

    @compute_stress.setter
    def compute_stress(self, value):
        self._compute_stress = value
        self.update_model_outputs()

    def update_model_outputs(self):
        if self._compute_forces:
            if "forces" not in self.model_outputs:
                self.model_outputs.append("forces")
        else:
            if "forces" in self.model_outputs:
                self.model_outputs.remove("forces")

        if self._compute_stress:
            if "stress" not in self.model_outputs:
                self.model_outputs.append("stress")
        else:
            if "stress" in self.model_outputs:
                self.model_outputs.remove("stress")

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
                        atomic_stress = torch.einsum("ij, ik -> ijk", edge_diff, dE_ddiff)           # I'm quite not sure if a negative sign should be added before dE_ddiff, but I think it should be right
                        cell = data[properties.cell].view(-1, 3, 3)
                        volumes = torch.sum(cell[:, 0] * cell[:, 1].cross(cell[:, 2]), dim=1)
                        i_stress = torch.zeros(
                            (forces_dim, 3, 3),                                         
                            dtype=forces.dtype,
                            # it seens like the calculation is not very right... because f_ij is not absolutely right here. Maybe we need to do something like in force calculation
                            # add i_stress and j_stress together then it is the total stress. need verification
                            device=forces.device).index_add(0, edge_idx[:, 0], atomic_stress)
                        j_stress = torch.zeros_like(i_stress).index_add(0, edge_idx[:, 1], atomic_stress)
                        atomic_stress = -i_stress + j_stress          
                        stress = torch.zeros_like(cell).index_add(0, image_idx, atomic_stress)
                        stress = stress / volumes[:, None, None] / 2
                        data[properties.stress] = full_3x3_to_voigt_6_stress(stress)
            
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
                        volumes = torch.sum(cell[:, 0] * cell[:, 1].cross(cell[:, 2]), dim=1)
                        data[properties.stress] = stress / volumes[:, None, None]
        
        else:
            raise ValueError("Gradients must be calculated with respect to positions or R_ij. Nothing is given!")
                    
        return data