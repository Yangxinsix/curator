"""Local chemical modulation of the nine existing STF invariants."""
from torch import nn
from torch.nn import functional as F


class ChemicalGate(nn.Module):
    def __init__(self, ref, scalar_indices):
        super().__init__()
        self.register_buffer("scalar_indices", scalar_indices.clone(), persistent=False)
        self.weight = nn.Parameter(ref.new_zeros((*ref.shape, scalar_indices.numel())))

    def forward(self, normalized_features):
        local = normalized_features.index_select(-1, self.scalar_indices)
        return F.linear(local, self.weight.flatten(0, 1)).unflatten(-1, self.weight.shape[:2])


def attach(model):
    """Install after restoring the common STF model. Return only the new Adam parameter."""
    energy = model.output_modules[0].internal_energy
    branch = energy.tensor_branch
    if (getattr(branch, "kind", None) != "stf_frame" or not branch.exclude_self
            or hasattr(branch, "chemical_gate")):
        raise ValueError("Install the chemical gate once, on a self-excluded STF readout")
    branch.chemical_gate = ChemicalGate(branch.stf_weights, energy.feature_norm.scalar_indices)
    return [branch.chemical_gate.weight]


install_gate = attach
