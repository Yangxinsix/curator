"""Loss reductions shared by matched long-range benchmark comparisons."""
import torch
from torch import nn
from curator.data import properties


class PerStructureForceMSE(nn.Module):
    """Average force-component MSE within each structure, then over structures.

    Ordinary MSE on a concatenated atom batch gives larger structures more
    weight. This reduction instead matches VariationalJointLoss's one-condition
    per-geometry force loss, which gives each geometry equal weight.
    ``ModelOutput`` supplies the batch through the requires_batch protocol.
    """
    requires_batch = True

    def forward(self, prediction, target, batch):
        if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 3:
            raise ValueError('force predictions and targets must both have shape [n_atoms, 3]')
        counts = batch[properties.n_atoms]
        if counts.ndim != 1 or counts.numel() == 0 or counts.dtype not in (torch.int32, torch.int64):
            raise ValueError('n_atoms must be a nonempty rank-one integer tensor')
        sizes = counts.tolist()
        if any(n <= 0 for n in sizes) or sum(sizes) != prediction.shape[0]:
            raise ValueError('positive n_atoms must partition every force row exactly once')
        squared_error = (prediction - target).square()
        return torch.stack([part.mean() for part in squared_error.split(sizes, dim=0)]).mean()
