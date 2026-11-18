"""Physical-unit energy rescaling independent of train/eval mode."""
import copy
import torch
from torch import nn
from curator.data import properties
from curator.utils import scatter_add

class PhysicalEnergyRescale(nn.Module):
    """Preserve a single pretrained energy head in physical units for ACVR.

    Unlike GlobalRescaleShift this has no unscale protocol. Structured ACVR
    energy/force targets and baseline values already share physical units.
    """
    def __init__(self, rescale):
        super().__init__()
        if len(rescale.heads) != 1:
            raise ValueError('PhysicalEnergyRescale requires one energy head')
        head = rescale.heads[0]
        if head.key != properties.energy or head.reduction != 'sum':
            raise ValueError('Expected one summed energy head')
        self.atomwise_key = head.atomwise_key
        self.transforms = nn.ModuleList(copy.deepcopy(list(rescale.scales)
            + list(rescale.atomic_scales) + list(rescale.atomic_shifts)
            + list(rescale.shifts)))

    def forward(self, data):
        data = data.copy()
        for transform in self.transforms:
            data = transform(data)
        if self.atomwise_key in data:
            data[properties.energy] = scatter_add(
                data[self.atomwise_key], data[properties.image_idx], dim=0)
        return data
