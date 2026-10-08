"""Loading a head-based checkpoint must retain learned output calibration."""
import copy

import pytest
import torch

from curator.data import properties as p
from curator.data.properties import HeadConfig
from curator.layer import GlobalRescaleShift
from curator.model.checkpoint_upgrade import _upgrade_legacy_rescale_module


@pytest.mark.parametrize('trainable', [False, True])
@pytest.mark.parametrize('atomwise', [False, True])
def test_missing_atomic_scales_preserves_existing_calibration(trainable, atomwise):
    head = HeadConfig(key=p.energy, is_atomwise=True, reduction='sum',
                      atomwise_key=p.atomic_energy, scale_by=0.7735790334431056,
                      shift_by=0.25, per_species_shift={1: -3.0, 8: -5.0})
    module = GlobalRescaleShift([head], scale_trainable=trainable,
                                shift_trainable=trainable).double().eval()
    # Checkpoint buffers, rather than constructor defaults, are authoritative.
    with torch.no_grad():
        module.scales[0].scale.fill_(0.61)
        module.shifts[0].shift.fill_(0.37)
        module.atomic_shifts[0].values[8] = -7.0
    data = {p.energy: torch.tensor([2.0, 4.0], dtype=torch.float64),
            p.n_atoms: torch.tensor([2, 1]), p.image_idx: torch.tensor([0, 0, 1]),
            p.Z: torch.tensor([1, 8, 1])}
    if atomwise:
        data[p.atomic_energy] = torch.tensor([0.8, 1.2, 4.0], dtype=torch.float64)
    expected = module(copy.deepcopy(data))
    scale = module.scales[0].scale
    heads = module.heads
    retained = {k: v.clone() for k, v in module.state_dict().items() if not k.startswith('atomic_scales.')}
    del module.atomic_scales
    _upgrade_legacy_rescale_module(module)
    assert module.heads is heads
    assert module.scales[0].scale is scale
    assert isinstance(scale, torch.nn.Parameter) == trainable
    for key, value in retained.items():
        torch.testing.assert_close(module.state_dict()[key], value, rtol=0, atol=0)
    assert module.atomic_scales[0].values.dtype == torch.float64
    assert not module.atomic_scales[0].enabled
    actual = module(copy.deepcopy(data))
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    once = {k: v.clone() for k,v in module.state_dict().items()}
    _upgrade_legacy_rescale_module(module)
    for key, value in once.items():
        torch.testing.assert_close(module.state_dict()[key], value, rtol=0, atol=0)


def test_explicit_per_species_scales_survive_partial_format():
    module = GlobalRescaleShift([HeadConfig(key=p.energy, is_atomwise=True,
        atomwise_key=p.atomic_energy, per_species_scale={1: 2.0, 8: 3.0})]).eval()
    expected = module.atomic_scales[0].values.clone()
    del module.atomic_scales
    _upgrade_legacy_rescale_module(module)
    assert module.atomic_scales[0].enabled
    torch.testing.assert_close(module.atomic_scales[0].values, expected, rtol=0, atol=0)
