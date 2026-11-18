import torch
from curator.data import properties
from curator.data.properties import HeadConfig
from curator.layer import GlobalRescaleShift
from curator.layer._physical_energy_rescale import PhysicalEnergyRescale


def test_physical_scale_keeps_train_eval_energy_and_derivatives_identical():
    source = GlobalRescaleShift(heads=[HeadConfig(
        key=properties.energy, is_atomwise=True, atomwise_key=properties.atomic_energy,
        reduction='sum', scale_by=2.0, shift_by=0.0,
        per_species_shift={1: -3.0}, atomwise_normalization=True)])
    physical = PhysicalEnergyRescale(source)
    assert not any(hasattr(module, 'unscale') for module in physical.modules())
    for training in [True, False]:
        physical.train(training)
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        data = {properties.atomic_energy: x.square(), properties.energy: x.square().sum().reshape(1),
                properties.n_atoms: torch.tensor([2]), properties.image_idx: torch.zeros(2,dtype=torch.long),
                properties.Z: torch.ones(2,dtype=torch.long)}
        energy = physical(data)[properties.energy]
        assert torch.allclose(energy, torch.tensor([4.0]))
        gradient, = torch.autograd.grad(energy.sum(),x,create_graph=True)
        assert torch.allclose(gradient, torch.tensor([4.0,8.0]))
        curvature, = torch.autograd.grad(gradient.sum(),x)
        assert torch.allclose(curvature, torch.tensor([4.0,4.0]))
