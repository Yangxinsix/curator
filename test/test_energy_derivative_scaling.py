"""Energy gradients must share calibration in inference and Curator's loss path."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from curator.data import properties as p
from curator.data.datamodule import DataContext
from curator.data.properties import HeadConfig
from curator.layer import GlobalRescaleShift, GradientOutput
from curator.layer._rescale import MultiDomainRescaleShift
from curator.model.base import NeuralNetworkPotential
from curator.model.lit_module import LitNNP
from curator.train.model_output import ModelOutput
from curator.utils import load_model


class HarmonicEnergy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.3, dtype=torch.float64))

    def forward(self, data):
        positions = data[p.positions]
        positions = positions + positions @ data[p.strain][0]
        data[p.energy] = (self.weight * positions.square().sum() / 2).reshape(1)
        return data


def batch():
    return {
        p.positions: torch.tensor([[0.2, 0.4, -0.1], [0.8, -0.3, 0.5]],
                                  dtype=torch.float64, requires_grad=True),
        p.strain: torch.zeros(1, 3, 3, dtype=torch.float64, requires_grad=True),
        p.cell: 3 * torch.eye(3, dtype=torch.float64).unsqueeze(0),
        p.n_atoms: torch.tensor([2]),
        p.image_idx: torch.tensor([0, 0]),
        p.Z: torch.tensor([1, 8]),
    }


def rescale(explicit_force_head=False, scale=2.5, trainable=False):
    heads = [HeadConfig(key=p.energy, scale_by=scale, shift_by=0.7,
                        atomwise_normalization=False)]
    if explicit_force_head:
        # A derived force must not acquire an independent scale or any shift.
        heads.append(HeadConfig(key=p.forces, scale_by=11.0, shift_by=7.0,
                                atomwise_normalization=False))
    return GlobalRescaleShift(heads, scale_trainable=trainable).double()


def potential(layer, scale_first):
    gradient = GradientOutput(grad_on_edge_diff=False, grad_on_positions=True,
                              model_outputs=[p.energy, p.forces, p.stress, p.virial])
    modules = [layer, gradient] if scale_first else [gradient, layer]
    return NeuralNetworkPotential(HarmonicEnergy(), output_modules=modules).double()


@pytest.mark.parametrize("scale_first", [False, True])
@pytest.mark.parametrize("explicit_force_head", [False, True])
def test_energy_force_stress_and_training_loss_agree(scale_first, explicit_force_head):
    layer = rescale(explicit_force_head)
    model = potential(layer, scale_first)
    data = batch()
    physical = model.eval()(data)
    energy_gradient = torch.autograd.grad(physical[p.energy].sum(), data[p.positions],
                                          retain_graph=True)[0]
    torch.testing.assert_close(physical[p.forces], -energy_gradient)
    torch.testing.assert_close(physical[p.forces], -2.5 * model.representation.weight * data[p.positions])
    strain_gradient = torch.autograd.grad(physical[p.energy].sum(), data[p.strain])[0]
    voigt = strain_gradient.reshape(-1, 9)[:, [0, 4, 8, 5, 2, 1]]
    torch.testing.assert_close(physical[p.stress], voigt / 27)
    torch.testing.assert_close(physical[p.virial], -voigt)

    raw = model.train()(batch())
    scaled = layer.scale(raw, force_process=True)
    unscaled = layer.unscale(physical, force_process=True)
    for key in physical:
        torch.testing.assert_close(scaled[key], physical[key])
        torch.testing.assert_close(unscaled[key], raw[key])

    # Exercise the existing Curator loss with both normalized train and val inputs.
    target = {key: value.detach().clone() for key, value in physical.items()}
    target[p.forces] += 0.1
    task = LitNNP(model, [ModelOutput(p.forces, nn.MSELoss())], torch.optim.Adam)
    normalized_target = layer.unscale(target, force_process=True)
    train_loss, _ = task.loss_fn(raw, normalized_target, "train")
    val_loss, _ = task.loss_fn(unscaled, normalized_target, "val")
    torch.testing.assert_close(train_loss["train_total_loss"], torch.tensor(0.04 ** 2, dtype=torch.float64))
    torch.testing.assert_close(train_loss["train_total_loss"], val_loss["val_total_loss"])


def test_live_scale_is_shared_and_no_parameters_are_added():
    layer = rescale(trainable=True)
    original_keys = set(layer.state_dict())
    model = potential(layer, scale_first=False).eval()
    with torch.no_grad():
        layer.scales[0].scale.fill_(3.0)
    data = batch()
    force = model(data)[p.forces]
    torch.testing.assert_close(force, -3.0 * model.representation.weight * data[p.positions])
    grad = torch.autograd.grad(force.sum(), layer.scales[0].scale)[0]
    torch.testing.assert_close(grad, (-model.representation.weight * data[p.positions].sum()).reshape(1))
    assert set(layer.state_dict()) == original_keys


def test_independent_force_head_keeps_its_own_scaling():
    layer = rescale(explicit_force_head=True)
    NeuralNetworkPotential(nn.Identity(), output_modules=[layer], model_outputs=[p.forces])
    data = {p.forces: torch.ones(2, 3, dtype=torch.float64)}
    scaled = layer.eval()(data)
    torch.testing.assert_close(scaled[p.forces], data[p.forces] * 11 + 7)
    torch.testing.assert_close(layer.unscale(scaled, force_process=True)[p.forces], data[p.forces])


def test_separate_force_rescale_cannot_rescale_energy_gradients_twice():
    energy_scale = rescale()
    model = potential(energy_scale, scale_first=True)
    model.output_modules.append(GlobalRescaleShift([
        HeadConfig(key=p.forces, scale_by=11.0, shift_by=7.0, atomwise_normalization=False)
    ]).double())
    data = batch()
    physical = model.eval()(data)
    gradient = torch.autograd.grad(physical[p.energy].sum(), data[p.positions])[0]
    torch.testing.assert_close(physical[p.forces], -gradient)


@pytest.mark.parametrize("scale_first", [False, True])
@pytest.mark.parametrize("domain,scale", [(0, 2.5), (1, 4.0)])
def test_domain_specific_energy_scale_is_inherited(scale_first, domain, scale):
    layer = MultiDomainRescaleShift(["energy"])
    layer.domain_modules = nn.ModuleDict({"0": rescale(), "1": rescale(scale=4.0)})
    model = potential(layer, scale_first)
    data = batch()
    data[p.domain] = torch.tensor([domain])
    physical = model.eval()(data)
    torch.testing.assert_close(physical[p.forces], -scale * model.representation.weight * data[p.positions])
    physical[p.domain] = data[p.domain]
    raw = layer.unscale(physical, force_process=True)
    torch.testing.assert_close(raw[p.forces], -model.representation.weight * data[p.positions])


def test_datamodule_initialization_keeps_derivative_binding():
    layer = MultiDomainRescaleShift(["energy"])
    model = potential(layer, scale_first=False)
    dm = SimpleNamespace(rescale_shift_heads=[], build_context=lambda heads: DataContext(
        head_scale_shift={p.energy: {"std": 4.0, "mean": 0.0}}))
    model.initialize_modules(dm)
    data = batch()
    physical = model.eval()(data)
    torch.testing.assert_close(physical[p.forces], -4.0 * model.representation.weight * data[p.positions])


@pytest.mark.parametrize("scale_first", [False, True])
def test_old_pickled_model_is_rebound_without_changing_weights(tmp_path, scale_first):
    layer = rescale(trainable=True)
    model = potential(layer, scale_first).eval()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    del layer.energy_derivatives
    del layer._derivatives_before_scale
    path = tmp_path / "old.model"
    torch.save(model, path)
    restored = load_model(path, device="cpu", load_weights_only=False)
    assert set(restored.state_dict()) == set(state)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    data = batch()
    physical = restored(data)
    torch.testing.assert_close(physical[p.forces], -2.5 * restored.representation.weight * data[p.positions])


def test_per_species_energy_scale_cannot_silently_normalize_force_labels():
    layer = rescale()
    layer.atomic_scales[0].load_values({1: 2.0, 8: 3.0})
    potential(layer, scale_first=True)
    with pytest.raises(ValueError, match="per-species energy scales"):
        layer.unscale({p.forces: torch.ones(2, 3)}, force_process=True)


def test_new_gradient_outputs_are_bound_through_existing_callback():
    layer = rescale()
    gradient = GradientOutput(model_outputs=[p.forces])
    NeuralNetworkPotential(HarmonicEnergy(), output_modules=[gradient, layer])
    gradient.update_model_outputs([p.stress, p.edge_forces])
    assert set(layer.energy_derivatives) == {p.forces, p.stress, p.virial, p.edge_forces}


@pytest.mark.parametrize("operation", ["append", "extend", "insert"])
def test_adding_gradient_and_swapping_order_refreshes_shared_scale(operation):
    layer = rescale()
    model = NeuralNetworkPotential(HarmonicEnergy(), output_modules=[layer], model_outputs=[p.energy])
    gradient = GradientOutput(grad_on_edge_diff=False, grad_on_positions=True, model_outputs=[p.forces])
    if operation == "extend":
        model.output_modules.extend([gradient])
    elif operation == "insert":
        model.output_modules.insert(0, gradient)
    else:
        model.output_modules.append(gradient)
    for _ in range(2):
        data = batch()
        physical = model.eval()(data)
        torch.testing.assert_close(physical[p.forces], -2.5 * model.representation.weight * data[p.positions])
        model.output_modules[0], model.output_modules[1] = model.output_modules[1], model.output_modules[0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ase_calculator_inputs_follow_model_precision(dtype):
    from ase import Atoms
    from curator.simulate.core.calculator import MLCalculator

    calculator = MLCalculator(nn.Linear(1, 1).to(dtype=dtype), cutoff=2.0)
    data = calculator.ase_data_reader(Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.7]]))
    assert data[p.positions].dtype == dtype


@pytest.mark.parametrize("prepare_model", [False, True])
def test_rescale_datamodule_setup_preserves_fp64_head_values(prepare_model):
    from omegaconf import OmegaConf
    from curator.commands.train import LoadedModel, _prepare_model
    from curator.data.datamodule import AtomsDataModule
    from curator.layer.wrappers.utils import temporary_default_dtype

    scale, shift = 0.773579123456789, 0.0123456789012345
    offsets = {1: -1.123456789012345, 8: -0.234567890123456}
    species_scales = {1: 1.123456789012345, 8: 1.234567890123456}
    head = HeadConfig(key=p.energy, scale_by=scale, shift_by=shift,
                      per_species_shift=offsets, per_species_scale=species_scales)
    dm = AtomsDataModule(batch_size=1, species=["H", "O"], avg_num_neighbors=1.0,
                         default_dtype=torch.float64, rescale_shift_heads=[head])
    with temporary_default_dtype(torch.float32):
        layer = GlobalRescaleShift(["energy"])
        if prepare_model:
            # A FP32 checkpoint and FP64 data must initialize calibration in FP64.
            model = potential(layer, scale_first=False).float()
            _prepare_model(LoadedModel(model, wrapper_transform_applied=True),
                           config=OmegaConf.create({"compile": False}),
                           datamodule=dm, data_dtype=torch.float64)
        else:
            layer.double()
            layer.setup_from_datamodule(dm)
        assert torch.get_default_dtype() == torch.float32

    assert layer.scales[0].scale.dtype == torch.float64
    assert layer.scales[0].scale.item() == scale
    assert layer.shifts[0].shift.item() == shift
    for z in offsets:
        assert layer.atomic_shifts[0].values[z].item() == offsets[z]
        assert layer.atomic_scales[0].values[z].item() == species_scales[z]
    assert all(buffer.device.type == "cpu" for buffer in layer.buffers())


def test_rescale_construction_follows_default_precision():
    from curator.layer.wrappers.utils import temporary_default_dtype

    value = 1.123456789012345
    with temporary_default_dtype(torch.float64):
        layer = GlobalRescaleShift([HeadConfig(key=p.energy, scale_by=value,
            shift_by=value, atomwise_normalization=False, per_species_shift={8: value})])
    assert layer.scales[0].scale.item() == value
    assert layer.shifts[0].shift.item() == value
    assert layer.atomic_shifts[0].values[8].item() == value
