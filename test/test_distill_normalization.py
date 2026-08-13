import torch
from torch import nn

from curator.data import properties
from curator.data.properties import HeadConfig
from curator.layer import GlobalRescaleShift
from curator.layer import EnergyHessianOutput, ForceOutput
from curator.model import NeuralNetworkPotential
from curator.model.lit_module import LitNNP
from curator.train.model_output import DistillOutput


def test_distill_output_normalizes_offline_teacher_targets():
    rescale = GlobalRescaleShift(
        heads=[
            HeadConfig(
                key=properties.energy,
                is_atomwise=True,
                reduction="sum",
                atomwise_key=properties.atomic_energy,
                scale_by=2.0,
                shift_by=3.0,
                per_species_shift={"H": 1.5},
            ),
            HeadConfig(key=properties.forces, scale_by=2.0),
        ]
    )
    target = {
        properties.energy: torch.tensor([1.0]),
        properties.Z: torch.tensor([1, 1]),
        properties.n_atoms: torch.tensor([2]),
        properties.image_idx: torch.tensor([0, 0]),
        "teacher_energy": torch.tensor([15.0]),
        "teacher_forces": torch.tensor([[1.0, 2.0, 3.0]]),
        "teacher_energy_hessian": torch.full((3, 3), 8.0),
    }

    energy = DistillOutput("energy_distill", None, "energy", "teacher_energy")
    forces = DistillOutput("forces_distill", None, "forces", "teacher_forces")
    hessian = DistillOutput(
        "hessian_distill",
        None,
        "energy_hessian_sampled",
        "teacher_energy_hessian",
    )
    for output in (energy, forces, hessian):
        output.bind_rescale_layers([rescale])

    torch.testing.assert_close(energy._teacher_target_from_batch(target), torch.tensor([3.0]))
    torch.testing.assert_close(
        forces._teacher_target_from_batch(target),
        target["teacher_forces"] / 2.0,
    )
    torch.testing.assert_close(
        hessian._teacher_target_from_batch(target),
        torch.full((3, 3), 4.0),
    )

    loss, _ = energy.calculate_loss({"energy": torch.tensor([3.0])}, target)
    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_distill_output_uses_force_scale_for_curvature():
    rescale = GlobalRescaleShift(
        heads=[
            HeadConfig(key=properties.energy, scale_by=2.0),
            HeadConfig(key=properties.forces, scale_by=4.0),
        ]
    )
    target = {"teacher_energy_hessian": torch.full((3, 3), 8.0)}
    output = DistillOutput(
        "hessian_distill",
        None,
        properties.energy_hessian_sampled,
        "teacher_energy_hessian",
    )
    output.bind_rescale_layers([rescale])

    torch.testing.assert_close(
        output._teacher_target_from_batch(target),
        torch.full((3, 3), 2.0),
    )


def test_distill_output_uses_identity_when_force_scale_is_missing():
    rescale = GlobalRescaleShift(
        heads=[HeadConfig(key=properties.energy, scale_by=2.0)]
    )
    target = {"teacher_energy_hessian": torch.full((3, 3), 8.0)}
    output = DistillOutput(
        "hessian_distill",
        None,
        properties.energy_hessian,
        "teacher_energy_hessian",
    )
    output.bind_rescale_layers([rescale])

    torch.testing.assert_close(
        output._teacher_target_from_batch(target),
        torch.full((3, 3), 8.0),
    )


def test_lit_nnp_injects_only_model_rescale_layers(monkeypatch):
    rescale = GlobalRescaleShift(
        heads=[
            HeadConfig(key=properties.energy, scale_by=2.0),
            HeadConfig(key=properties.forces, scale_by=4.0),
        ]
    )

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self._initialized = True
            self.output_modules = nn.ModuleList([rescale])

    output = DistillOutput(
        "hessian_distill",
        None,
        properties.energy_hessian_projected,
        "teacher_energy_hessian",
    )
    task = LitNNP(Model(), [output], optimizer=torch.optim.Adam)
    monkeypatch.setattr(task, "_log_runtime_configuration", lambda: None)
    task.setup("fit")

    assert output._student_rescale_layers == [rescale]


class _LinearDirectForces(ForceOutput):
    model_outputs = [properties.forces]

    def forward(self, data):
        data[properties.forces] = 2.0 * data[properties.positions]
        return data


def test_output_pipeline_rescales_force_and_hessian_once():
    rescale = GlobalRescaleShift(
        heads=[HeadConfig(key=properties.forces, scale_by=4.0)]
    )
    hessian = EnergyHessianOutput(
        vectorize=False,
        model_outputs=[properties.energy_hessian],
    )
    model = NeuralNetworkPotential(
        representation=nn.Identity(),
        output_modules=[_LinearDirectForces(), hessian, rescale],
        model_outputs=[properties.forces, properties.energy_hessian],
    )
    assert [type(module) for module in model.output_modules] == [
        _LinearDirectForces,
        EnergyHessianOutput,
        GlobalRescaleShift,
    ]

    batch = {
        properties.positions: torch.randn(2, 3, requires_grad=True),
        properties.n_atoms: torch.tensor([2]),
    }
    model.train()
    normalized = model(batch)
    model.eval()
    physical = model(batch)

    torch.testing.assert_close(
        physical[properties.forces],
        4.0 * normalized[properties.forces],
    )
    torch.testing.assert_close(
        physical[properties.energy_hessian],
        4.0 * normalized[properties.energy_hessian],
    )
    restored = rescale.unscale(physical, force_process=True)
    torch.testing.assert_close(
        restored[properties.energy_hessian],
        normalized[properties.energy_hessian],
    )


def test_force_derivatives_use_force_scale_not_energy_scale():
    rescale = GlobalRescaleShift(
        heads=[
            HeadConfig(key=properties.energy, scale_by=2.0),
            HeadConfig(key=properties.forces, scale_by=4.0),
        ]
    )
    raw = {
        properties.forces: torch.ones(1, 3),
        properties.energy_hessian: torch.ones(1, 3, 1, 3),
    }

    physical = rescale.scale(raw, force_process=True)

    torch.testing.assert_close(physical[properties.forces], 4.0 * raw[properties.forces])
    torch.testing.assert_close(
        physical[properties.energy_hessian],
        4.0 * raw[properties.energy_hessian],
    )
    restored = rescale.unscale(physical, force_process=True)
    torch.testing.assert_close(restored[properties.forces], raw[properties.forces])
    torch.testing.assert_close(
        restored[properties.energy_hessian],
        raw[properties.energy_hessian],
    )
