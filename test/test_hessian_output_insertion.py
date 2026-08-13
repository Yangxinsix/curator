import torch
from torch import nn
import pytest

from curator.data import properties
from curator.layer import EnergyHessianOutput, ForceOutput, PairwiseDistance
from curator.train.distill import add_hessian_output
from curator.train.model_output import DistillOutput


class _AtomicForces(ForceOutput):
    def forward(self, data):
        return data


class _Model(nn.Module):
    def __init__(self, output_modules):
        super().__init__()
        self.input_modules = nn.ModuleList(
            [
                PairwiseDistance(
                    compute_distance_from_R=False,
                    compute_forces=False,
                )
            ]
        )
        self.output_modules = nn.ModuleList(output_modules)


def test_add_hessian_output_inserts_after_forces_without_reordering():
    before = nn.Identity()
    forces = _AtomicForces()
    after = nn.ReLU()
    rescale = nn.Sigmoid()
    model = _Model([before, forces, after, rescale])
    output_modules = model.output_modules
    hessian = EnergyHessianOutput(vectorize=False)

    returned = add_hessian_output(model, hessian)

    assert returned is model
    assert model.output_modules is output_modules
    assert list(model.output_modules) == [before, forces, hessian, after, rescale]
    assert model.input_modules[0].compute_distance_from_R
    assert model.input_modules[0].compute_forces

    with pytest.raises(ValueError, match="already has"):
        add_hessian_output(model, EnergyHessianOutput())


class _RecordingRescale(nn.Module):
    def __init__(self, divisor):
        super().__init__()
        self.divisor = divisor
        self.calls = 0

    def unscale(self, data, force_process=False):
        assert force_process
        self.calls += 1
        result = data.copy()
        for key in (
            properties.energy_hessian,
            properties.energy_hessian_sampled,
            properties.energy_hessian_projected,
        ):
            if key in result:
                result[key] = result[key] / self.divisor
        return result


def test_distill_uses_same_student_rescale_path_online_and_offline(monkeypatch):
    student_property = properties.energy_hessian_projected
    physical_teacher_value = torch.full((2, 3), 8.0)
    expected = torch.full((2, 3), 2.0)
    rescale = _RecordingRescale(divisor=4.0)

    offline = DistillOutput(
        "offline_hessian",
        None,
        student_property=student_property,
        teacher_property="teacher_hessian",
    )
    offline.bind_rescale_layers([rescale])
    torch.testing.assert_close(
        offline._teacher_target_from_batch(
            {"teacher_hessian": physical_teacher_value}
        ),
        expected,
    )

    online = DistillOutput(
        "online_hessian",
        "unused-teacher-path",
        student_property=student_property,
        teacher_output_property="teacher_hessian",
    )
    online.bind_rescale_layers([rescale])
    monkeypatch.setattr(
        online,
        "_teacher_prediction",
        lambda pred, target: {"teacher_hessian": physical_teacher_value},
    )
    _, normalized_target = online._resolve_inputs(
        {student_property: torch.zeros_like(physical_teacher_value)},
        {},
    )

    torch.testing.assert_close(normalized_target[student_property], expected)
    assert rescale.calls == 2
