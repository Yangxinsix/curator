from types import SimpleNamespace

import torch

from curator.data import properties
from curator.data.datamodule import DataContext
from curator.data.properties import HeadConfig
from curator.layer import GlobalRescaleShift, MultiDomainRescaleShift


DERIVATIVE_KEYS = (
    properties.energy_hessian,
    properties.energy_hessian_sampled,
    properties.energy_hessian_projected,
)


def _datamodule(scale: float, energy_shift: float = 0.0):
    context = DataContext(
        head_scale_shift={
            properties.energy: {"mean": energy_shift, "std": scale},
            properties.forces: {"mean": 123.0, "std": scale},
        }
    )
    return SimpleNamespace(
        scale_forces=True,
        rescale_shift_heads=[],
        build_context=lambda heads: context,
    )


def _force_head(scale: float, shift=False):
    return HeadConfig(
        key=properties.forces,
        is_atomwise=True,
        reduction=None,
        scale_by=scale,
        shift_by=shift,
    )


def test_scale_forces_uses_force_rms_for_energy_and_forces_only():
    layer = GlobalRescaleShift(heads=["energy"])
    layer.setup_from_datamodule(_datamodule(scale=2.5, energy_shift=7.0))

    assert [head.key for head in layer.heads] == [properties.energy, properties.forces]
    assert torch.equal(layer.scales[0].scale, torch.tensor([2.5]))
    assert torch.equal(layer.scales[1].scale, torch.tensor([2.5]))
    assert torch.equal(layer.shifts[0].shift, torch.tensor([7.0]))
    assert torch.equal(layer.shifts[1].shift, torch.tensor([0.0]))
    assert layer.heads[1].shift_by is False

    physical = layer.scale(
        {
            properties.atomic_energy: torch.tensor([1.0]),
            properties.image_idx: torch.tensor([0]),
            properties.n_atoms: torch.tensor([1]),
            properties.forces: torch.ones(1, 3),
        },
        force_process=True,
    )
    assert torch.allclose(physical[properties.energy], torch.tensor([9.5]))
    assert torch.allclose(physical[properties.forces], torch.full((1, 3), 2.5))


def test_force_scale_is_shared_by_all_force_derivatives():
    layer = GlobalRescaleShift(heads=[_force_head(scale=4.0)])
    original = {
        properties.forces: torch.arange(6.0).reshape(2, 3),
        properties.energy_hessian: torch.arange(36.0).reshape(6, 6),
        properties.energy_hessian_sampled: torch.arange(12.0).reshape(2, 6),
        properties.energy_hessian_projected: torch.arange(3.0),
    }

    scaled = layer.scale(original, force_process=True)
    assert torch.allclose(scaled[properties.forces], original[properties.forces] * 4.0)
    for key in DERIVATIVE_KEYS:
        assert torch.allclose(scaled[key], original[key] * 4.0)

    restored = layer.unscale(scaled, force_process=True)
    for key, value in original.items():
        assert torch.allclose(restored[key], value)


def test_force_scale_handles_sampled_rows_grouped_by_structure():
    layer = GlobalRescaleShift(heads=[_force_head(scale=4.0)])
    rows = [torch.ones(2, 3, 3), torch.full((1, 2, 3), 2.0)]

    scaled = layer.scale(
        {properties.energy_hessian_sampled: rows},
        force_process=True,
    )
    assert isinstance(scaled[properties.energy_hessian_sampled], list)
    for actual, expected in zip(scaled[properties.energy_hessian_sampled], rows):
        torch.testing.assert_close(actual, 4.0 * expected)

    restored = layer.unscale(scaled, force_process=True)
    for actual, expected in zip(restored[properties.energy_hessian_sampled], rows):
        torch.testing.assert_close(actual, expected)


def test_force_shift_does_not_apply_to_derivatives():
    layer = GlobalRescaleShift(heads=[_force_head(scale=2.0, shift=11.0)])
    original = {
        properties.forces: torch.ones(1, 3),
        properties.energy_hessian: torch.ones(3, 3),
        properties.n_atoms: torch.tensor([1]),
    }

    scaled = layer.scale(original, force_process=True)

    assert torch.allclose(scaled[properties.forces], torch.full((1, 3), 13.0))
    assert torch.allclose(scaled[properties.energy_hessian], torch.full((3, 3), 2.0))


def test_hessian_is_identity_without_a_force_scale():
    energy_head = HeadConfig(
        key=properties.energy,
        is_atomwise=False,
        reduction=None,
        scale_by=5.0,
        shift_by=False,
    )
    layer = GlobalRescaleShift(heads=[energy_head])
    hessian = torch.arange(9.0).reshape(3, 3)

    scaled = layer.scale(
        {properties.energy: torch.tensor([2.0]), properties.energy_hessian: hessian},
        force_process=True,
    )

    assert torch.allclose(scaled[properties.energy], torch.tensor([10.0]))
    assert torch.equal(scaled[properties.energy_hessian], hessian)


def test_trainable_force_scale_stays_synchronized_with_derivatives():
    layer = GlobalRescaleShift(
        heads=[_force_head(scale=2.0)],
        scale_trainable=True,
    )
    force_scale = layer.scales[0].scale
    with torch.no_grad():
        force_scale.fill_(3.5)

    scaled = layer.scale(
        {
            properties.forces: torch.ones(1, 3),
            properties.energy_hessian: torch.ones(3, 3),
        },
        force_process=True,
    )
    assert torch.allclose(scaled[properties.forces], torch.full((1, 3), 3.5))
    assert torch.allclose(scaled[properties.energy_hessian], torch.full((3, 3), 3.5))

    scaled[properties.energy_hessian].sum().backward()
    assert torch.allclose(force_scale.grad, torch.tensor([9.0]))


def test_multi_domain_uses_each_domains_force_scale_for_derivatives():
    first = _datamodule(scale=2.0)
    second = _datamodule(scale=5.0)
    datamodule = SimpleNamespace(
        domain_modules={"first": first, "second": second},
        domain_to_id={"first": 0, "second": 1},
        build_contexts=lambda heads: {},
    )
    layer = MultiDomainRescaleShift(heads=["energy"])
    layer.setup_from_datamodule(datamodule)

    for domain, scale in ((0, 2.0), (1, 5.0)):
        scaled = layer.scale(
            {
                properties.domain: torch.tensor([domain]),
                properties.energy_hessian_projected: torch.ones(4),
            },
            force_process=True,
        )
        assert torch.allclose(
            scaled[properties.energy_hessian_projected],
            torch.full((4,), scale),
        )


def test_global_rescale_with_force_derivatives_is_torchscriptable():
    layer = torch.jit.script(GlobalRescaleShift(heads=[_force_head(scale=3.0)]).eval())
    scaled = layer(
        {
            properties.forces: torch.ones(1, 3),
            properties.energy_hessian: torch.ones(3, 3),
        }
    )

    assert torch.allclose(scaled[properties.forces], torch.full((1, 3), 3.0))
    assert torch.allclose(scaled[properties.energy_hessian], torch.full((3, 3), 3.0))
