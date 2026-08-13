import torch

from curator.data import properties
from curator.data.collate_atoms_data import _collate_tensor_property
from curator.data.properties import HeadConfig
from curator.layer import GlobalRescaleShift
from curator.layer._energy_hessian import project_hessian, sample_hessian_projections
from curator.train.model_output import DistillOutput


def test_project_hessian_matches_matrix_product():
    matrix = torch.arange(36, dtype=torch.float32).reshape(6, 6)
    hessian = matrix.reshape(2, 3, 2, 3)
    probes = torch.randn(4, 2, 3)
    expected = (probes.reshape(4, 6) @ matrix.T).reshape(4, 2, 3)
    torch.testing.assert_close(project_hessian(hessian, probes), expected)


def test_dynamic_teacher_projection_uses_student_probe_and_scale():
    rescale = GlobalRescaleShift(
        heads=[
            HeadConfig(key=properties.energy, scale_by=2.0),
            HeadConfig(key=properties.forces, scale_by=2.0),
        ]
    )
    output = DistillOutput(
        "dynamic_phl",
        None,
        student_property=properties.energy_hessian_projected,
        teacher_property="teacher_energy_hessian",
        teacher_output_property=properties.energy_hessian,
        teacher_projection_probe_key=properties.energy_hessian_probe_vectors,
    )
    output.bind_rescale_layers([rescale])

    teacher_hessian = 4.0 * torch.eye(6).reshape(2, 3, 2, 3)
    probes = torch.randn(3, 2, 3)
    target = {
        "teacher_energy_hessian": teacher_hessian,
        properties.energy_hessian_probe_vectors: probes,
    }
    student = {
        properties.energy_hessian_projected: 2.0 * probes,
        properties.energy_hessian_probe_vectors: probes,
    }
    loss, _ = output.calculate_loss(student, target)
    torch.testing.assert_close(loss, torch.zeros(()))


def test_probe_normalization_is_per_structure_and_changes_each_call():
    positions = torch.randn(5, 3, requires_grad=True)
    forces = -positions
    torch.manual_seed(7)
    probes_a, _ = sample_hessian_projections(
        forces,
        positions,
        num_probes=4,
        n_atoms=torch.tensor([2, 3]),
        normalize_probes=True,
        probe_distribution="rademacher",
    )
    probes_b, _ = sample_hessian_projections(
        forces,
        positions,
        num_probes=4,
        n_atoms=torch.tensor([2, 3]),
        normalize_probes=True,
        probe_distribution="rademacher",
    )
    torch.testing.assert_close(probes_a[:, :2].flatten(1).norm(dim=1), torch.ones(4))
    torch.testing.assert_close(probes_a[:, 2:].flatten(1).norm(dim=1), torch.ones(4))
    assert not torch.equal(probes_a, probes_b)


def test_full_hessians_collate_as_block_diagonal():
    first = torch.eye(6).reshape(2, 3, 2, 3)
    second = 2.0 * torch.eye(3).reshape(1, 3, 1, 3)
    collated = _collate_tensor_property("teacher_energy_hessian", [first, second])
    assert collated.shape == (3, 3, 3, 3)
    torch.testing.assert_close(collated[:2, :, :2, :], first)
    torch.testing.assert_close(collated[2:, :, 2:, :], second)
    torch.testing.assert_close(collated[:2, :, 2:, :], torch.zeros(2, 3, 1, 3))
