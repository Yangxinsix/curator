import torch

from curator.data import properties
from curator.data.properties import HeadConfig
from curator.layer import GlobalRescaleShift
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
            )
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
    torch.testing.assert_close(forces._teacher_target_from_batch(target), target["teacher_forces"])
    torch.testing.assert_close(
        hessian._teacher_target_from_batch(target),
        torch.full((3, 3), 4.0),
    )

    loss, _ = energy.calculate_loss({"energy": torch.tensor([3.0])}, target)
    torch.testing.assert_close(loss, torch.tensor(0.0))
