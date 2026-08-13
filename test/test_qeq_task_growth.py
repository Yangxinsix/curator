import copy

import torch

from curator.data import properties
from curator.data.properties import HeadConfig
from curator.layer import (
    ChargeEquilibration,
    Dense,
    GlobalRescaleShift,
    GradientOutput,
    MACEAtomwiseNN,
    PairwiseDistance,
)
from curator.model import MACE, NeuralNetworkPotential
from curator.model.conversion import transform_mace_to_qeq


def _source_model():
    energy_head = HeadConfig(
        key=properties.energy,
        is_atomwise=True,
        reduction="sum",
        atomwise_key=properties.atomic_energy,
    )
    representation = MACE(
        cutoff=5.0,
        num_interactions=2,
        correlation=2,
        species=["Na", "Cl"],
        lmax=1,
        num_features=4,
        num_basis=4,
        avg_num_neighbors=2.0,
        heads=[energy_head],
    )
    model = NeuralNetworkPotential(
        representation=representation,
        input_modules=[PairwiseDistance()],
        output_modules=[
            GlobalRescaleShift(
                heads=[
                    HeadConfig(
                        key=properties.energy,
                        scale_by=2.5,
                        shift_by=0.0,
                        per_species_shift={11: -1.0, 17: -2.0},
                    )
                ]
            ),
            GradientOutput(),
        ],
        model_outputs=[properties.energy, properties.forces],
    )
    model._initialized = True
    return model


def test_mace_qeq_growth_preserves_energy_readout_and_zero_initializes_charge():
    torch.manual_seed(7)
    source = _source_model()
    source_state = copy.deepcopy(source.representation.readout.state_dict())
    grown = transform_mace_to_qeq(source, cutoff=13.2)

    assert isinstance(grown.representation.readout, MACEAtomwiseNN)
    assert grown.representation.readout.separate_heads
    assert [head.key for head in grown.representation.readout.heads] == [
        properties.energy,
        properties.atomic_charge,
    ]
    for index, source_readout in enumerate(source.representation.readout.readouts):
        target = (
            grown.representation.readout.readouts_by_head[properties.energy][index]
            if index == 0
            else grown.representation.readout.final_readouts[properties.energy]
        )
        for key, value in source_readout.state_dict().items():
            torch.testing.assert_close(target.state_dict()[key], value)

    features = torch.randn(3, sum(grown.representation.readout.in_features_list))
    output = grown.representation.readout._compute(features)
    torch.testing.assert_close(output[:, 1], torch.zeros_like(output[:, 1]))

    output[:, 1].sub(1.0).square().mean().backward()
    final = grown.representation.readout.final_readouts[properties.atomic_charge]
    last_dense = [module for module in final.modules() if isinstance(module, Dense)][-1]
    assert last_dense.linear.weight.grad is not None
    assert torch.count_nonzero(last_dense.linear.weight.grad) > 0
    for key, value in source_state.items():
        torch.testing.assert_close(source.representation.readout.state_dict()[key], value)


def test_mace_qeq_growth_sets_physical_module_order_and_scales():
    grown = transform_mace_to_qeq(_source_model(), cutoff=13.2)

    assert isinstance(grown.input_modules[0], PairwiseDistance)
    assert grown.input_modules[0].compute_distance_from_R
    assert [type(module) for module in grown.output_modules] == [
        GradientOutput,
        ChargeEquilibration,
        GlobalRescaleShift,
    ]
    rescale = grown.output_modules[-1]
    torch.testing.assert_close(
        rescale.get_scale(properties.energy),
        torch.tensor([2.5]),
    )
    torch.testing.assert_close(
        rescale.get_scale(properties.forces),
        torch.tensor([2.5]),
    )
    assert rescale._physical_contribution_sources == [
        properties.qeq_energy,
        properties.ewald_forces,
    ]
    assert grown._initialized
