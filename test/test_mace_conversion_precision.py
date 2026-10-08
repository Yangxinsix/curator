from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("mace")
from e3nn import o3
from mace.modules import ScaleShiftMACE, blocks

from curator.model.conversion import create_model_from_mace


def _native_model(dtype):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        return ScaleShiftMACE(
            r_max=6.0, num_bessel=4, num_polynomial_cutoff=6, max_ell=1,
            interaction_cls=blocks.RealAgnosticResidualInteractionBlock,
            interaction_cls_first=blocks.RealAgnosticInteractionBlock,
            num_interactions=2, num_elements=2,
            hidden_irreps=o3.Irreps("4x0e + 4x1o"),
            MLP_irreps=o3.Irreps("4x0e"),
            atomic_energies=np.array([-1.1234567890123, -0.1234567890123]),
            avg_num_neighbors=61.964672446250916,
            atomic_numbers=[8, 12], correlation=2,
            gate=torch.nn.functional.silu, pair_repulsion=True,
            use_reduced_cg=False,
            atomic_inter_scale=0.773579123456789, atomic_inter_shift=0.0123456789012345,
        )
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_conversion_preserves_weights_and_radial_values_across_default_dtype(dtype):
    source = _native_model(dtype)
    previous = torch.get_default_dtype()
    opposite = torch.float64 if dtype == torch.float32 else torch.float32
    try:
        torch.set_default_dtype(opposite)
        converted = create_model_from_mace(source)
        assert torch.get_default_dtype() == opposite
    finally:
        torch.set_default_dtype(previous)

    assert next(converted.parameters()).dtype == dtype
    rescale = converted.output_modules[1]
    assert torch.equal(rescale.scales[0].scale.flatten(), source.scale_shift.scale.flatten())
    assert torch.equal(rescale.shifts[0].shift.flatten(), source.scale_shift.shift.flatten())
    assert torch.equal(
        rescale.atomic_shifts[0].values[source.atomic_numbers].flatten(),
        source.atomic_energies_fn.atomic_energies.flatten(),
    )
    source_weights = source.node_embedding.linear.weight
    target_weights = converted.representation.embeddings.chemical_embedding.linear.weight
    assert torch.equal(source_weights, target_weights)
    for index, interaction in enumerate(source.interactions):
        for name, parameter in interaction.named_parameters():
            target = dict(converted.representation.interactions[index].named_parameters())[name]
            assert torch.equal(parameter, target), name
    for index, product in enumerate(source.products):
        for name, buffer in product.named_buffers():
            if "U_matrix" not in name:
                continue
            target = dict(converted.representation.products[index].named_buffers())[name]
            assert torch.equal(buffer, target), name

    basis = converted.representation.embeddings.radial_basis.basis
    assert basis.prefactor == float(source.radial_embedding.bessel_fn.prefactor)
    x = torch.linspace(0.5, 5.8, 16, dtype=dtype)
    assert torch.equal(basis(x), source.radial_embedding.bessel_fn(x[:, None]))


@pytest.mark.parametrize("trainable", [False, True])
def test_conversion_preserves_nondefault_zbl_parameters_and_derivatives(trainable):
    source = _native_model(torch.float64)
    for name, value in (("a_exp", 0.37234567890123), ("a_prefactor", 0.5234567890123)):
        tensor = torch.tensor(value, dtype=torch.float64)
        if trainable:
            delattr(source.pair_repulsion_fn, name)
            source.pair_repulsion_fn.register_parameter(name, torch.nn.Parameter(tensor))
        else:
            setattr(source.pair_repulsion_fn, name, tensor)
    converted = create_model_from_mace(source)
    pair = converted.output_modules[0].pair_fn
    for native_name, target_name in (("a_exp", "screening_exponent"), ("a_prefactor", "screening_length")):
        target = getattr(pair, target_name)
        assert torch.equal(getattr(source.pair_repulsion_fn, native_name), target)
        assert isinstance(target, torch.nn.Parameter) == trainable

    x = torch.tensor([[1.2], [1.2]], dtype=torch.float64, requires_grad=True)
    attrs = torch.eye(2, dtype=torch.float64)
    edges = torch.tensor([[0, 1], [1, 0]])
    previous = torch.get_default_dtype()
    try:
        # Native ZBL multiplies integer atomic numbers by Python floats; those
        # intermediate tensors follow the default dtype rather than x.dtype.
        torch.set_default_dtype(torch.float64)
        native_energy = source.pair_repulsion_fn(x, attrs, edges, source.atomic_numbers)
    finally:
        torch.set_default_dtype(previous)
    curator_energy = pair(x, attrs, edges, source.atomic_numbers)
    torch.testing.assert_close(curator_energy, native_energy, atol=1e-12, rtol=1e-12)
    native_gradient = torch.autograd.grad(native_energy.sum(), x)[0]
    curator_gradient = torch.autograd.grad(curator_energy.sum(), x)[0]
    torch.testing.assert_close(curator_gradient, native_gradient, atol=1e-12, rtol=1e-12)


def test_old_zbl_without_length_factor_retains_behavior():
    from curator.layer import ZBLBasis

    pair = ZBLBasis()
    x = torch.tensor([[1.2], [1.2]])
    attrs = torch.eye(2)
    edges = torch.tensor([[0, 1], [1, 0]])
    numbers = torch.tensor([8, 12])
    before = pair(x, attrs, edges, numbers)
    del pair.screening_length_factor
    assert torch.equal(before, pair(x, attrs, edges, numbers))


def test_conversion_rejects_incompatible_product_basis_and_restores_dtype():
    source = _native_model(torch.float64)
    contraction = source.products[0].symmetric_contractions.contractions[0]
    contraction.U_matrix_2 = contraction.U_matrix_2[..., :0]
    previous = torch.get_default_dtype()
    with pytest.raises(ValueError, match="official MACE product 0.*U_matrix_2"):
        create_model_from_mace(source)
    assert torch.get_default_dtype() == previous


def test_converted_neighbor_normalization_survives_datamodule_initialization():
    from curator.data.datamodule import AtomsDataModule

    source = _native_model(torch.float64)
    converted = create_model_from_mace(source)
    rescale = converted.output_modules[1]
    dm = AtomsDataModule(batch_size=1, species=["O", "Mg"],
                         avg_num_neighbors=91.0625, default_dtype=torch.float64,
                         rescale_shift_heads=rescale.heads)
    converted.initialize_modules(dm)
    for native, target in zip(source.interactions, converted.representation.interactions):
        assert target._initialized
        assert target.avg_num_neighbors.item() == native.avg_num_neighbors
