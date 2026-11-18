"""Proposal ablation structural contracts; full E/F training is tested separately."""
import copy
import io
import math
from types import SimpleNamespace
import pytest
import torch
from e3nn import o3
from curator.layer._variational_proposal import (
    BoundaryConditionedProposal, EquivariantFeatureNorm, EquivariantUpdate,
    AttentionQueryKeyNorm, TensorProductBranchGain, _invariants,
)

VARIANTS = [{}, {"product_gain": 0.25},
            {"product_normalization": "smooth_per_irrep", "product_norm_scale": 1.0},
            {"attention_normalization": "smooth"}]
IRREPS = o3.Irreps("4x0e + 4x1o + 4x2e")


def proposal(options):
    torch.manual_seed(3407)
    result = BoundaryConditionedProposal(communication="attention", hidden_features=4,
        num_heads=2, num_layers=2, state_scale=.05, use_physics_features=False, **options)
    result.bind(SimpleNamespace(irreps=IRREPS), components=1, state_kind="scalar", cutoff=6.)
    return result.double()


def context(grad=False):
    g = torch.Generator().manual_seed(97)
    return {"features": (.2 * torch.randn(3, 36, generator=g, dtype=torch.float64)).requires_grad_(grad),
        "origin": torch.tensor([[.2], [.3], [.5]], dtype=torch.float64, requires_grad=grad),
        "network_field": torch.zeros(3, 1, dtype=torch.float64),
        "positions": torch.tensor([[0., 0., 0.], [1.1, .2, .1], [-.2, 1.3, .4]], dtype=torch.float64, requires_grad=grad),
        "pbc": torch.zeros(3, dtype=torch.bool), "cell": torch.eye(3, dtype=torch.float64),
        "cutoff": torch.tensor(6., dtype=torch.float64),
        "constraint_matrix": torch.ones(1, 3, dtype=torch.float64),
        "environment_matrix": torch.zeros(0, 3, dtype=torch.float64),
        "environment_stiffness": torch.zeros(0, 0, dtype=torch.float64)}


def legacy_update(model, values):
    scale = torch.rsqrt(1 + _invariants(values, model.irreps).square().mean(-1, keepdim=True))
    values = model.mix(values + model.product(values * scale, values * scale))
    gates = torch.sigmoid(model.gates(_invariants(values, model.irreps)))
    blocks, offset = [], 0
    for (channels, irrep), part in zip(model.irreps, model.irreps.slices()):
        block = values[:, part].reshape(values.shape[0], channels, irrep.dim)
        block = (torch.nn.functional.silu(block) if irrep == o3.Irrep("0e")
                 else block * gates[:, offset:offset + channels, None])
        blocks.append(block.flatten(1))
        offset += channels
    return torch.cat(blocks, -1)


def test_norm_groups_repeated_types_and_preserves_o3():
    ir = o3.Irreps("2x0e + 1x1o + 1x0e + 2x1o + 1x2e")
    torch.manual_seed(11)
    x = torch.randn(3, ir.dim, dtype=torch.float64)
    norm = EquivariantFeatureNorm(ir, scale=1.3).double()
    expected = torch.empty_like(x)
    for kind in {kind for _, kind in ir}:
        parts = [part for (_, k), part in zip(ir, ir.slices()) if k == kind]
        joined = torch.cat([x[:, p] for p in parts], -1)
        scale = torch.rsqrt(1 + joined.square().mean(-1, keepdim=True) / 1.3**2)
        for part in parts:
            expected[:, part] = x[:, part] * scale
    torch.testing.assert_close(norm(x), expected, rtol=1e-13, atol=1e-13)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
    for matrix in (rotation, -rotation):
        d = ir.D_from_matrix(matrix)
        torch.testing.assert_close(norm(x @ d.T), norm(x) @ d.T, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(norm(torch.zeros_like(x)), torch.zeros_like(x))


def test_normalizers_have_correct_first_and_second_derivatives():
    torch.manual_seed(21)
    norm = EquivariantFeatureNorm(o3.Irreps("1x0e + 1x1o"), scale=1.).double()
    qk = AttentionQueryKeyNorm().double()
    for amplitude in (0., .7):
        x = (amplitude * torch.randn(1, 4, dtype=torch.float64)).requires_grad_(True)
        q = (amplitude * torch.randn(1, 2, 3, dtype=torch.float64)).requires_grad_(True)
        for module, value in ((norm, x), (qk, q)):
            assert torch.autograd.gradcheck(module, (value,))
            assert torch.autograd.gradgradcheck(module, (value,))


def test_qk_formula_and_content_score_bound():
    torch.manual_seed(22)
    norm = AttentionQueryKeyNorm().double()
    q = torch.randn(3, 2, 16, dtype=torch.float64) * 100
    k = torch.randn_like(q) * 100
    torch.testing.assert_close(norm(q), q * torch.rsqrt(1 + q.square().mean(-1, keepdim=True)))
    score = torch.einsum("ihd,jhd->ijh", norm(q), norm(k)) / math.sqrt(16)
    assert score.abs().max().item() <= 4. + 1e-12


def test_gain_adds_no_parameters_or_persistent_state():
    gain = TensorProductBranchGain(gain=.25)
    x = torch.tensor([1., -2., 0.], requires_grad=True)
    torch.testing.assert_close(gain(x), x * .25)
    assert not list(gain.parameters()) and not gain.state_dict()


def test_variants_preserve_initial_parameters_rng_and_strict_loading():
    base = proposal(VARIANTS[0])
    state = copy.deepcopy(base.state_dict())
    rng = torch.get_rng_state().clone()
    params = dict(base.named_parameters())
    for options in VARIANTS[1:]:
        model = proposal(options)
        assert torch.equal(torch.get_rng_state(), rng)
        assert list(dict(model.named_parameters())) == list(params)
        for name, value in model.named_parameters():
            assert torch.equal(value, params[name]), name
        model.load_state_dict(state, strict=True)
        assert model.checkpoint_contract() != base.checkpoint_contract()


def test_default_matches_old_formula_and_unpickled_object_without_new_attributes():
    torch.manual_seed(23)
    model = EquivariantUpdate(o3.Irreps("4x0e + 4x1o")).double()
    x = torch.randn(3, 16, dtype=torch.float64, requires_grad=True)
    expected = legacy_update(model, x)
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    for name in list(model._modules):
        if name not in {"product", "mix", "gates"}:
            delattr(model, name)
    for name in ("product_gain", "product_normalization", "product_norm_scale", "normalization", "norm_scale"):
        if name in model.__dict__:
            delattr(model, name)
    stream = io.BytesIO()
    torch.save(model, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("options", VARIANTS)
def test_proposal_symmetries_and_projected_charge(options):
    model, ctx = proposal(options), context()
    out = model(ctx)
    r = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
    changed = dict(ctx, features=ctx["features"] @ IRREPS.D_from_matrix(r).T,
                   positions=ctx["positions"] @ r.T + 2.7)
    torch.testing.assert_close(model(changed), out, rtol=1e-6, atol=1e-8)
    index = torch.tensor([2, 0, 1])
    changed = dict(ctx)
    for key in ("features", "origin", "network_field", "positions"):
        changed[key] = ctx[key][index]
    torch.testing.assert_close(model(changed), out[index], rtol=1e-11, atol=1e-11)
    q = ctx["origin"] + out - out.mean(0, keepdim=True)
    torch.testing.assert_close(q.sum(), ctx["origin"].sum(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("options", VARIANTS)
def test_mixed_third_derivatives_support_parameter_backward(options):
    model, ctx = proposal(options), context(grad=True)
    out = model(ctx)
    variables = (ctx["features"], ctx["origin"], ctx["positions"])
    first = torch.autograd.grad(out.square().sum(), variables, create_graph=True)
    second = torch.autograd.grad(sum(x.square().sum() for x in first), variables, create_graph=True)
    third = torch.autograd.grad(sum(x.square().sum() for x in second), variables, create_graph=True)
    assert all(torch.isfinite(x).all() for x in first + second + third)
    objective = out.square().sum() + sum(x.square().sum() for x in first + second + third)
    objective.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(x).all() for x in grads)


def test_default_fp32_coordinate_and_parameter_gradients_match_original_graph():
    # Reusing a scaled intermediate twice can change FP32 gradient accumulation
    # relative to the legacy two multiply nodes despite identical forward values.
    torch.manual_seed(41)
    model = EquivariantUpdate(o3.Irreps("4x0e + 4x1o")).float()
    values = torch.randn(3, 16, dtype=torch.float32, requires_grad=True)
    weight = torch.linspace(-1.7, 2.3, values.numel(), dtype=values.dtype).reshape_as(values)
    actual = model(values)
    expected = legacy_update(model, values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    named = list(model.named_parameters())
    targets = (values,) + tuple(parameter for _, parameter in named)
    actual_grad = torch.autograd.grad((actual * weight).sum(), targets, retain_graph=True)
    expected_grad = torch.autograd.grad((expected * weight).sum(), targets)
    for name, left, right in zip(["input"] + [name for name, _ in named], actual_grad, expected_grad):
        torch.testing.assert_close(left, right, rtol=0, atol=0, msg=lambda message: name + ": " + message)
