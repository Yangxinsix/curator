"""Actual Curator EFA heads and task: replay/reuse E/F/R and gradient parity.

The backbone is deliberately small, but the two-layer global proposal, convex
energy, boundary compilation, force computation, structured loss and backward
routing are the production implementations. Run inside a GPU allocation.
"""
import copy
import io
import math
from functools import partial

import pytest
import torch
from torch import nn

from curator.data import properties
from curator.layer._variational_output import VariationalEnergyOutput
from curator.layer.efa.core import EuclideanFastAttention
from curator.layer.efa.proposal import EFAChargeProposal
from curator.layer.efa.two_scale import TwoScaleQConditionedEFAEnergy
from curator.model.base import NeuralNetworkPotential, Representation
from curator.model.features import ScalarVectorFeatureSpec
from curator.model.lit_module import LitNNP
from curator.train.variational_losses import VariationalJointLoss, VariationalModelOutput
from curator.train.variational_optimization import VariationalGradientTask

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires allocated GPU")


class SmallRepresentation(Representation):
    """Smooth equivariant features with trainable geometry and baseline energy."""

    cutoff = 6.0

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(3, 4)
        self.radial = nn.Parameter(torch.tensor([.15, -.2, .3, .4]))
        self.vector_weight = nn.Parameter(torch.tensor([.4, -.2]))
        self.readout = nn.Linear(4, 1)

    def response_feature_spec(self):
        return ScalarVectorFeatureSpec("test_scalar", "test_vector", 4, 2)

    def forward(self, data):
        positions = data[properties.positions].requires_grad_(True)
        counts = data[properties.n_atoms]
        centered = torch.cat([part - part.mean(0, keepdim=True)
                              for part in positions.split(counts.tolist())])
        scalar = (self.embedding(data[properties.Z] % 3)
                  + torch.tanh(centered.square().sum(-1, keepdim=True) * self.radial))
        data["test_scalar"] = scalar
        data["test_vector"] = centered[:, :, None] * self.vector_weight
        atomic = self.readout(scalar).flatten()
        data[properties.energy] = torch.stack([part.sum() for part in atomic.split(counts.tolist())])
        return data


def _model():
    torch.manual_seed(20261022)
    head = VariationalEnergyOutput(
        internal_energy=TwoScaleQConditionedEFAEnergy(
            hidden_features=4, charge_features=4, num_directions=50,
            max_frequency=math.pi, energy_scale=.1,
            driving_scale=.2, curvature_scale=.3),
        proposal=EFAChargeProposal(
            hidden_irreps="4x0e + 2x1o", num_layers=2, hidden_features=4,
            readout_features=8, num_directions=50, max_frequency=math.pi),
        state_unit="e", conjugate_unit="V", batch_observables=True,
        residual_gradient_policy="proposal_only", residual_evaluation="replay")
    model = NeuralNetworkPotential(SmallRepresentation(), output_modules=[head])
    for module in model.modules():
        if isinstance(module, EuclideanFastAttention):
            module.contraction = "linear"
    return model.to(device="cuda", dtype=torch.float64).train()


def _batch(seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return {
        properties.positions: torch.randn((9, 3), generator=generator,
                                          device="cuda", dtype=torch.float64),
        properties.Z: torch.tensor([11, 17, 11, 17, 11, 17, 11, 17, 11], device="cuda"),
        properties.n_atoms: torch.tensor([4, 5], device="cuda"),
        properties.image_idx: torch.tensor([0] * 4 + [1] * 5, device="cuda"),
        properties.total_charge: torch.tensor([1., -1.], device="cuda", dtype=torch.float64),
        properties.pbc: torch.zeros((2, 3), device="cuda", dtype=torch.bool),
        properties.edge_idx: torch.empty((0, 2), device="cuda", dtype=torch.long),
        properties.edge_dist: torch.empty(0, device="cuda", dtype=torch.float64),
        properties.edge_diff: torch.empty((0, 3), device="cuda", dtype=torch.float64),
    }


def _targets(batch, prediction):
    # The model uses ordinary total_charge inputs; targets are only added after
    # forward so this test can use its public automatic boundary construction.
    labels = {}
    for name in ("condition_energy", "condition_forces"):
        entry = prediction["variational_prediction"]["labels"][name]
        labels[name] = {
            "values": torch.zeros_like(entry["values"]), "ptr": entry["ptr"],
            "shape": entry["shape"], "unit": entry["unit"],
            "present": torch.ones(2, dtype=torch.bool, device="cuda"), "metadata": ({}, {}),
        }
    return {**batch, "variational": {
        "condition_weight": torch.tensor([1., .7], dtype=torch.float64, device="cuda"),
        "condition_geometry_index": torch.tensor([0, 1], device="cuda"),
        "condition_ptr": torch.tensor([0, 1, 2], device="cuda"),
        "boundary_spec": [{}, {}], "labels": labels,
    }}


def _task(model, weight=2.7, residual_only=False):
    loss = VariationalJointLoss(
        supervised_weights={"condition_energy": 0. if residual_only else 2.,
                            "condition_forces": 0. if residual_only else 3.},
        supervised_scales={"condition_energy": 1., "condition_forces": 1.},
        residual_weights={"stationarity": 7.}, energy_per_atom=True)
    output = VariationalModelOutput(loss, loss_weight=weight)
    task_type = (VariationalGradientTask if model.output_modules[0].residual_evaluation == "reuse"
                 else LitNNP)
    return task_type(model=model, outputs=[output], optimizer=partial(torch.optim.AdamW, lr=1e-4))


def _assert_prediction_equal(left, right):
    left, right = left["variational_prediction"], right["variational_prediction"]
    for name in ("condition_energy", "condition_forces", "variable_value"):
        torch.testing.assert_close(left["labels"][name]["values"], right["labels"][name]["values"],
                                   rtol=3e-9, atol=3e-11, msg=name)
    torch.testing.assert_close(left["residuals"]["stationarity"]["values"],
                               right["residuals"]["stationarity"]["values"],
                               rtol=3e-9, atol=3e-11, msg="stationarity")


def _assert_gradients_equal(left, right):
    expected, actual = dict(left.named_parameters()), dict(right.named_parameters())
    assert expected.keys() == actual.keys()
    for name, parameter in expected.items():
        other = actual[name]
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=3e-8, atol=3e-10, msg=name)


@pytest.mark.parametrize("contraction", ["linear", "auto"])
@pytest.mark.parametrize("output_weight", [1., 2.7])
def test_full_heads_and_task_accumulation_match_replay(contraction, output_weight):
    reference = _model()
    optimized = copy.deepcopy(reference)
    optimized.output_modules[0].residual_evaluation = "reuse"
    for module in optimized.modules():
        if isinstance(module, EuclideanFastAttention):
            module.contraction = contraction
    expected_task, actual_task = _task(reference, output_weight), _task(optimized, output_weight)
    # Two minibatches divided by three intentionally exercise an incomplete
    # accumulation window. Preserve accumulated gradients between backwards.
    for seed in (31, 32):
        batch = _batch(seed)
        expected = reference(copy.deepcopy(batch))
        actual = optimized(copy.deepcopy(batch))
        _assert_prediction_equal(expected, actual)
        target = _targets(batch, expected)
        expected_loss = expected_task.loss_fn(expected, target, "train")[0]["train_total_loss"]
        actual_loss = actual_task.loss_fn(actual, target, "train")[0]["train_total_loss"]
        torch.testing.assert_close(actual_loss, expected_loss, rtol=3e-9, atol=3e-11)
        expected_task.backward(expected_loss / 3)
        actual_task.backward(actual_loss / 3)
        _assert_gradients_equal(reference, optimized)
    proposal_gradients = [p.grad for p in optimized.output_modules[0].proposal.parameters()
                          if p.grad is not None]
    assert any(bool(g.abs().max() > 1e-10) for g in proposal_gradients)


def test_residual_only_reuse_does_not_update_geometry_or_energy():
    reference = _model()
    optimized = copy.deepcopy(reference)
    optimized.output_modules[0].residual_evaluation = "reuse"
    tasks = [_task(reference, residual_only=True), _task(optimized, residual_only=True)]
    for model, task in zip((reference, optimized), tasks):
        batch = _batch(37)
        prediction = model(batch)
        loss = task.loss_fn(prediction, _targets(batch, prediction), "train")[0]["train_total_loss"]
        task.backward(loss / 3)
        assert all(parameter.grad is None for parameter in model.representation.parameters())
        assert all(parameter.grad is None for parameter in model.output_modules[0].internal_energy.parameters())
        assert any(parameter.grad is not None for parameter in model.output_modules[0].proposal.parameters())
    _assert_gradients_equal(reference, optimized)


def test_full_head_reuse_rejects_unrouted_loss_and_eval_needs_no_router():
    model = _model()
    model.output_modules[0].residual_evaluation = "reuse"
    batch = _batch(38)
    task = _task(model)
    prediction = model(copy.deepcopy(batch))
    target = _targets(batch, prediction)
    with pytest.raises(RuntimeError, match="VariationalGradientTask"):
        task.outputs[0].calculate_loss(prediction, target, subset="train")
    model.eval()
    prediction = model(copy.deepcopy(batch))
    loss, _ = task.outputs[0].calculate_loss(prediction, target, subset="val")
    assert bool(torch.isfinite(loss))
    assert not prediction["variational_prediction"]["residuals"]["stationarity"]["requires_parameter_routing"]


def test_whole_model_checkpoint_without_new_attributes_keeps_replay():
    model = _model()
    parameter_keys = set(model.state_dict())
    del model.output_modules[0].residual_evaluation
    for module in model.modules():
        if isinstance(module, EuclideanFastAttention):
            del module.contraction
            del module.pairwise_max_atoms
    stream = io.BytesIO()
    torch.save(model, stream)
    stream.seek(0)
    restored = torch.load(stream, map_location="cuda", weights_only=False)
    assert restored.output_modules[0].residual_evaluation == "replay"
    assert set(restored.state_dict()) == parameter_keys
    assert all(getattr(module, "contraction", "linear") == "linear"
               for module in restored.modules() if isinstance(module, EuclideanFastAttention))
    batch = _batch(40)
    prediction = restored(batch)
    assert bool(torch.isfinite(prediction["variational_prediction"]["labels"]["condition_energy"]["values"]).all())
