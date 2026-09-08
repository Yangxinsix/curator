import pytest
import torch

from curator.train import VariationalJointLoss, VariationalModelOutput
from curator.train.train import run_step, tree_to_device


def _batch():
    return {
        "weight": torch.tensor([1.0, 0.5]),
        "variational": {
            "condition_ptr": torch.tensor([0, 2, 3]),
            "condition_weight": torch.tensor([1.0, 2.0, 1.0]),
            "condition_geometry_index": torch.tensor([0, 0, 1]),
            "boundary_hash": ("hash-0", "hash-1", "hash-2"),
            "boundary_spec": (
                {"compile_status": "compiled"},
                {"compile_status": "compiled"},
                {"compile_status": "compiled"},
            ),
            "labels": {
                "energy": {
                    "values": torch.tensor([1.0, 2.0, 3.0]),
                    "ptr": torch.tensor([0, 1, 2, 3]),
                    "shape": ((1,), (1,), (1,)),
                    "present": torch.tensor([True, True, True]),
                    "unit": "eV",
                    "metadata": (
                        {"energy_kind": "fixed-Q total energy"},
                        {"energy_kind": "fixed-Q total energy"},
                        {"energy_kind": "fixed-Q total energy"},
                    ),
                },
                "group_charge": {
                    "values": torch.tensor([0.0, 1.0, -1.0]),
                    "ptr": torch.tensor([0, 1, 1, 3]),
                    "shape": ((1,), None, (2,)),
                    "present": torch.tensor([True, False, True]),
                    "unit": "e",
                    "metadata": ({}, {}, {}),
                },
            },
            "condition_response_ptr": torch.tensor([0, 1, 1, 2]),
            "response_values": torch.tensor([2.0, 4.0]),
            "response_value_ptr": torch.tensor([0, 1, 2]),
            "response_shape": torch.tensor([[1, 1], [1, 1]]),
            "response_input_ref": ("potential", "field"),
            "response_output_ref": ("group_charge", "dipole"),
            "response_unit": ("e/V", "eA/V"),
            "response_gauge": ({"reference": "SHE"}, {"origin": "cell_center"}),
        },
    }


def _prediction():
    energy = torch.tensor([1.0, 4.0, 3.0], requires_grad=True)
    group_charge = torch.tensor([1.0, 100.0, 100.0, 2.0, -2.0], requires_grad=True)
    residual = torch.ones(6, requires_grad=True)
    response = torch.tensor([5.0, 4.0], requires_grad=True)
    curvature = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    return {
        "variational_prediction": {
            "value_space": "physical",
            "labels": {
                "energy": {
                    "values": energy,
                    "ptr": torch.tensor([0, 1, 2, 3]),
                    "shape": ((1,), (1,), (1,)),
                    "unit": "eV",
                },
                "group_charge": {
                    "values": group_charge,
                    "ptr": torch.tensor([0, 1, 3, 5]),
                    "shape": ((1,), (2,), (2,)),
                    "unit": "e",
                },
            },
            "certificates": {
                "kkt": {
                    "values": residual,
                    "ptr": torch.tensor([0, 2, 3, 6]),
                    "eligible": torch.tensor([True, True, True]),
                    "source": "compiled_variational_operator",
                    "normalization": "dimensionless_backward_error",
                }
            },
            # Deliberately reverse the two response blocks.  Matching is semantic.
            "responses": {
                "values": response,
                "ptr": torch.tensor([0, 1, 2]),
                "condition_index": torch.tensor([2, 0]),
                "input_ref": ("field", "potential"),
                "output_ref": ("dipole", "group_charge"),
                "shape": ((1, 1), (1, 1)),
                "unit": ("eA/V", "e/V"),
                "gauge": ({"origin": "cell_center"}, {"reference": "SHE"}),
                "boundary_hash": ("hash-2", "hash-0"),
            },
            "stability": {
                "minimum_curvature": curvature,
                "eligible": torch.tensor([True, True, True]),
                "source": "reduced_lagrangian_hessian",
                "normalization": "metric_normalized_dimensionless_curvature",
            },
        }
    }


def _loss(**kwargs):
    options = dict(
        supervised_weights={"energy": 2.0, "group_charge": 3.0},
        supervised_scales={"energy": 1.0, "group_charge": 1.0},
        required_labels=("energy",),
        required_metadata={"energy": {"energy_kind": "fixed-Q total energy"}},
        certificate_weights={"kkt": 4.0},
        response_weight=5.0,
        response_scales={"e/V": 2.0, "eA/V": 1.0},
        stability_weight=6.0,
    )
    options.update(kwargs)
    return VariationalJointLoss(**options)


def test_joint_loss_is_geometry_balanced_masked_and_differentiable():
    prediction = _prediction()
    result = _loss()(prediction, _batch())

    expected_energy = 2.0 * 16.0 / 9.0
    expected_group = 3.0
    expected_kkt = 4.0
    expected_response = 5.0
    expected_stability = 6.0 * 2.0 / 9.0
    assert float(result["loss"].detach()) == pytest.approx(
        expected_energy + expected_group + expected_kkt + expected_response + expected_stability
    )
    assert result["balance_mode"] == "geometry"
    assert result["terms"]["energy"]["n_conditions"] == 3
    assert result["terms"]["energy"]["n_geometries"] == 2
    assert result["terms"]["group_charge"]["coverage"] == pytest.approx(2 / 3)
    assert float(result["terms"]["energy"]["numerator"].detach()) == pytest.approx(8 / 3)
    assert float(result["terms"]["energy"]["denominator"].detach()) == pytest.approx(1.5)

    result["loss"].backward()
    values = prediction["variational_prediction"]
    assert values["labels"]["energy"]["values"].grad is not None
    assert values["certificates"]["kkt"]["values"].grad is not None
    assert values["responses"]["values"].grad is not None
    assert values["stability"]["minimum_curvature"].grad is not None
    assert values["labels"]["group_charge"]["values"].grad[1:3].abs().sum() == 0


def test_condition_balance_is_an_explicit_different_ensemble_weighting():
    result = _loss(balance_mode="condition")(_prediction(), _batch())
    assert float(result["terms"]["energy"]["unweighted"].detach()) == pytest.approx(16 / 7)
    geometry_loss = _loss()(_prediction(), _batch())["terms"]["energy"]["unweighted"]
    assert float(geometry_loss.detach()) == pytest.approx(16 / 9)


def test_target_copy_cannot_masquerade_as_a_prediction():
    with pytest.raises(KeyError, match="independent variational_prediction"):
        _loss()({"variational": _batch()["variational"]}, _batch())

    prediction = _prediction()
    del prediction["variational_prediction"]["value_space"]
    with pytest.raises(ValueError, match="value_space='physical'"):
        _loss()(prediction, _batch())


def test_optional_missing_label_is_inactive_but_required_coverage_fails():
    optional = VariationalJointLoss(
        supervised_weights={"dipole": 1.0},
        supervised_scales={"dipole": 1.0},
    )(_prediction(), _batch())
    assert optional["terms"]["dipole"]["status"] == "inactive"
    assert optional["terms"]["dipole"]["coverage"] == 0

    with pytest.raises(KeyError, match="required variational label dipole"):
        VariationalJointLoss(
            supervised_weights={"dipole": 1.0},
            supervised_scales={"dipole": 1.0},
            required_labels=("dipole",),
        )(_prediction(), _batch())

    batch = _batch()
    batch["variational"]["labels"]["energy"]["eligible"] = torch.tensor([True, True, True])
    batch["variational"]["labels"]["energy"]["present"][1] = False
    with pytest.raises(ValueError, match="absent for an eligible condition"):
        _loss()(_prediction(), batch)


def test_label_contract_checks_units_shapes_scales_and_semantics():
    with pytest.raises(ValueError, match="require scales"):
        VariationalJointLoss(supervised_weights={"energy": 1.0}, supervised_scales={})

    prediction = _prediction()
    prediction["variational_prediction"]["labels"]["energy"]["unit"] = "hartree"
    with pytest.raises(ValueError, match="units do not match"):
        _loss()(prediction, _batch())

    prediction = _prediction()
    prediction["variational_prediction"]["labels"]["energy"]["shape"] = ((1,), (2,), (1,))
    with pytest.raises(ValueError, match="shape metadata is inconsistent"):
        _loss()(prediction, _batch())

    batch = _batch()
    batch["variational"]["labels"]["energy"]["metadata"] = ({}, {}, {})
    with pytest.raises(ValueError, match="metadata requirement failed"):
        _loss()(_prediction(), batch)


def test_certificates_are_capability_gated_and_cannot_be_arbitrary_heads():
    prediction = _prediction()
    prediction["variational_prediction"]["certificates"]["kkt"]["source"] = "learned_head"
    with pytest.raises(ValueError, match="compiled variational operator"):
        _loss()(prediction, _batch())

    batch = _batch()
    batch["variational"]["boundary_spec"] = (
        {"compile_status": "compiled"},
        {"compile_status": "requires_sign_audit"},
        {"compile_status": "compiled"},
    )
    with pytest.raises(ValueError, match="not eligible"):
        _loss()(_prediction(), batch)

    prediction = _prediction()
    prediction["variational_prediction"]["stability"]["source"] = "full_hessian"
    with pytest.raises(ValueError, match="reduced Lagrangian Hessian"):
        _loss()(prediction, _batch())


def test_response_blocks_match_by_physical_identity_not_list_order():
    assert _loss()(_prediction(), _batch())["terms"]["response"]["status"] == "active"
    prediction = _prediction()
    prediction["variational_prediction"]["responses"]["boundary_hash"] = ("wrong", "hash-0")
    with pytest.raises(KeyError, match="missing response block"):
        _loss()(prediction, _batch())


def test_zero_weight_terms_and_absent_optional_targets_do_not_produce_nan():
    result = VariationalJointLoss(
        supervised_weights={"dipole": 0.0}, supervised_scales={}
    )(_prediction(), _batch())
    assert torch.isfinite(result["loss"])
    assert float(result["loss"].detach()) == 0


def test_inactive_terms_after_a_nonzero_term_do_not_reuse_its_loss_or_gradient():
    prediction = _prediction()
    batch = _batch()
    energy = prediction["variational_prediction"]["labels"]["energy"]["values"]
    # Make the reference tensor unambiguous so inactive terms remain connected
    # to the same prediction graph while contributing exactly zero gradient.
    prediction["variational_prediction"] = {
        "value_space": "physical",
        "labels": {"energy": prediction["variational_prediction"]["labels"]["energy"]},
    }
    baseline = VariationalJointLoss({"energy": 2.0}, {"energy": 1.0})(prediction, batch)
    result = VariationalJointLoss(
        {"energy": 2.0, "missing_optional": 3.0, "disabled": 0.0},
        {"energy": 1.0, "missing_optional": 1.0},
        certificate_weights={"disabled_certificate": 0.0},
    )(prediction, batch)
    assert result["loss"].item() == pytest.approx(baseline["loss"].item())
    assert result["loss"].item() > 0
    for name, status in (
        ("missing_optional", "inactive"),
        ("disabled", "disabled"),
        ("certificate:disabled_certificate", "disabled"),
    ):
        term = result["terms"][name]
        assert term["status"] == status
        assert term["n_conditions"] == term["n_geometries"] == 0
        for key in ("loss", "unweighted", "numerator", "denominator"):
            assert term[key].item() == 0.0
            gradient, = torch.autograd.grad(term[key], energy, retain_graph=True)
            assert torch.equal(gradient, torch.zeros_like(energy))
    actual_gradient, = torch.autograd.grad(result["loss"], energy)
    expected_gradient, = torch.autograd.grad(baseline["loss"], energy)
    assert torch.equal(actual_gradient, expected_gradient)


def test_nested_variational_batch_moves_tensors_and_preserves_metadata():
    batch = _batch()
    moved = tree_to_device(batch, torch.device("cpu"))
    assert moved["variational"]["labels"]["energy"]["values"].device.type == "cpu"
    assert moved["variational"]["response_gauge"] == batch["variational"]["response_gauge"]
    assert moved["variational"]["boundary_hash"] == batch["variational"]["boundary_hash"]


def test_variational_output_runs_inside_the_existing_training_step():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.energy = torch.nn.Parameter(torch.tensor([1.0, 4.0, 3.0]))

        def forward(self, batch):
            return {
                "variational_prediction": {
                    "value_space": "physical",
                    "labels": {
                        "energy": {
                            "values": self.energy,
                            "ptr": torch.tensor([0, 1, 2, 3], device=self.energy.device),
                            "shape": ((1,), (1,), (1,)),
                            "unit": "eV",
                        }
                    },
                }
            }

    model = ToyModel()
    output = VariationalModelOutput(
        VariationalJointLoss(
            supervised_weights={"energy": 2.0},
            supervised_scales={"energy": 1.0},
            required_labels=("energy",),
            required_metadata={"energy": {"energy_kind": "fixed-Q total energy"}},
        )
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.energy.detach().clone()
    losses, metrics = run_step(
        model, _batch(), [output], [], torch.device("cpu"), optimizer, "train"
    )
    assert torch.isfinite(losses["train_total_loss"])
    assert "train_variational_energy_loss" in metrics
    assert not torch.equal(before, model.energy.detach())
    assert output.accumulate_loss() == pytest.approx(2.0 * 16.0 / 9.0)
    output.reset_loss()
    assert output.accumulate_loss() == 0
