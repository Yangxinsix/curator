"""Reference force-constant reconstruction and analytic full-proposal phonons."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/acvr_phonon_response.py"
SPEC = importlib.util.spec_from_file_location("acvr_phonon_response", SCRIPT)
phonons = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phonons)
SOURCE = Path("/workspace/data/datasets/acvr/public_response/phonondb/MgO/data/phono3py_params.yaml")


def _split_metadata(path):
    # A minimal saved-metadata fixture with the audited seed-17 reference roles;
    # the actual full 148-row split is separately audited through the data module.
    path.write_text(json.dumps({"dataset_id": "phonondb/MgO", "split_audit": {"seed": 17},
        "source_hashes": {"data/phono3py_params.yaml": hashlib.sha256(SOURCE.read_bytes()).hexdigest()},
        "source_ids": ["single_displacement:1", "single_displacement:2", "phonon_holdout:1", "phonon_holdout:2"],
        "split_indices": {"train": [1], "val": [], "test": [0], "phonon_holdout": [2, 3]}}))
    return path


def test_coordinate_hvp_retains_proposal_derivative_and_does_not_use_dense_autograd(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation must use coordinate HVPs")
    monkeypatch.setattr(torch.autograd.functional, "hessian", forbidden)
    positions = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64, requires_grad=True)
    proposal = positions * 1.0
    # Phi(R,x)=2R^2+(x-R)^2. Full xhat=R gives H=4I; freezing
    # xhat would incorrectly give the clamped curvature 6I.
    energy = 2 * positions.square().sum() + (proposal - positions).square().sum()
    hessian, forces = phonons.coordinate_hessian(energy, positions)
    np.testing.assert_allclose(hessian, 4 * np.eye(3), atol=1e-12)
    np.testing.assert_allclose(forces, -4 * positions.detach().numpy(), atol=1e-12)
    linear = positions.sum()
    zeros, _ = phonons.coordinate_hessian(linear, positions)
    np.testing.assert_array_equal(zeros, np.zeros((3, 3)))


def test_asr_projection_preserves_raw_and_restores_only_translation_nullspace():
    rng = np.random.default_rng(5)
    factor = rng.normal(size=(12, 12))
    matrix = factor.T @ factor
    original = phonons.matrix_fc(matrix)
    symmetric, corrected, diagnostic = phonons.correct_asr(original)
    np.testing.assert_array_equal(phonons.fc_matrix(original), matrix)
    translations = np.tile(np.eye(3), (4, 1))
    np.testing.assert_allclose(phonons.fc_matrix(corrected) @ translations, 0, atol=1e-13)
    assert diagnostic["raw_asr_max_eV_per_A2"] > 0.1
    assert diagnostic["corrected_asr_max_eV_per_A2"] < 1e-13
    projector = np.eye(12) - translations @ translations.T / 4
    np.testing.assert_allclose(phonons.fc_matrix(corrected), projector @ matrix @ projector, atol=1e-13)
    np.testing.assert_allclose(phonons.fc_matrix(symmetric), matrix)


def test_actual_joint_painn_periodic_hvp_is_symmetric_and_translation_invariant(monkeypatch):
    pytest.importorskip("phonopy")
    from phonopy.structure.atoms import PhonopyAtoms
    from curator.layer import PairwiseDistance, VariationalEnergyOutput
    from curator.model import NeuralNetworkPotential, Painn
    import curator.layer._variational_output as output_layer

    def forbidden(*args, **kwargs):
        raise AssertionError("phonon model evaluation must not solve an internal teacher")
    monkeypatch.setattr(output_layer, "solve_functional_reference", forbidden)
    torch.manual_seed(5)
    model = NeuralNetworkPotential(
        Painn(1, 4, 3.1, num_basis=4),
        input_modules=[PairwiseDistance(compute_distance_from_R=True)],
        output_modules=[VariationalEnergyOutput(hidden_features=4)],
    ).double()
    # Two-atom periodic cell includes image edges; the equilibrium of this
    # untrained model is irrelevant to the AD/symmetry correctness check.
    cell = PhonopyAtoms(symbols=["Mg", "O"], cell=np.eye(3) * 4.2,
                        scaled_positions=[[0, 0, 0], [0.5, 0, 0]])
    fc, metadata = phonons.model_force_constants(model, cell, total_charge=0.0)
    matrix = phonons.fc_matrix(fc)
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-10)
    np.testing.assert_allclose(matrix @ np.tile(np.eye(3), (2, 1)), 0, atol=1e-10)
    assert metadata["state_mode"] == "predicted"
    assert metadata["coordinate_hvp_count"] == 6
    assert metadata["total_charge_condition"] == 0.0
    assert np.linalg.norm(matrix) > 1e-8


@pytest.mark.skipif(not SOURCE.is_file(), reason="public MgO DFT displacement file is not installed")
def test_mgo_reference_commensurate_modes_and_reference_nac_artifact(tmp_path):
    pytest.importorskip("phonopy")
    args = argparse.Namespace(params=SOURCE, cell="64", checkpoint=None, device="cpu",
                              cutoff=None, total_charge=None, nac_direction=[1.0, 0.0, 0.0],
                              output=tmp_path / "reference", holdout_reference_512=True,
                              split_metadata=_split_metadata(tmp_path / "split.json"))
    metadata = phonons.evaluate(args)
    assert metadata["status"] == "reference_only"
    assert metadata["single_displacements"] == 2
    assert metadata["n_supercell_atoms"] == 64
    assert metadata["n_primitive_atoms"] == 2
    assert metadata["n_commensurate_qpoints"] == 32
    assert "not model predictions" in metadata["nac"]["born_and_dielectric"]
    arrays = np.load(tmp_path / "reference.npz")
    assert arrays["reference_fc_raw"].shape == (64, 64, 3, 3)
    assert arrays["reference_qpoints_asr_THz"].shape == (32, 6)
    np.testing.assert_allclose(np.sort(arrays["reference_qpoints_asr_THz"].reshape(-1)),
                               arrays["reference_supercell_asr_THz"], atol=1e-5)
    np.testing.assert_allclose(arrays["reference_gamma_asr_THz"][:3], 0, atol=1e-5)
    np.testing.assert_allclose(arrays["reference_gamma_asr_THz"][3:], 11.7495268, atol=1e-5)
    assert arrays["reference_gamma_asr_reference_nac_THz"][-1] > 20
    assert np.max(np.abs(arrays["reference_gamma_raw_THz"][:3])) > 0.001
    assert "model_fc_raw" not in arrays.files
    assert [entry["split"] for entry in metadata["split_provenance"]["reference_force_labels"]] == ["test", "train"]
    assert metadata["split_provenance"]["contains_training_labels"]
    assert arrays["holdout512_fc_raw"].shape == (512, 512, 3, 3)
    assert arrays["holdout512_qpoints_asr_THz"].shape == (32, 6)
    holdout = metadata["holdout_reference_512"]
    assert holdout["additional_model_coordinate_hvp_count"] == 0
    assert not holdout["split_provenance"]["contains_training_labels"]
    assert all(entry["split"] == "phonon_holdout" for entry in holdout["split_provenance"]["reference_force_labels"])
    saved = json.loads((tmp_path / "reference.json").read_text())
    assert saved["source_sha256"] == metadata["source_sha256"]


@pytest.mark.skipif(not SOURCE.is_file(), reason="public MgO DFT displacement file is not installed")
def test_generic_condition_force_checkpoint_compares_64_model_with_512_reference(tmp_path):
    pytest.importorskip("phonopy")
    from curator.layer import PairwiseDistance, VariationalEnergyOutput
    from curator.layer._differentiable_inputs import DifferentiableInputs
    from curator.model import NeuralNetworkPotential, Painn

    torch.manual_seed(73)
    model = NeuralNetworkPotential(
        Painn(1, 4, 5.0, num_basis=4),
        input_modules=[PairwiseDistance(compute_neighbor_list=True, compute_distance_from_R=True,
                                        compute_forces=True, cutoff=5.0),
                       DifferentiableInputs(uniform_field=False, strain=False)],
        output_modules=[VariationalEnergyOutput(
            state_kind="vector", components=3, hidden_features=4, state_unit="e*Angstrom",
            conjugate_unit="V/Angstrom", compute_forces=False, compute_responses=False,
            observables=[{"name": "condition_forces", "derivatives": ["positions"],
                          "factor": -1, "unit": "eV/Angstrom"}])],
    ).double()
    checkpoint_directory = tmp_path / "checkpoints"
    checkpoint_directory.mkdir()
    checkpoint = checkpoint_directory / "best_model_epoch=0001_val_loss=0.5.ckpt"
    torch.save({"model": model}, checkpoint)
    (checkpoint_directory / "best_model_epoch=0000_val_loss=3.0.ckpt").write_bytes(b"not the selected model")
    args = argparse.Namespace(params=SOURCE, cell="64", checkpoint=checkpoint_directory, device="cpu",
                              cutoff=None, total_charge=None, nac_direction=None,
                              output=tmp_path / "comparison", holdout_reference_512=True,
                              split_metadata=_split_metadata(tmp_path / "split.json"))
    metadata = phonons.evaluate(args)
    assert metadata["status"] == "model_reference_comparison"
    assert metadata["model"]["coordinate_hvp_count"] == 192
    assert metadata["model"]["total_charge_condition"] is None
    assert metadata["model"]["state_mode"] == "predicted"
    assert metadata["model"]["checkpoint"] == str(checkpoint)
    assert metadata["model"]["checkpoint_selection_metric"] == 0.5
    assert metadata["holdout_reference_512"]["additional_model_coordinate_hvp_count"] == 0
    assert np.isfinite(metadata["holdout512_comparison"]["asr"]["qpoint_sorted_frequency_rmse_THz"])
    arrays = np.load(tmp_path / "comparison.npz")
    expected = np.sqrt(np.mean((arrays["model_qpoints_asr_THz"] - arrays["holdout512_qpoints_asr_THz"]) ** 2))
    assert metadata["holdout512_comparison"]["asr"]["qpoint_sorted_frequency_rmse_THz"] == pytest.approx(expected)
    assert arrays["model_fc_raw"].shape == (64, 64, 3, 3)


def test_heldout_claim_requires_experiment_split_metadata():
    args = argparse.Namespace(cell="64", holdout_reference_512=True, split_metadata=None)
    with pytest.raises(ValueError, match="requires --split-metadata"):
        phonons.evaluate(args)
    audit = phonons.split_provenance(None, {})
    assert not audit["verified"]
    assert "not a held-out claim" in audit["interpretation"]
