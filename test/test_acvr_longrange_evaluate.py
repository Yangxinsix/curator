"""Scan evaluation does not fabricate splits or treat zero placeholders as DFT."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

spec = importlib.util.spec_from_file_location('acvr_longrange_evaluate', Path(__file__).resolve().parents[1] / 'scripts/acvr_longrange_evaluate.py')
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


def _frame(charge=1.):
    frame = Atoms('H2', positions=[[0., 0., 0.], [1., 0., 0.]], pbc=False)
    frame.info['total_charge'] = charge
    return frame


def test_one_frame_scan_needs_no_training_split_and_has_explicit_boundary():
    from curator.data import properties
    frame = _frame()
    loader = evaluate.scan_loader([frame], torch.float64)
    assert len(loader.dataset) == 1
    sample = loader.dataset[0]
    assert sample.targets == {}
    assert len(sample.variational.conditions) == 1
    condition = sample.variational.conditions[0]
    assert condition.labels == {} and float(condition.net_charge) == 1.
    assert condition.provenance['evaluation_only']
    batch = next(iter(loader))
    assert batch[properties.positions].dtype == torch.float64
    assert batch['meta'][0]['scan_index'] == 0
    assert not batch[properties.pbc].any()


def test_scan_preserves_order_and_requires_explicit_charge():
    frames = [_frame(1.), _frame(-1.)]
    loader = evaluate.scan_loader(frames, torch.float32)
    assert [float(sample.variational.conditions[0].net_charge) for sample in loader.dataset] == [1., -1.]
    del frames[0].info['total_charge']
    with pytest.raises(ValueError, match='Missing explicit'):
        evaluate.scan_loader(frames, torch.float32)


def test_unlabelled_scan_missing_or_placeholder_calculator_never_produces_zero_error():
    frame = _frame()
    assert evaluate.reference_labels(frame, allow_missing=True) == (None, None)
    frame.calc = SinglePointCalculator(frame, energy=0., forces=np.zeros((2, 3)))
    frame.info['reference_labels_available'] = False
    assert evaluate.reference_labels(frame, allow_missing=True) == (None, None)
    with pytest.raises(ValueError, match='real energy and force labels'):
        evaluate.reference_labels(frame)


def test_actual_labels_including_physical_zero_are_kept_without_placeholder_flag():
    frame = _frame()
    force = np.array([[0.1, 0., 0.], [-0.1, 0., 0.]])
    frame.calc = SinglePointCalculator(frame, energy=0., forces=force)
    energy, actual_force = evaluate.reference_labels(frame)
    assert energy == 0.
    np.testing.assert_array_equal(actual_force, force)
