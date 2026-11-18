"""Real-label adapter must not leak charge partitions or mask constrained forces."""
import json
import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.io import write

from curator.data import properties
from curator.data.acvr_ef import ACVREFDataModule


def _source(tmp_path):
    frames = []
    for index in range(5):
        atoms = Atoms('HO', positions=[[0, 0, 0], [1 + .01 * index, 0, 0]], cell=[8, 8, 8], pbc=True)
        atoms.set_initial_charges([100 + index, -200 - index])
        atoms.set_constraint(FixAtoms(indices=[0]))
        atoms.calc = SinglePointCalculator(atoms, energy=-30.0 + index,
            forces=np.array([[1., 2., 3.], [-1., -2., -3.]]))
        frames.append(atoms)
    path = tmp_path / 'source.traj'
    write(path, frames)
    return path


def test_physical_labels_and_no_charge_supervision(tmp_path):
    source = _source(tmp_path)
    dm = ACVREFDataModule(source, num_train=3, num_val=2, batch_size=2, val_batch_size=2,
                         manifest_dir=tmp_path / 'manifest', default_dtype='float64')
    dm.setup()
    sample = dm.train_dataset[0]
    assert set(sample.targets) == {properties.energy, properties.forces}
    assert torch.equal(sample[properties.forces][0], torch.tensor([1., 2., 3.], dtype=torch.float64))
    assert sample[properties.total_charge].item() == 0
    condition = sample.variational.conditions[0]
    assert set(condition.labels) == {'condition_energy', 'condition_forces'}
    assert condition.net_charge.item() == 0
    assert condition.labels['condition_energy'].item() == -30 + sample.meta['source_index']
    batch = next(iter(dm.train_dataloader()))
    labels = batch['variational']['labels']
    assert set(labels) == {'condition_energy', 'condition_forces'}
    for entry in labels.values():
        assert entry['present'].all() and entry['eligible'].all()
    assert set(dm.train_idx).isdisjoint(dm.val_idx)
    assert sorted(dm.train_idx + dm.val_idx) == list(range(5))
    split = json.loads((tmp_path / 'manifest' / 'split.json').read_text())
    assert split['test'] == []
    second = ACVREFDataModule(source, num_train=3, num_val=2, default_dtype='float64')
    torch.manual_seed(999)
    second.setup()
    assert dm.metadata == second.metadata
    state = dm.state_dict()
    second.load_state_dict(state)
    assert torch.equal(dm.generator.get_state(), second.generator.get_state())


def test_reject_split_leakage_and_manifest_change(tmp_path):
    source = _source(tmp_path)
    path = tmp_path / 'split.json'
    path.write_text(json.dumps({'train': [0, 1, 2], 'validation': [2, 3], 'test': []}))
    with pytest.raises(ValueError, match='partition'):
        ACVREFDataModule(source, num_train=3, num_val=2, split_file=path).setup()
    manifest = tmp_path / 'manifest'
    ACVREFDataModule(source, num_train=3, num_val=2, manifest_dir=manifest).setup()
    with pytest.raises(ValueError, match='incompatible'):
        ACVREFDataModule(source, num_train=3, num_val=2, manifest_dir=manifest, total_charge=1).setup()


def test_explicit_charge_test_split_and_large_energy_reference(tmp_path):
    from ase.io import read
    source = _source(tmp_path)
    frames = read(source, ':')
    for index, atoms in enumerate(frames):
        atoms.info['total_charge'] = float(index % 2)
        atoms.calc = SinglePointCalculator(atoms, energy=-1000000.0 + .01 * index,
                                          forces=np.ones((2, 3)))
    write(source, frames)
    split_path = tmp_path / 'split.json'
    split_path.write_text(json.dumps({'train': [0, 1, 2], 'validation': [3], 'test': [4]}))
    ref = tmp_path / 'reference.json'
    ref.write_text(json.dumps({'per_element_eV': {'1': -500000., '8': -500000.}}))
    dm = ACVREFDataModule(source, num_train=3, num_val=1, num_test=1,
        split_file=split_path, total_charge_key='total_charge', energy_reference_file=ref)
    dm.setup()
    assert dm.test_idx == [4] and set(dm.test_idx).isdisjoint(dm.train_idx + dm.val_idx)
    sample = dm.datasets['validation'][0]
    assert sample[properties.total_charge].item() == 1
    assert sample.variational.conditions[0].net_charge.item() == 1
    assert sample[properties.energy].item() == pytest.approx(.03, abs=1e-8)
    batch = next(iter(dm.test_dataloader()))
    assert batch[properties.energy].item() == pytest.approx(.04, abs=1e-8)
    assert dm.metadata['independent_test_set']
    assert torch.equal(sample[properties.forces], torch.ones(2, 3))
    with pytest.raises(ValueError, match='missing explicit total charge'):
        ACVREFDataModule(source, num_train=3, num_val=1, num_test=1,
            split_file=split_path, total_charge_key='missing').setup()


@pytest.mark.parametrize('boundary, charge', [
    ({'type': 'fixed_charge', 'total_charge': 1.0}, 1.0),
    ({'type': 'fixed_potential', 'potential': .4}, None),
    ({'type': 'fixed_potential', 'potential': {'left': -.3, 'right': .3},
      'groups': {'left': [0], 'right': [1]}, 'total_charge': 0.0}, 0.0),
])
@pytest.mark.parametrize('convention', ['internal', 'thermodynamic'])
def test_declarative_boundary_and_energy_convention(tmp_path, boundary, charge, convention):
    from curator.layer._variational_output import _boundary_from_dict
    from curator.variational import compile_boundary, VariableLayout
    dm = ACVREFDataModule(_source(tmp_path), num_train=3, num_val=2,
                         boundary=boundary, energy_convention=convention)
    dm.setup()
    sample = dm.train_dataset[0]
    condition = sample.variational.conditions[0]
    spec = _boundary_from_dict(condition.boundary_spec)
    compiled = compile_boundary(spec, VariableLayout(spec.variable, 2))
    if charge is None:
        assert properties.total_charge not in sample.atoms
        assert condition.net_charge is None
        assert compiled.constraint_matrix.shape == (0, 2)
    else:
        assert condition.net_charge.item() == charge
        torch.testing.assert_close(compiled.constraint_target, torch.tensor([charge], dtype=torch.float64))
    if boundary['type'] == 'fixed_potential':
        assert compiled.control_matrix.shape[0] > 0
    assert dm.metadata['energy_convention'] == convention
    assert dm.metadata['force_potential'] == 'thermodynamic'
    assert condition.label_metadata['condition_energy']['energy_kind'] == convention
    assert condition.label_metadata['condition_forces']['force_potential'] == 'thermodynamic'
    # Neither labels nor losses are transformed into charge targets.
    assert set(next(iter(dm.train_dataloader()))['variational']['labels']) == {
        'condition_energy', 'condition_forces'}


def test_potential_and_groups_can_come_from_each_frame(tmp_path):
    from ase.io import read
    source = _source(tmp_path)
    frames = read(source, ':')
    for index, atoms in enumerate(frames):
        atoms.info.update(voltage={'left': -.1 * index, 'right': .2 * index},
                          electrodes={'left': [0], 'right': [1]})
    write(source, frames)
    dm = ACVREFDataModule(source, num_train=3, num_val=2,
        boundary={'type': 'fixed_potential', 'potential_key': 'voltage', 'groups_key': 'electrodes'},
        energy_convention='internal')
    dm.setup()
    for sample in dm.train_dataset:
        condition = sample.variational.conditions[0]
        index = sample.meta['source_index']
        assert [control['value'] for control in condition.boundary_spec['controls']] == [-.1 * index, .2 * index]
        assert sample.variational.group_names == ('left', 'right')
    assert not next(iter(dm.train_dataloader()))['variational']['net_charge_present'].any()


@pytest.mark.parametrize('boundary, convention, match', [
    ({'type': 'fixed_potential', 'potential': .3}, None, 'energy_convention'),
    ({'type': 'fixed_charge'}, 'internal', 'explicit total_charge'),
    ({'type': 'fixed_potential'}, 'internal', 'potential or'),
    ({'type': 'fixed_potential', 'potential_key': 'missing'}, 'internal', 'missing boundary'),
    ({'type': 'fixed_potential', 'potential': float('nan')}, 'internal', 'finite'),
    ({'type': 'fixed_potential', 'potential': .3, 'groups': {'a': [0]}}, 'internal', 'cover'),
])
def test_invalid_boundary_configuration_fails_before_training(tmp_path, boundary, convention, match):
    with pytest.raises(ValueError, match=match):
        ACVREFDataModule(_source(tmp_path), num_train=3, num_val=2,
                        boundary=boundary, energy_convention=convention).setup()


def test_hydra_boundary_configs_share_loss_and_serialize_metadata(tmp_path):
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'curator/configs'),
                               version_base=None):
        charge = compose(config_name='acvr_efa_charge')
        potential = compose(config_name='acvr_efa_potential')
    assert OmegaConf.to_container(charge.task.outputs) == OmegaConf.to_container(potential.task.outputs)
    potential.data.datapath = str(_source(tmp_path))
    potential.data.num_train, potential.data.num_val = 3, 2
    potential.data.manifest_dir = str(tmp_path / 'hydra_manifest')
    potential.data.boundary = {'type': 'fixed_potential', 'potential': {'left': -.1, 'right': .1},
                               'groups': {'left': [0], 'right': [1]}}
    dm = instantiate(potential.data)
    dm.setup()
    assert json.loads((tmp_path / 'hydra_manifest/manifest.json').read_text())['boundary']['groups'] == {
        'left': [0], 'right': [1]}


def test_null_charge_metadata_cannot_silently_open_fixed_charge_boundary():
    from curator.data.charge_conditions import ChargeBoundary
    atoms = Atoms('NaCl', info={'net_charge': None})
    boundary = ChargeBoundary(type='fixed_charge', total_charge_key='net_charge')
    with pytest.raises(ValueError, match='missing boundary input total_charge'):
        boundary.build(atoms)


@pytest.mark.parametrize('mode, semantics', [
    ('variational', 'variational_internal'), ('latent', 'predicted_auxiliary'),
])
def test_explicit_none_has_no_charge_or_potential_constraints(tmp_path, mode, semantics):
    from curator.layer._variational_output import _boundary_from_dict
    from curator.variational import compile_boundary, VariableLayout
    dm = ACVREFDataModule(_source(tmp_path), num_train=3, num_val=2,
        boundary={'type': 'none'}, charge_mode=mode, energy_convention='internal')
    dm.setup()
    sample = dm.train_dataset[0]
    assert properties.total_charge not in sample.atoms
    assert sample.variational.conditions[0].net_charge is None
    spec = _boundary_from_dict(sample.variational.conditions[0].boundary_spec)
    assert spec.variable.semantics == semantics
    compiled = compile_boundary(spec, VariableLayout(spec.variable, 2))
    assert compiled.constraint_matrix.shape == compiled.control_matrix.shape == (0, 2)
    assert dm.metadata['total_charge_e'] is None
    assert dm.metadata['total_charge_origin'] == 'unconstrained'
    assert dm.metadata['charge_mode'] == mode
    assert dm.metadata['state_semantics'] == semantics
    batch = next(iter(dm.train_dataloader()))
    assert not batch['variational']['net_charge_present'].any()
    assert set(batch['variational']['labels']) == {'condition_energy', 'condition_forces'}


@pytest.mark.parametrize('extra', [
    {'total_charge': 0}, {'total_charge_key': 'Q'}, {'potential': 0},
    {'groups': {'a': [0]}}, {'gauge': {'policy': 'allow'}},
])
def test_none_rejects_hidden_constraints(extra):
    from curator.data.charge_conditions import ChargeBoundary
    with pytest.raises(ValueError, match='cannot include'):
        ChargeBoundary(type='none', **extra)


def test_latent_requires_explicit_none_and_mode_is_checked(tmp_path):
    for boundary in (None, {'type': 'fixed_charge', 'total_charge': 0.0},
                     {'type': 'fixed_potential', 'potential': 0.0}):
        with pytest.raises(ValueError, match='requires explicit'):
            ACVREFDataModule(_source(tmp_path), charge_mode='latent', boundary=boundary)
    with pytest.raises(ValueError, match='charge_mode must'):
        ACVREFDataModule(_source(tmp_path), charge_mode='typo')


def test_unconstrained_presets_inherit_identical_loss_and_remove_charge_default(tmp_path):
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'curator/configs'),
                               version_base=None):
        configs = [compose(config_name=name) for name in
                   ('acvr_efa_charge', 'acvr_efa_unconstrained', 'acvr_efa_latent_proposal')]
    for cfg in configs[1:]:
        assert OmegaConf.to_container(cfg.task.outputs) == OmegaConf.to_container(configs[0].task.outputs)
        assert cfg.data.boundary.type == 'none' and cfg.data.boundary.total_charge is None
        cfg.data.datapath = str(_source(tmp_path))
        cfg.data.num_train, cfg.data.num_val = 3, 2
        cfg.data.manifest_dir = None
        dm = instantiate(cfg.data)
        dm.setup()
        assert dm.metadata['charge_mode'] == cfg.data.charge_mode
