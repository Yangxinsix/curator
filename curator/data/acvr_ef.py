"""Physical ASE energy/force labels with declarative charge boundaries."""
import copy
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from ase.data import chemical_symbols
from ase.io import read
from torch.utils.data import DataLoader, Dataset

from . import properties
from .atoms_data import AtomsData
from .collate_atoms_data import collate_atoms_data
from .charge_conditions import ChargeBoundary
from .variational_data import VariationalCondition, VariationalData, canonical_boundary_hash, geometry_hash


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class ACVREFDataset(Dataset):
    def __init__(self, samples, indices):
        self.samples, self.indices = samples, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return copy.deepcopy(self.samples[self.indices[index]])

    def get_n_atoms(self, index):
        return int(self.samples[self.indices[index]][properties.n_atoms].item())


class ACVREFDataModule(pl.LightningDataModule):
    """One configured charge condition per geometry, without charge targets.

    Energies and forces are loaded without reference subtraction or rescaling.
    ``total_charge`` is an explicit boundary assumption, or ``total_charge_key``
    reads each frame info. ``boundary`` selects fixed charge or fixed potential;
    ``energy_convention`` declares internal or thermodynamic energy labels.
    Forces always correspond to the boundary's thermodynamic potential.
    Neither path reads ASE initial_charges. The split uses a dedicated RNG.
    """
    def __init__(self, datapath, batch_size=12, val_batch_size=6, seed=3407,
                 num_train=270, num_val=30, split_file=None, manifest_dir=None,
                 total_charge=0.0, default_dtype='float32', num_workers=0,
                 shuffle=True, heads=None, num_test=0, total_charge_key=None, energy_reference_file=None,
                 boundary=None, energy_convention=None, charge_mode='variational'):
        super().__init__()
        self.datapath = str(Path(datapath).expanduser().resolve())
        self.train_path = self.datapath
        self.batch_size, self.val_batch_size = int(batch_size), int(val_batch_size)
        self.seed, self.num_train, self.num_val = int(seed), int(num_train), int(num_val)
        self.split_file, self.manifest_dir = split_file, manifest_dir
        self.total_charge = float(total_charge)
        self.num_test = int(num_test)
        self.total_charge_key = total_charge_key
        if OmegaConf.is_config(boundary):
            boundary = OmegaConf.to_container(boundary, resolve=True)
        self.boundary = ChargeBoundary(**dict(boundary)) if boundary is not None else None
        if charge_mode not in ('variational', 'latent'):
            raise ValueError('charge_mode must be variational or latent')
        if charge_mode == 'latent' and (self.boundary is None or self.boundary.type != 'none'):
            raise ValueError('latent charge_mode requires explicit boundary.type=none')
        self.charge_mode = charge_mode
        if self.boundary is not None and (total_charge_key is not None or self.total_charge != 0):
            raise ValueError('use boundary or legacy total_charge arguments, not both')
        if energy_convention not in (None, 'internal', 'thermodynamic'):
            raise ValueError('energy_convention must be internal or thermodynamic')
        if self.boundary is not None and energy_convention is None:
            raise ValueError('explicit boundary requires an energy_convention')
        self.energy_convention = energy_convention
        self.energy_reference_file = energy_reference_file
        self.energy_reference = ({int(k): float(v) for k, v in json.loads(
            Path(energy_reference_file).read_text())['per_element_eV'].items()}
            if energy_reference_file else {})
        if not all(np.isfinite(v) for v in self.energy_reference.values()):
            raise ValueError('nonfinite energy reference')
        if self.num_test < 0:
            raise ValueError('num_test must be nonnegative')
        self.default_dtype = {'float32': torch.float32, 'float64': torch.float64}[str(default_dtype).removeprefix('torch.')]
        self.num_workers, self.shuffle = int(num_workers), bool(shuffle)
        self.heads = list(heads or ['energy', 'forces'])
        if set(self.heads) != {'energy', 'forces'}:
            raise ValueError('ACVR EF data exposes exactly energy and forces; no charge supervision')
        if min(self.batch_size, self.val_batch_size, self.num_train, self.num_val) < 1 or self.num_workers < 0:
            raise ValueError('positive batch/split sizes and nonnegative workers required')
        if not np.isfinite(self.total_charge):
            raise ValueError('total_charge must be finite')
        self.generator = torch.Generator().manual_seed(self.seed)
        self.datasets = {}

    def setup(self, stage=None):
        if self.datasets:
            return
        from curator.variational import BoundarySpec, ConstraintSpec, GroupSpec, VariableSpec
        frames = read(self.datapath, index=':')
        digest = _sha256(self.datapath)
        if self.split_file:
            split = json.loads(Path(self.split_file).read_text())
        else:
            permutation = torch.randperm(len(frames), generator=torch.Generator().manual_seed(self.seed)).tolist()
            split = {'train': permutation[:self.num_train],
                     'validation': permutation[self.num_train:self.num_train + self.num_val], 'test': permutation[self.num_train + self.num_val:self.num_train + self.num_val + self.num_test]}
        if set(split) != {'train', 'validation', 'test'}:
            raise ValueError('split must define train, validation, and test')
        indices = split['train'] + split['validation'] + split['test']
        if (len(split['train']) != self.num_train or len(split['validation']) != self.num_val
                or len(split['test']) != self.num_test
                or any(type(i) is not int or not 0 <= i < len(frames) for i in indices)
                or len(set(indices)) != len(indices) or len(indices) != len(frames)):
            raise ValueError('split must partition every source frame exactly once at the requested sizes')
        split_by_index = {i: phase for phase, values in split.items() for i in values}
        variable = VariableSpec('charge', 'charge', unit='e', conjugate_unit='V')
        samples = []
        charges = []
        charge_origin = ('source_info:' + str(self.total_charge_key) if self.total_charge_key
                         else 'explicit_user_assumption')
        if self.boundary is not None:
            charge_origin = ('source_info:' + self.boundary.total_charge_key if self.boundary.total_charge_key
                             else 'explicit_user_assumption' if self.boundary.total_charge is not None
                             else 'open_charge_reservoir')
            if self.boundary.type == 'none':
                charge_origin = 'unconstrained'
        for index, frame in enumerate(frames):
            count = len(frame)
            if self.boundary is not None:
                specification, charge = self.boundary.build(frame)
                if self.charge_mode == 'latent':
                    specification = replace(specification, variable=replace(
                        specification.variable, semantics='predicted_auxiliary'))
            elif self.total_charge_key:
                if self.total_charge_key not in frame.info:
                    raise ValueError(f'missing explicit total charge in frame {index}')
                charge = float(frame.info[self.total_charge_key])
            else:
                charge = self.total_charge
            if charge is not None and not np.isfinite(charge):
                raise ValueError(f'nonfinite total charge in frame {index}')
            charges.append(charge)
            if self.boundary is None:
                specification = BoundarySpec(variable, groups=(GroupSpec.all(),), equalities=(
                    ConstraintSpec.group_sum('total_charge', 'all', charge, unit='e'),))
            boundary = asdict(specification)
            boundary_hash = canonical_boundary_hash(boundary)
            atoms = {
                properties.atomic_numbers: torch.tensor(frame.numbers, dtype=torch.long),
                properties.positions: torch.tensor(frame.positions, dtype=self.default_dtype),
                properties.cell: torch.tensor(np.asarray(frame.cell), dtype=self.default_dtype),
                properties.pbc: torch.tensor(frame.pbc.reshape(1, 3), dtype=torch.bool),
                properties.n_atoms: torch.tensor([count], dtype=torch.long),
            }
            if charge is not None:
                atoms[properties.total_charge] = torch.tensor([charge], dtype=self.default_dtype)
            targets = {
                properties.energy: torch.tensor([float(frame.get_potential_energy()) -
                    sum(self.energy_reference.get(int(z), 0.0) for z in frame.numbers)], dtype=self.default_dtype),
                properties.forces: torch.tensor(frame.get_forces(apply_constraint=False), dtype=self.default_dtype),
            }
            for name, value in {**atoms, **targets}.items():
                if not torch.isfinite(value).all():
                    raise ValueError(f'nonfinite source frame {index}: {name}')
            if count < 1 or targets[properties.forces].shape != (count, 3):
                raise ValueError(f'invalid source geometry/forces at frame {index}')
            uid = f'ase-{digest}-{index}'
            labels = {'condition_energy': targets[properties.energy], 'condition_forces': targets[properties.forces]}
            kind = self.boundary.type if self.boundary is not None else 'fixed_charge'
            label_metadata = {'condition_energy': {
                'energy_kind': self.energy_convention or 'source_total_energy'}}
            if self.energy_convention is not None:
                label_metadata['condition_forces'] = {'force_potential': 'thermodynamic'}
            condition = VariationalCondition(
                condition_uid=uid + '-' + kind.replace('_', '-'), electron_count=None, electron_count_convention=None,
                net_charge=atoms[properties.total_charge].reshape(()) if charge is not None else None,
                boundary_spec=boundary,
                boundary_hash=boundary_hash, gauge={'boundary': 'fixed_total_charge' if kind == 'fixed_charge' else kind,
                    'total_charge_origin': charge_origin, 'energy_zero': 'source_calculator'},
                labels=labels, label_units={'condition_energy': 'eV', 'condition_forces': 'eV/Angstrom'},
                label_metadata=label_metadata,
                label_eligible={'condition_energy': True, 'condition_forces': True},
                provenance={'source_index': index, 'split': split_by_index[index]})
            group_indices = [list(range(count)) if group.selection == 'all' else list(group.indices)
                             for group in specification.groups]
            group_ptr = [0]
            for indices_in_group in group_indices:
                group_ptr.append(group_ptr[-1] + len(indices_in_group))
            payload = VariationalData(schema_version='1', geometry_uid=uid,
                geometry_hash=geometry_hash(atoms[properties.atomic_numbers], atoms[properties.positions],
                    atoms[properties.cell], atoms[properties.pbc]), variable_spec=asdict(specification.variable),
                group_member_ptr=torch.tensor(group_ptr),
                group_atom_index=torch.tensor([i for group in group_indices for i in group], dtype=torch.long),
                group_names=tuple(group.name for group in specification.groups),
                group_roles=tuple('electrode' if kind == 'fixed_potential' and group.name != 'system'
                                  else 'system' for group in specification.groups), conditions=(condition,),
                provenance={'split': split_by_index[index]})
            payload.validate(count)
            samples.append(AtomsData(atoms=atoms, targets=targets, task='acvr_ef', variational=payload,
                meta={'source_index': index, 'split': split_by_index[index]}))
        numbers = sorted({int(z) for frame in frames for z in frame.numbers})
        self.species = [chemical_symbols[z] for z in numbers]
        self.metadata = {'schema': 'acvr_ef_v1', 'source': self.datapath, 'source_sha256': digest,
            'source_frames': len(frames), 'split': split,
            'split_sha256': hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest(),
            'split_seed': self.seed, 'split_algorithm': 'torch.randperm with dedicated generator',
            'historical_split_identity_verified': False, 'independent_test_set': bool(self.num_test),
            'total_charge_e': sorted(set(charges)) if self.total_charge_key else self.total_charge, 'total_charge_origin': charge_origin,
            'labels': ['condition_energy', 'condition_forces'], 'atomic_charge_supervision': False,
            'label_transform': 'none', 'forces_apply_constraint': False,
            'units': {'energy': 'eV', 'forces': 'eV/Angstrom', 'charge': 'e', 'conjugate': 'V'},
            'species': self.species, 'dtype': str(self.default_dtype)}
        if self.boundary is not None:
            self.metadata.update(boundary=asdict(self.boundary),
                                 total_charge_e=sorted({q for q in charges if q is not None}) or None,
                                 total_charge_origin=charge_origin)
            if self.boundary.type == 'none':
                self.metadata.update(charge_mode=self.charge_mode, state_semantics=(
                    'predicted_auxiliary' if self.charge_mode == 'latent' else 'variational_internal'))
        if self.energy_convention is not None:
            self.metadata.update(energy_convention=self.energy_convention, force_potential='thermodynamic')
        if self.energy_reference_file:
            self.metadata['energy_reference'] = {str(k): v for k, v in self.energy_reference.items()}
            self.metadata['energy_reference_sha256'] = _sha256(self.energy_reference_file)
            self.metadata['label_transform'] = 'subtract train-only elemental energy reference in float64 before dtype conversion; forces unchanged'
        self.train_idx, self.val_idx, self.test_idx = split['train'], split['validation'], split['test']
        self.datasets = {name: ACVREFDataset(samples, values) for name, values in split.items()}
        self.train_dataset, self.val_dataset = self.datasets['train'], self.datasets['validation']
        self.dataset = self.train_dataset
        if self.manifest_dir:
            root = Path(self.manifest_dir)
            root.mkdir(parents=True, exist_ok=True)
            for name, value in [('split.json', split), ('manifest.json', self.metadata)]:
                path = root / name
                if path.exists() and json.loads(path.read_text()) != value:
                    raise ValueError(f'refusing to overwrite incompatible data manifest: {path}')
                if not path.exists():
                    path.write_text(json.dumps(value, indent=2) + '\n')

    def _loader(self, phase):
        self.setup()
        training = phase == 'train'
        return DataLoader(self.datasets[phase], batch_size=self.batch_size if training else self.val_batch_size,
            shuffle=training and self.shuffle, generator=self.generator if training else torch.Generator().manual_seed(self.seed + 1),
            num_workers=self.num_workers, collate_fn=collate_atoms_data)

    def train_dataloader(self):
        return self._loader('train')

    def val_dataloader(self):
        return self._loader('validation')

    def test_dataloader(self):
        return self._loader('test')

    def state_dict(self):
        self.setup()
        return {'metadata': self.metadata, 'generator_state': self.generator.get_state()}

    def load_state_dict(self, state_dict):
        self.setup()
        if state_dict['metadata'] != self.metadata:
            raise ValueError('checkpoint source, label semantics, or split does not match')
        self.generator.set_state(state_dict['generator_state'])
