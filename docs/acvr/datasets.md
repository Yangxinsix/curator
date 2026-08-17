# ACVR data selection and capability contracts

ACVR separates two questions that are often mixed together:

1. Can a dataset train a useful energy/force model with charge-dependent long
   range interactions?
2. Can it validate a claimed thermodynamic boundary or response?

A dataset can answer the first question without answering the second.  The
machine-readable `DatasetCapabilityManifest` makes this distinction explicit
before training or evaluation starts.

## Recommended first molecular dataset: SPICE 2.x

The official [SPICE repository](https://github.com/openmm/spice-dataset)
provides a CC0 HDF5 dataset with DFT energies, gradients and selected electronic
properties.  For the first ACVR charge experiments, use:

- **Ion Pairs** for charged fragments, separation curves and long-range
  extrapolation;
- **Dipeptides** for neutral/zwitterionic, chemically heterogeneous structures;
- optional solvated subsets only after the small smoke tests are stable.

The built-in `SpiceDataset` reads the official HDF5 schema lazily.  It converts
bohr/hartree data to Curator's angstrom/eV convention, converts energy gradients
to forces, obtains total charge from mapped-SMILES formal charges, and exposes
MBIS charges as optional auxiliary labels.  When MBIS data exist, their rounded
sum is checked against the formal molecular charge.

```yaml
data:
  _target_: curator.data.AtomsDataModule
  data_type: Spice
  datapath: /absolute/path/to/SPICE-2.0.1.hdf5
  cutoff: 6.0
  compute_neighbor_list: true
  batch_size: 16
  num_train: 0.8
  num_val: 0.1
  num_test: 0.1
  dataset_kwargs:
    subsets: [ion_pairs, dipeptides]
    energy_key: formation_energy
    include_atomic_charges: true
    include_dipole: true
    max_conformations_per_molecule: null
    stride: 1
```

The reader rejects explicit incompatible unit metadata instead of silently
converting it.  Historical files with no unit attributes remain accepted
because the official schema defines atomic units.

### Small official Ion Pairs artifact

The complete SPICE 2.0.1 release is a single 37.5 GB file.  For a first real
reader/training smoke test, use the pinned upstream QCArchive downloader with
[`example/acvr/spice_qcarchive_ion_pairs.yaml`](../../example/acvr/spice_qcarchive_ion_pairs.yaml)
as its `config.yaml`.  This downloads only
`SPICE Ion Pairs Single Points Dataset v1.2`.

The 2026-08-17 reference retrieval produced 28 molecular groups and 1428
two-atom conformations in a 215 KB file.  Its SHA-256 was
`631137e6ee16f6974721a5d6653b9f730fb57bd3dbea7e129e8c8a9f95e51ebd`.
The source artifact contains energies, gradients and SCF dipoles but no MBIS
charges, which is why total charge must not depend on a partition-charge field.

The optional real-artifact smoke test is enabled with:

```bash
CURATOR_SPICE_ION_PAIRS=/path/to/SPICE.hdf5 \
  python -m pytest -q test/test_spice_dataset.py
```

### Distance extrapolation split

Do not use a random conformation split to claim long-range generalization on
Ion Pairs.  Neighboring configurations on the same separation curve are too
similar.  Build a deterministic split in the flattened dataset index space:

```python
import json

from curator.data import SpiceDataset, ion_pair_distance_split

dataset = SpiceDataset(
    "/path/to/SPICE.hdf5",
    compute_neighbor_list=False,
    subsets=["ion_pairs"],
)
split = ion_pair_distance_split(dataset, 5.5, 6.5)
with open("distance_split_5.5_6.5.json", "w") as handle:
    json.dump(split, handle)
```

The intervals are train `r <= 5.5 angstrom`, validation
`5.5 < r <= 6.5 angstrom`, and test `r > 6.5 angstrom`.  The reference
1428-conformation artifact gives 868/280/280 samples.  Pass the resulting JSON
to the existing `AtomsDataModule.split_file`; no new trainer path is required.
See
[`example/acvr/spice_ion_pairs_distance.yaml`](../../example/acvr/spice_ion_pairs_distance.yaml).

The combined Ion Pairs/Dipeptides configuration remains useful as a reader and
training smoke test, but its random split is not long-range evidence.

## What SPICE can and cannot establish

SPICE is appropriate for:

- energy/force regression on isolated molecules;
- conditioning on a known molecular total charge;
- ion-pair distance and orientation extrapolation;
- checking whether dipole supervision, and MBIS supervision where available,
  improves observable transfer.

SPICE alone cannot validate:

- constant electrode potential;
- electrode capacitance or charge--voltage curves;
- the same atomic geometry at multiple applied voltages;
- a periodic metal--electrolyte interface.

MBIS charges are a partition-dependent auxiliary target, not a unique
observable.  They can regularize or diagnose a model, but they must not be used
as the sole evidence that a latent or variational charge is physically correct.

## Required second data tier

After the molecular implementation passes, construct or obtain a deliberately
small periodic electrochemical dataset containing the *same structural family*
under multiple electronic boundary conditions:

- different total electron counts or fragment charges;
- several imposed potential differences;
- DFT/cDFT energies and forces;
- electrode total charges and potential differences;
- dipoles/polarization or DFPT/finite-field response where defined.

That tier, not SPICE, is what identifies the cross-boundary physics required by
constant-potential simulations.

## Capability checks

```python
dataset.capabilities.require_labels(["energy", "forces", "total_charge"])
dataset.capabilities.require_claims(["ion_pair_distance_asymptote"])
```

Requesting `constant_electrode_potential` from the SPICE manifest raises an
error before an unsupported scientific claim can enter the benchmark.
