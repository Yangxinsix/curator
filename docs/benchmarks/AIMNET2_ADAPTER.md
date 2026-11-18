# AIMNet2 adapter for Curator

## What is implemented

The new AIMNet2Adapter representation wraps an official complete PyTorch energy model.
Curator retains its existing NeuralNetworkPotential, GradientOutput, LitNNP, data loading,
energy/force losses, optimizer, and checkpoint machinery. There is no new trainer and
no call through an ASE calculator or NumPy conversion. The adapter is opt-in and does
not modify existing representations.

The initial supported route is the authors' legacy TorchScript model in the ACDB
molecular-collision repository. Its energy includes the original neural atomic
energy, atomic reference shifts, long-range Coulomb term, and DFT-D3 term. The
adapter does not add another Coulomb correction. Coordinates are Angstrom, energy
is eV, and conservative forces are eV/Angstrom.

A molecule is sent directly to the original model as coord=(1,N,3), numbers=(1,N),
and charge=(1,). Different-size molecules are evaluated independently and their
energies are concatenated. This avoids cross-molecule interactions and preserves
PyTorch graphs through the coordinate indexing. It is a correctness-first adapter,
not optimized batched sparse AIMNet inference.

Only isolated molecules are supported; any periodic axis is rejected. No stress,
response feature export, ACVR stationarity residual, or proposal network is claimed
for this baseline. Missing total_charge is an error. Atomic charge labels are never
read. A spin_multiplicity graph tensor, when supplied, maps to AIMNet's mult input.

## Usage

    from curator.model.aimnet2_adapter import AIMNet2Adapter
    from curator.model.base import NeuralNetworkPotential
    from curator.layer._grad_output import GradientOutput

    representation = AIMNet2Adapter(
        model_path="/path/to/acdb_sa_bs_aimnet2.jpt",
        trainable=True,
        energy_reference={1: ..., 8: ..., 16: ...},  # train-only fitted eV references
    )
    model = NeuralNetworkPotential(
        representation=representation,
        output_modules=[
            GradientOutput(grad_on_edge_diff=False, grad_on_positions=True)
        ],
    )

Omit energy_reference for the author's original absolute energies. For the current
centered ACVREFDataModule, pass exactly the same train-only per_element_eV table.
Subtraction is performed in float64. This only changes the energy zero and never
changes forces. Missing reference species fail explicitly.

Use an ordinary Curator LitNNP with energy and forces outputs and no charge output.
Validation must run with gradients enabled and Lightning inference_mode=False,
as for the existing conservative-force models. No neighbor-list preprocessing is
needed: the original AIMNet dense-mode model manages its complete molecular pairs.

The trusted TorchScript loader is explicit even when the upstream file suffix is
.pkl. Nested ScriptModules need special serialization; the adapter saves their JIT
bytes and restores them on the checkpoint device, preserving native Curator model
checkpoints. New AIMNetCentral v2 files are not legacy JIT files. Their bare energy
model may externalize Coulomb or D3; metadata indicating missing terms is rejected.
An independently assembled complete eager nn.Module can be provided through
model=..., complete_energy=True. This declaration must not be used to relabel a
short-range-only model as full AIMNet2.

## Which parameters train

The ACDB export has 2,498,593 parameters but disables gradients on all of them.
trainable=True explicitly re-enables only the neural afv, conv_a, conv_q, mlps,
and outputs.energy_mlp modules. AEV widths/centres/cutoffs and atomic reference
shifts remain fixed. The selected prefixes are configurable and must select at
least one parameter. Default trainable=False is an inference reference.

This is EF fine-tuning of a published model when initialized from the ACDB file.
It is NOT a claim that the published checkpoint was trained without charge labels:
the authors' training configuration includes a charge-loss weight of 0.01.
Further, their model may already have seen geometries in our new random held-out
split. Do not report its score on that split as an independent fair EF-only test.
A fair from-scratch comparison must instantiate the original architecture with fresh
weights, use our frozen train split and EF losses, and reserve our test set from all
training. The adapter accepts such a complete eager module, but does not silently
replace the author's initialization with an invented reset recipe.

## Verified artifact and tests

Official source repository commit:
454a2e452b0228243781363393d8505798f8b353

Artifact:
Articles/neefjes26_long_range_NN/SA_BS/MODELS/AIMNet2/wB97X-3c/1sa_1bs_aimnet2_wb97x3c_model.pkl

Git blob: 910559fb657718babe16ef1a76e27ed7aa6a2608
Size: 12,781,025 bytes
SHA256: 6842861e7a0d87671d8f1c5543fa784010f1825c6b9dd6618bab4837b2d68d1d

HHU artifact:
 /data1/home/hzxy05/work/yangxin/acvr/datasets/longrange_benchmarks_20261009/reference_models/acdb_sa_bs_aimnet2.jpt

The test suite checks finite-difference conservative forces, force-loss gradients
to neural parameters, independent/interleaved structure batching, reference
subtraction, explicit boundary handling, partial-energy rejection, and JIT
checkpoint serialization. The optional real-artifact test additionally compares
direct original-model energy/force predictions, runs force-loss backward and one
optimizer update, and round-trips a full Curator NNP checkpoint.

Run with CURATOR_AIMNET2_TEST_MODEL and CURATOR_AIMNET2_TEST_DATA set inside the
runtime environment to enable the official artifact test; otherwise that one test
is skipped explicitly. CPU validation on 2026-10-10 passed all five tests in 6.73 s.
Log: experiments/benchmark_adapters_20261010/aimnet2/pytest.log.

## Primary sources

- Original model/data: https://github.com/elmjonas/ACDB/tree/454a2e452b0228243781363393d8505798f8b353/Articles/neefjes26_long_range_NN
- Exact architecture: https://github.com/elmjonas/ACDB/blob/454a2e452b0228243781363393d8505798f8b353/Articles/neefjes26_long_range_NN/aimnet2_model_definition.yaml
- AIMNet2 paper: https://doi.org/10.1039/D4SC08572H
- Maintained package and full-energy metadata contract: https://github.com/isayevlab/aimnetcentral


## Runnable Curator configuration

curator/configs/aimnet2_ef_finetune.yaml is a complete opt-in configuration with the
HHU ACDB-SA-BS paths, the frozen split, identical train-only reference table, and
energy/force-only ModelOutputs. Launch through the existing Curator entrypoint:

    python -c 'from curator.commands.hydra_entrypoints import train; train()' --config-name=aimnet2_ef_finetune

Submit formal GPU runs through Slurm and choose a fresh run_path. This config is
not a command that has been launched as part of implementation.

The ACDB Zenodo record 11422835 declares CC BY 4.0. The pinned GitHub tree has no
separate LICENSE file, so no additional repository-code license is inferred.
The original authors' artifact is stored outside Curator source with a SHA256
provenance JSON; third-party implementation source was not copied into Curator.

The complete YAML was independently instantiated through Hydra, converted with
the native float32 path, and evaluated on a real ACDB geometry. Combined energy
and force backward passed with 2,498,485 trainable neural parameters. The original
float64 energy constants and references survive device/dtype moves. See
experiments/benchmark_adapters_20261010/aimnet2/config_check.log.
