"""DPA4C-Small with the existing one-pass RMSNorm/global-b ACVR charge model."""
import os
import sys
from pathlib import Path

import torch

CAMPAIGN = Path(__file__).resolve().parents[1] / "periodic_qefa"
sys.path.insert(0, str(CAMPAIGN / "adapter"))
# The archived campaign selects its Curator variational source and SOAP/KAN
# modules at import time, just as in its original training entry point.
import scratch as campaign
from charge_only import install_charge_only
from curator.data import properties
from curator.layer import PairwiseDistance, VariationalEnergyOutput
from curator.layer.efa.shared import SharedGeometryChargeProposal, SharedTwoScaleQConditionedEFAEnergy
import curator.model
from curator.model import NeuralNetworkPotential
from multiboundary import install_scratch_multiboundary
from rms_parent import RMSChargeEnergy, ScalarRMSNorm

ScratchTrainer = campaign.ScratchTrainer
calibrate_training_inputs = campaign.calibrate_training_inputs

# The ACVR Curator checkout predates DPA4C. Use the actual representation
# from unified Curator alongside the variational machinery in the campaign.
DPA_SOURCE = Path(os.environ.get("CURATOR_DPA4C_SOURCE", Path.home() / "curator-lammps-unified"))
dpa_package = DPA_SOURCE / "curator" / "model"
if not (dpa_package / "dpa4c.py").is_file():
    raise FileNotFoundError(f"CURATOR_DPA4C_SOURCE must contain curator/model/dpa4c.py: {DPA_SOURCE}")
curator.model.__path__.append(str(dpa_package))
from curator.model.dpa4c import DPA4C

# DPA4C clears optional native gradient outputs in its reference forward.
# The earlier ACVR schema does not declare these two unified dictionary keys.
for key in ("precomputed_edge_energy_gradient", "precomputed_forces"):
    if not hasattr(properties, key):
        setattr(properties, key, "_" + key)


def build_model(species, *, seed=3407, energy_convention="internal", dtype=torch.float64,
                cutoff=5.5, backbone_state=None, periodic_backend="legacy", backbone_backend="reference",
                native_proposal_spline=False, native_qefa=False):
    """Build Small with the molecular EFA or the campaign's shared-spectrum periodic RSA."""
    if periodic_backend not in {"legacy", "rsa"}:
        raise ValueError("periodic_backend must be legacy or rsa")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        backbone = DPA4C(species=species, channels=32, l_max=3, radial_modes=4, cutoff=cutoff,
                         readout_hidden_features=[96, 96, 96],
                         inference_backend=backbone_backend).to(dtype=dtype)
        if backbone_state is not None:
            backbone.load_state_dict(backbone_state, strict=True)
        energy = SharedTwoScaleQConditionedEFAEnergy(
            hidden_features=32, charge_features=32, curvature_floor=0.01,
            hardness_init=1.0, driving_scale=0.1, curvature_scale=0.1)
        head = VariationalEnergyOutput(
            internal_energy=energy, proposal=SharedGeometryChargeProposal(),
            state_unit="e", conjugate_unit="V", residual_scale=1.0, compute_forces=True,
            state_mode="predicted", residual_evaluation="reuse",
            residual_gradient_policy="proposal_only", energy_convention=energy_convention)
        model = NeuralNetworkPotential(
            representation=backbone,
            input_modules=[PairwiseDistance(compute_neighbor_list=True,
                                            compute_distance_from_R=True, cutoff=cutoff)],
            output_modules=[head]).to(dtype=dtype)
        install_scratch_multiboundary(model, seed=seed)
        if native_proposal_spline:
            from native_spline import enable_native_splines
            enable_native_splines(head.proposal)
        install_charge_only(model)
        energy.feature_norm = ScalarRMSNorm(energy.feature_spec.irreps, next(energy.parameters()))
        energy.__class__ = RMSChargeEnergy
        if periodic_backend == "rsa":
            from rsa_candidate import install_rsa
            install_rsa(model)
        if native_qefa:
            from qefa_cuda import MolecularFastAttention, MolecularPeriodicRSA
            from curator.layer.efa.reciprocal_rsa import PeriodicRSA
            from curator.layer.efa.core import EuclideanFastAttention
            # Keep RSA periodic phases and cutoffs on the original implementation.
            for attention in model.modules():
                if type(attention) is EuclideanFastAttention:
                    attention.__class__ = MolecularFastAttention
                elif type(attention) is PeriodicRSA:
                    attention.__class__ = MolecularPeriodicRSA
        return campaign.install_linear_neighbors(model).requires_grad_(True)
