"""Train a fresh PaiNN/EFA/rank-16 KAN from labelled ASE frames at specified charge boundaries.

Run: python scratch.py CONFIG.json
The JSON contains "data" (mixed_data.build_data) and "training". Stage lengths
are explicit budgets. Checkpoints are written at stage boundaries. The warmup
and joint phases use the same native, non-iterative charge forward.
"""
import argparse
from itertools import islice
import json
import math
import os
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "efa-proposal-speed-20261022/source_snapshot"
os.environ.setdefault("ACVR_SOURCE", str(SOURCE))
SCRIPTS = os.environ.get("ACVR_SCRIPTS", str(HERE.parent / "proposal-response-kan-20261102/scripts"))
sys.path[:0] = [os.environ["ACVR_SOURCE"], SCRIPTS]
from curator.layer import PairwiseDistance, VariationalEnergyOutput
from curator.layer.efa.shared import SharedGeometryChargeProposal, SharedTwoScaleQConditionedEFAEnergy
from curator.model import Painn, NeuralNetworkPotential
from curator.train.train import tree_to_device
from curator.train.variational_losses import VariationalJointLoss
from multiboundary import install_scratch_multiboundary
from soap import SOAP


def build_scratch_model(seed=3407, energy_convention="internal", dtype=torch.float64):
    """Randomly initialize every learned weight. No reference charge or checkpoint is read."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        backbone = Painn(num_interactions=3, num_features=128, cutoff=6.0, heads=["energy"])
        energy = SharedTwoScaleQConditionedEFAEnergy(
            hidden_features=32, charge_features=32, curvature_floor=0.01,
            hardness_init=1.0, driving_scale=0.1, curvature_scale=0.1)
        output = VariationalEnergyOutput(
            internal_energy=energy, proposal=SharedGeometryChargeProposal(), state_unit="e",
            conjugate_unit="V", residual_scale=1.0, compute_forces=True, state_mode="predicted",
            residual_evaluation="reuse", residual_gradient_policy="proposal_only",
            energy_convention=energy_convention)
        model = NeuralNetworkPotential(
            representation=backbone, input_modules=[PairwiseDistance(
                compute_neighbor_list=True, compute_distance_from_R=True, cutoff=6.0)],
            output_modules=[output]).to(dtype=dtype)
        install_scratch_multiboundary(model, seed=seed)
        return model.requires_grad_(True)


def load_checkpoint(path, device="cpu"):
    """Reconstruct a deployable/evaluable model from this runner's checkpoint."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_scratch_model(**state["model_options"])
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()


def calibrate_training_inputs(model, train_batches):
    """Measure scaling once, after physical warmup, before the first proposal update."""
    proposal = model.output_modules[0].proposal
    # Warmup has not changed these identity buffers or introduced nonzero splines.
    # Observe actual native inputs in each pass so later moments include prior normalization.
    def estimate(module, center, scale, floor):
        if not torch.all(center == 0) or not torch.all(scale == 1):
            raise ValueError("Calibration requires fresh identity normalization")
        count = 0
        mean = torch.zeros_like(center, dtype=torch.float64)
        m2 = torch.zeros_like(mean)
        def observe(layer, inputs):
            nonlocal count, mean, m2
            values = inputs[0].detach().reshape(-1, center.numel()).double()
            size = values.shape[0]
            local_mean = values.mean(0)
            shift = local_mean - mean
            m2 += (values - local_mean).square().sum(0) + shift.square() * (count * size / (count + size))
            mean += shift * (size / (count + size))
            count += size
        hook = module.register_forward_pre_hook(observe)
        try:
            for batch in train_batches:
                model(tree_to_device(batch, center.device))
        finally:
            hook.remove()
        if not count or not torch.isfinite(mean).all() or not torch.isfinite(m2).all():
            raise ValueError("Need finite training features for calibration")
        center.copy_(mean)
        std = (m2 / count).sqrt()
        # A constant training coordinate has no measured scale. In particular
        # a Q or Q/N value seen only in one ensemble must not acquire a 1e6 gain.
        scale.copy_(torch.where(std >= floor, std, torch.ones_like(std)))
    model.eval()
    try:
        estimate(proposal.physical_input, proposal.physical_center, proposal.physical_scale, 1e-6)
        estimate(proposal.operator_head, proposal.invariant_center, proposal.invariant_scale, 1e-6)
        for layer in (proposal.operator_head.first, proposal.operator_head.last):
            estimate(layer, layer.center, layer.scale, 1e-3)
    finally:
        model.train()


class ScratchTrainer:
    def __init__(self, model, *, proposal_lr=1e-4, energy_lr=1e-5, backbone_lr=5e-6,
                 energy_weight=1., force_weight=1.):
        self.model = model
        head = model.output_modules[0]
        self.proposal = list(head.proposal.parameters())
        self.energy = list(head.internal_energy.parameters())
        self.backbone = list(model.representation.parameters())
        self.physical = self.energy + self.backbone
        ids = [id(p) for p in self.physical + self.proposal]
        if len(set(ids)) != len(ids) or set(ids) != {id(p) for p in model.parameters()}:
            raise ValueError("Physical and proposal parameter sets must be disjoint and complete")
        self.physics_optimizer = torch.optim.Adam([
            {"params": self.backbone, "lr": backbone_lr}, {"params": self.energy, "lr": energy_lr}])
        self.solver_optimizer = SOAP(self.proposal, lr=proposal_lr, weight_decay=0.,
                                     betas=(.95, .95), precondition_frequency=10)
        self.supervision = VariationalJointLoss(
            supervised_weights={"condition_energy": energy_weight, "condition_forces": force_weight},
            supervised_scales={"condition_energy": 1.0, "condition_forces": 1.0},
            required_labels=["condition_energy", "condition_forces"], energy_per_atom=True)

    def step(self, batch, phase):
        """Route gradients by parameters, keeping the full coordinate derivative for forces."""
        if phase not in {"energy_warmup", "proposal_warmup", "joint"}:
            raise ValueError(f"Unknown phase {phase}")
        self.model.train()
        for optimizer in (self.physics_optimizer, self.solver_optimizer):
            optimizer.zero_grad(set_to_none=True)
        batch = tree_to_device(batch, self.physical[0].device)
        prediction = self.model(batch)
        residual = prediction["variational_prediction"]["residuals"]["stationarity"]["values"]
        residual_mse = residual.square().mean()
        reports = {"stationarity_rms_V": float(residual_mse.detach().sqrt())}
        if phase != "proposal_warmup":
            physical_loss = self.supervision(prediction, batch)["loss"]
            if not torch.isfinite(physical_loss):
                raise FloatingPointError("Nonfinite physical loss")
            torch.autograd.backward(physical_loss, inputs=self.physical, retain_graph=phase == "joint")
            reports["physical_loss"] = float(physical_loss.detach())
        if phase != "energy_warmup":
            torch.autograd.backward(30 * residual_mse, inputs=self.proposal)
        if not torch.isfinite(residual_mse):
            raise FloatingPointError("Nonfinite stationarity residual")
        # Finish both backward passes and validate both groups before moving either model.
        active = []
        if phase != "proposal_warmup":
            active.append(("physical", self.physical, self.physics_optimizer))
        if phase != "energy_warmup":
            active.append(("proposal", self.proposal, self.solver_optimizer))
        for name, parameters, _ in active:
            reports[name + "_grad_norm"] = float(torch.nn.utils.clip_grad_norm_(
                parameters, 10.0, error_if_nonfinite=True))
        for _, _, optimizer in active:
            optimizer.step()
        return reports

    def validate(self, batches):
        self.model.eval()
        numerator = {"condition_energy": 0., "condition_forces": 0.}
        denominator = dict.fromkeys(numerator, 0.)
        by_boundary = {}
        for batch in batches:
            batch = tree_to_device(batch, self.physical[0].device)
            pred = self.model(batch)
            terms = self.supervision(pred, batch)["terms"]
            for name in numerator:
                numerator[name] += float(terms[name]["numerator"].detach())
                denominator[name] += float(terms[name]["denominator"].detach())
            stat = pred["variational_prediction"]["residuals"]["stationarity"]
            ptr = stat["ptr"].tolist()
            for i, boundary in enumerate(batch["variational"]["boundary_spec"]):
                fixed, voltage = bool(boundary["equalities"]), bool(boundary["controls"])
                kind = ("fixed_Q_plus_V" if voltage else "fixed_Q") if fixed else "open_V"
                row = by_boundary.setdefault(kind, [0., 0])
                r = stat["values"][ptr[i]:ptr[i + 1]].detach()
                row[0] += float(r.square().sum())
                row[1] += r.numel()
        return {"energy_rms_eV_per_atom": math.sqrt(numerator["condition_energy"]/denominator["condition_energy"]),
                "force_rms_eV_per_A_geometry_weighted": math.sqrt(numerator["condition_forces"]/denominator["condition_forces"]),
                "stationarity_rms_V_by_boundary": {kind: math.sqrt(sq/n) for kind, (sq,n) in by_boundary.items()}}


def endless(loader):
    while True:
        yield from loader


def run(config_path):
    from mixed_data import build_data
    cfg = json.loads(config_path.read_text())
    opt = cfg["training"]
    seed = int(opt.get("seed", 3407))
    torch.manual_seed(seed)
    data = build_data(cfg["data"], base_dir=config_path.parent)
    if cfg.get("energy_convention", data.energy_convention) != data.energy_convention:
        raise ValueError("Model and dataset energy conventions must agree")
    dtype = {"float32": torch.float32, "float64": torch.float64}[cfg["data"].get("default_dtype", "float64")]
    options = {"seed": seed, "dtype": dtype, "energy_convention": data.energy_convention}
    steps = {phase: opt[phase + "_steps"] for phase in ("energy_warmup", "proposal_warmup", "joint")}
    if any(type(n) is not int or n < 0 for n in steps.values()) or steps["joint"] < 1:
        raise ValueError("Nonnegative integer phase lengths and at least one joint update are required")
    calibration_count = opt.get("calibration_batches", 2)
    if type(calibration_count) is not int or calibration_count < 1:
        raise ValueError("calibration_batches must be a positive integer")
    output = config_path.parent / opt["output"]
    output.mkdir(parents=True, exist_ok=False)
    model = build_scratch_model(**options).to(opt.get("device", "cpu"))
    trainer = ScratchTrainer(model, **{k: opt[k] for k in (
        "proposal_lr", "backbone_lr", "energy_lr", "energy_weight", "force_weight") if k in opt})
    loader = data.train_dataloader()
    if len(loader) == 0:
        raise ValueError("Empty training loader")
    batches = endless(loader)
    for phase, length in steps.items():
        if phase == "proposal_warmup":
            # Guarantee that a source with few structures also calibrates its boundary inputs.
            samples = []
            for index, source in enumerate(data.sources):
                source_loader = torch.utils.data.DataLoader(
                    source.datasets["train"], batch_size=data.batch_size, shuffle=True,
                    generator=torch.Generator().manual_seed(seed + index), collate_fn=loader.collate_fn)
                samples.extend(islice(source_loader, calibration_count))
            calibrate_training_inputs(model, samples)
        if not length:
            continue
        for step in range(1, length + 1):
            metrics = trainer.step(next(batches), phase)
            if step == 1 or step == length or step % 100 == 0:
                print(json.dumps({"phase": phase, "step": step, **metrics}), flush=True)
        validation = trainer.validate(data.val_dataloader())
        print(json.dumps({"phase": phase, "validation": validation}), flush=True)
        path = output / (phase + ".pt")
        torch.save({"model": model.state_dict(), "model_options": options,
                    "phase": phase, "optimizer_calls_requested": steps, "validation": validation,
                    "config": cfg}, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)
    return model.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    run(parser.parse_args().config.resolve())
