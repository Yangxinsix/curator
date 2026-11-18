"""DPA4C molecular pure, official LES and one-pass ACVR updates on the same ragged minibatch."""

import os
from pathlib import Path
import sys

import torch
from torch import nn
from torch.optim import Adam

DPA = Path(__file__).resolve().parents[1] / "dpa4c_acvr"
sys.path.insert(0, str(DPA))
from model import build_model
from curator.data import properties as p
from curator.layer._latent_ewald import LatentEwaldEnergy
from curator.train.train import tree_to_device
from fixed_q_batch import panel_energy_charge


def build_les_model(species, *, seed, backbone_state=None, native=True):
    """Same Small representation, local E/F weights and official LES readout as the historical comparator."""
    source = Path(
        os.environ.get(
            "ACVR_LES_SOURCE",
            "/data1/home/hzxy05/work/yangxin/acvr/runtime/les-35b971a/src",
        )
    )
    if source.is_dir():
        sys.path.insert(0, str(source))
    model = build_model(
        species,
        seed=seed,
        backbone_backend="native" if native else "reference",
        backbone_state=backbone_state,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        les = LatentEwaldEnergy().bind(model.representation.response_feature_spec())
    model.output_modules = nn.ModuleList([les])
    return model


def physical_ef(model, batch, *, method, create_graph):
    data = batch.copy()
    for layer in model.input_modules:
        data = layer(data)
    data = model.representation(data)
    if method == "les":
        data = model.output_modules[0](data)
    elif method != "pure":
        raise ValueError(method)
    energy = data[p.energy].reshape(-1)
    (force,) = torch.autograd.grad(
        -energy.sum(), data[p.positions], create_graph=create_graph
    )
    return energy, force, data.get("les_latent_charges")


def length_groups(samples, collate):
    """Keep the minibatch composition; group by atom count *inside* each optimizer update."""
    by_length = {}
    for sample in samples:
        by_length.setdefault(int(sample.atoms[p.n_atoms]), []).append(sample)
    return [collate(items) for items in by_length.values()]


def update(model, trainer, batches, method, phase="joint", freeze_backbone=False):
    """Physical loss weights each frame equally; ACVR stationarity weights each atom equally.

    For the LES/pure controls 'trainer' is an ordinary Adam optimizer. For an ACVR arm it
    is the normal ScratchTrainer. Optimizer steps and clipping happen once per original
    minibatch, irrespective of the number of distinct atom counts.
    """
    acvr = method == "acvr"
    if not acvr and phase != "joint":
        raise ValueError("Baseline updates only take E/F steps")
    natoms = [int(b[p.n_atoms].sum()) for b in batches]
    total_atoms, nframes = sum(natoms), sum(len(b[p.n_atoms]) for b in batches)
    model.train()
    device = next(model.parameters()).device
    if acvr:
        phys = trainer.energy if freeze_backbone else trainer.physical
        proposal = trainer.proposal
        adam, soap = trainer.physics_optimizer, trainer.solver_optimizer
    else:
        phys = list(model.representation.parameters())
        if method == "les":
            phys += list(model.output_modules[0].parameters())
        adam = trainer
    adam.zero_grad(set_to_none=True)
    if acvr:
        soap.zero_grad(set_to_none=True)
    loss_log = r_log = 0.0
    for b, atom_count in zip(batches, natoms):
        b = tree_to_device(b, device)
        frames = len(b[p.n_atoms])
        n = atom_count // frames
        if not bool((b[p.n_atoms] == n).all()):
            raise ValueError("Use length_groups() to split a ragged batch")
        if acvr:
            data = b.copy()
            for layer in model.input_modules:
                data = layer(data)
            data = model.representation(data)
            energy, q, dq = panel_energy_charge(
                model.output_modules[0], data, return_partial=True
            )
            if dq is None:
                dx, dq = torch.autograd.grad(
                    energy.sum(), (data[p.positions], q), create_graph=True
                )
                force = -dx
            else:
                (force,) = torch.autograd.grad(
                    -energy.sum(), data[p.positions], create_graph=True
                )
            r = (
                (dq - dq.mean(1, keepdim=True)).square().mean()
                * atom_count
                / total_atoms
            )
        else:
            energy, force, _ = physical_ef(model, b, method=method, create_graph=True)
        labs = b["variational"]["labels"]
        err_e = (
            ((energy - labs["condition_energy"]["values"].flatten()) / n)
            .square()
            .mean()
        )
        err_f = (force.flatten() - labs["condition_forces"]["values"]).square().mean()
        loss = 200.0 * (err_e + err_f) * frames / nframes
        if not bool(loss.isfinite()) or (acvr and not bool(r.isfinite())):
            raise FloatingPointError("Nonfinite E/F or stationarity loss")
        if phase != "proposal_warmup":
            torch.autograd.backward(
                loss, inputs=phys, retain_graph=acvr and phase == "joint"
            )
        if acvr and phase != "energy_warmup":
            torch.autograd.backward(30 * r, inputs=proposal)
        loss_log += float(loss.detach())
        if acvr:
            r_log += float(r.detach())
    if phase != "proposal_warmup":
        torch.nn.utils.clip_grad_norm_(phys, 10.0, error_if_nonfinite=True)
    if acvr and phase != "energy_warmup":
        torch.nn.utils.clip_grad_norm_(proposal, 10.0, error_if_nonfinite=True)
    if phase != "proposal_warmup":
        adam.step()
    if acvr and phase != "energy_warmup":
        soap.step()
    return dict(
        physical_loss=loss_log,
        stationarity_rms_V=r_log**0.5 if acvr else None,
        frames=nframes,
        atoms=total_atoms,
        length_groups=len(batches),
    )


def physical_optimizer(model, *, method, backbone_lr, head_lr):
    groups = [dict(params=list(model.representation.parameters()), lr=backbone_lr)]
    if method == "les":
        groups.append(
            dict(params=list(model.output_modules[0].parameters()), lr=head_lr)
        )
    return Adam(groups)
