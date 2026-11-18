"""Opt-in RSA conversion for RMSNorm ACVR. Serialized model_options select the kernel."""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scratch as periodic
from cache import output_head
from curator.layer.efa.reciprocal_rsa import PeriodicRSA


def install_rsa(model, *, periodic_cutoff=0.86, periodic_reference_volume=45.,
                periodic_charge_condition="density", rsa_groups=0, rsa_sigma=6 / math.sqrt(-2 * math.log(1e-3))):
    if periodic_charge_condition not in ("density", "total"):
        raise ValueError("periodic_charge_condition must be density or total")
    head = output_head(model)
    head.internal_energy.attention = PeriodicRSA.from_attention(
        head.internal_energy.attention, periodic_cutoff=periodic_cutoff,
        periodic_reference_volume=periodic_reference_volume, rsa_sigma=rsa_sigma)
    if rsa_groups:
        from curator.layer.efa.reciprocal_rsa_channels import ChannelSpectrumRSA
        head.internal_energy.attention = ChannelSpectrumRSA.from_rsa(head.internal_energy.attention, rsa_groups)
    head.proposal.periodic_charge_condition = periodic_charge_condition
    head.proposal.periodic_reference_volume = periodic_reference_volume
    return model


def build_scratch_model(*, periodic_backend="rsa", **options):
    if periodic_backend != "rsa":
        options.pop("rsa_sigma", None)
        options.pop("rsa_groups", None)
        return periodic.build_scratch_model(periodic_backend=periodic_backend, **options)
    keys = ("periodic_cutoff", "periodic_reference_volume", "periodic_charge_condition", "rsa_sigma", "rsa_groups")
    kernel = {key: options.pop(key) for key in keys if key in options}
    return install_rsa(periodic.build_scratch_model(periodic_backend="legacy", **options), **kernel)


def load_checkpoint(path, device="cpu", *, periodic_backend=None, **overrides):
    state = torch.load(path, map_location="cpu", weights_only=False)
    options = dict(state["model_options"])
    if "periodic_backend" in options:
        options.setdefault("periodic_charge_condition", "total")
    options.setdefault("periodic_backend", "legacy")
    if periodic_backend is not None:
        options["periodic_backend"] = periodic_backend
    options.update(overrides)
    model = build_scratch_model(**options)
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()
