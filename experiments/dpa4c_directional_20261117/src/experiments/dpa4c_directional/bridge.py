"""A single geometry-only transport call, sharing the physical branch's EFA factors."""
import importlib.util
import sys
from pathlib import Path
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cumulene_gate_runner", HERE.parent / "dpa4c_cumulene/run.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
p = old.p
sys.path.insert(0, str(HERE))
from transport import ARMS, DirectionalTransport


class GeometryCorrection(nn.Module):
    def __init__(self, arm, backbone, attention, seed):
        super().__init__()
        self.transport = DirectionalTransport(
            arm, tuple(backbone.degree_channels), tuple(backbone.moment_offsets), attention.grid, seed, channels=4
        )
        self.fuse = arm in ("B1H", "J2H")
        if self.fuse:
            # No additional random draw or trainable degrees of freedom at the initial output.
            self.fuse_weight = nn.ParameterList(
                [nn.Parameter(torch.zeros(4, w, device="cpu")) for w in backbone.degree_channels[:3]]
            )

    def forward(self, backbone, data, left, right):
        b, n = left.shape[:2]
        moments = data[p.node_embedding].reshape(b, n, -1)
        geom, messages = self.transport(moments, left, right)
        if not self.fuse:
            return geom

        # Modify *only* the local energy stream; the original moments/descriptor sent to
        # the ACVR head and proposal are unmodified. No second geometry pass through DPA.
        extra = [(msg @ weight).flatten(2) for msg, weight in zip(messages, self.fuse_weight)]
        end = backbone.moment_offsets[3]
        updated = torch.cat((moments[..., :end] + torch.cat(extra, -1), moments[..., end:]), -1)
        invariants, scalar, quartic = backbone._invariants(updated.flatten(0, 1))
        descriptor = data[p.node_feat]
        # Original layout: element, scalar moments, two local divisors, Gram/bispectrum/quartic.
        # Element and divisors are unchanged. Reuse those normalized slices without rounding.
        ne = backbone.element_embedding.weight.shape[-1]
        ns = scalar.shape[-1]
        first = ne + ns + 2
        tail = torch.cat((*invariants, quartic), -1)
        mean, scale = backbone.calibration_mean, backbone.calibration_inv_std
        descriptor_h = torch.cat((
            descriptor[:, :ne],
            (scalar - mean[ne:ne+ns]) * scale[ne:ne+ns],
            descriptor[:, ne+ns:first],
            (tail - mean[first:]) * scale[first:]
        ), -1)
        local = backbone._atomic_readout(descriptor_h, data[p.atomic_types])
        return geom + (local - data[p.atomic_energy].reshape_as(local)).reshape(b,n).sum(1)


def build(ckpt, arm, seed, dtype=torch.float32):
    model, cp, _, digest = old.build_model_from_gate80k(ckpt, "baseline", seed, dtype=torch.float64)
    if arm != "B0":
        e = model.output_modules[0].internal_energy
        e.directional_v = GeometryCorrection(arm, model.representation, e.attention, seed).cuda()
    model.to(dtype=dtype)
    fresh = [] if arm == "B0" else list(model.output_modules[0].internal_energy.directional_v.parameters())
    return model, cp, fresh, digest


def forward(model, batch, return_partial=True):
    for mod in model.input_modules:
        batch = mod(batch)
    d = model.representation(batch)
    head = model.output_modules[0]
    correction = getattr(head.internal_energy, "directional_v", None)
    fn = None if correction is None else lambda data, left, right: correction(model.representation, data, left, right)
    res = old.fixed_q_batch.panel_energy_charge(head, d, return_partial=return_partial, geometry_correction=fn)
    return res[0], res[1], res[2] if return_partial else None, d
