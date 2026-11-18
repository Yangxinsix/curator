"""Residual scalar and equivariant STF-frame readouts on the orient2 branch."""
import torch
from torch import nn

from curator.model.dpa4c import _packed_l2_to_stf
from tensor_model import TensorBranch


def _chebyshev_3_to_6(s):
    s2 = s * s
    t3 = s * (4.0 * s2 - 3.0)
    t4 = 8.0 * s2 * s2 - 8.0 * s2 + 1.0 - 1.0  # T_4(s) - T_4(0), where T_4(0) = 1
    t5 = s * (16.0 * s2 * s2 - 20.0 * s2 + 5.0)
    t6 = 32.0 * s2 * s2 * s2 - 48.0 * s2 * s2 + 18.0 * s2 - 1.0 + 1.0  # T_6(s) - T_6(0), where T_6(0) = -1
    return torch.stack((t3, t4, t5, t6), dim=-1)


class ReadoutBranch(TensorBranch):
    def orient_energy(self, raw, left, right):
        t = self.projections["2"](self.moments(raw, 2))
        x = t.flatten(-2)
        msg = left @ (right.mT @ x) - (left * right).sum(-1, keepdim=True) * x
        m = msg.reshape_as(t)
        z = (t * m).sum(-2)
        original = (z * self.readout["2"]).sum((-2, -1))
        if self.kind == "quadratic":
            # Unique B_ab: diagonal carries 1/2, off-diagonals appear twice in z^T B z / 2.
            terms = z.index_select(-1, self.rows) * z.index_select(-1, self.cols)
            residual = (terms * self.quad_weights * self.multiplicity).sum((-2, -1))
        elif self.kind == "chebyshev_z":
            terms = z.index_select(-1, self.rows) * z.index_select(-1, self.cols)
            quad = (terms * self.quad_weights * self.multiplicity).sum((-2, -1))
            cheb = (_chebyshev_3_to_6(torch.tanh(z)) * self.cheb_weights).sum((-3, -2, -1))
            residual = quad + cheb
        elif self.kind == "stf_frame":
            terms = z.index_select(-1, self.rows) * z.index_select(-1, self.cols)
            quad = (terms * self.quad_weights * self.multiplicity).sum((-2, -1))
            Q = _packed_l2_to_stf(t.transpose(-1, -2))
            G = _packed_l2_to_stf(m.transpose(-1, -2))
            eps = 1e-6
            t_norm = torch.sqrt((Q * Q).sum((-1, -2)) + eps * eps)
            m_norm = torch.sqrt((G * G).sum((-1, -2)) + eps * eps)
            amp = t_norm * (m_norm - eps)
            Q_hat = Q / t_norm.unsqueeze(-1).unsqueeze(-1)
            G_hat = G / m_norm.unsqueeze(-1).unsqueeze(-1)
            eye = torch.eye(3, dtype=Q.dtype, device=Q.device)
            Q2 = Q_hat @ Q_hat - (1.0 / 3.0) * eye
            u_norm = torch.sqrt((Q2 * Q2).sum((-1, -2), keepdim=True) + eps * eps)
            U_hat = Q2 / u_norm
            C_hat = Q_hat @ G_hat - G_hat @ Q_hat
            mu1 = (Q_hat * G_hat).sum((-1, -2))
            mu2 = (U_hat * G_hat).sum((-1, -2))
            # For unit STF tensors, max ||[Q_hat, G_hat]||_F^2 is 1/2, so s = (C_hat*C_hat).sum() in [0, 1].
            s = (C_hat * C_hat).sum((-1, -2))
            s2 = 2.0 * s * s
            s3 = 4.0 * s * s * s
            kap = 2.0 * (Q_hat * (C_hat @ C_hat)).sum((-1, -2))
            basis = amp.unsqueeze(-1) * torch.stack(
                (mu2, s, s * mu1, s * mu2, s2, s2 * mu1, s2 * mu2, s3, kap),
                dim=-1,
            )
            residual = quad + (basis * self.stf_weights).sum((-3, -2, -1))
        else:
            def features(z, m2):
                return torch.cat((t.square().sum(-2).log1p(), z.asinh(), m2.log1p()), dim=-1)
            m2 = m.square().sum(-2)
            # The residual vanishes if there is no remote message, even after training.
            residual = (self.net(features(z, m2)) -
                        self.net(features(torch.zeros_like(z), torch.zeros_like(m2)))).sum((-2, -1))
        return original + residual


def attach(model, kind, seed=3407):
    b = model.output_modules[0].internal_energy.tensor_branch
    if b.arm != "orient2" or not b.exclude_self or b.widths[2] != 4:
        raise ValueError("Readout continuation requires the four-channel, self-excluded orient2 parent")
    if kind == "linear":
        return []
    if kind not in ("quadratic", "mlp", "chebyshev_z", "stf_frame") or isinstance(b, ReadoutBranch):
        raise ValueError(f"Invalid readout installation: {kind}")
    ref = b.readout["2"]
    b.__class__ = ReadoutBranch
    b.kind = kind
    if kind in ("quadratic", "chebyshev_z", "stf_frame"):
        r, c = torch.triu_indices(4, 4, device=ref.device)
        b.register_buffer("rows", r, persistent=False)
        b.register_buffer("cols", c, persistent=False)
        b.register_buffer("multiplicity", torch.where(r == c, .5, 1.).to(ref), persistent=False)
        b.quad_weights = nn.Parameter(ref.new_zeros(10))
        if kind == "quadratic":
            return [b.quad_weights]
        if kind == "chebyshev_z":
            b.cheb_weights = nn.Parameter(ref.new_zeros(4, 4))
            return [b.quad_weights, b.cheb_weights]
        b.stf_weights = nn.Parameter(ref.new_zeros(4, 9))
        return [b.quad_weights, b.stf_weights]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        b.net = nn.Sequential(nn.Linear(12, 16), nn.SiLU(), nn.Linear(16, 1, bias=False))
        nn.init.zeros_(b.net[-1].weight)
    b.net.to(ref)
    return list(b.net.parameters())
