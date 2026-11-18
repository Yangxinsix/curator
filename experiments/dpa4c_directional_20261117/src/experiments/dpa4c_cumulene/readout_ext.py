"""Cumulene-focused geometric readout extensions on top of the gate80k checkpoint.

Every variant attaches cleanly as a zero-initialized residual over the restored
80k gate ReadoutBranch so step-0 energy, forces, and charges match the parent
checkpoint to numerical precision, and remote-zero baselines are subtracted
explicitly by evaluating with zeroed remote messages.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from curator.model.dpa4c import _packed_l2_to_stf
from readout import ReadoutBranch


def _stf_invariants_rich(t2, m2, eps_amp=1e-6):
    """Extract smooth STF invariants including regularized square-root commutator terms.

    t2, m2: [B, N, 5, C] (here C=4).
    Returns:
        amp: [B, N, C] smooth remote amplitude (0 when m2 == 0).
        inv: [B, N, C, 20] unit-scale angular/relative-frame invariants.
    """
    Q = _packed_l2_to_stf(t2.transpose(-1, -2))  # [B, N, C, 3, 3]
    G = _packed_l2_to_stf(m2.transpose(-1, -2))  # [B, N, C, 3, 3]
    tn = torch.sqrt((Q * Q).sum((-1, -2)) + eps_amp * eps_amp)
    mn = torch.sqrt((G * G).sum((-1, -2)) + eps_amp * eps_amp)
    amp = tn * (mn - eps_amp)

    Qh = Q / tn.unsqueeze(-1).unsqueeze(-1)
    Gh = G / mn.unsqueeze(-1).unsqueeze(-1)
    eye = torch.eye(3, dtype=Q.dtype, device=Q.device)
    Q2 = Qh @ Qh - (1.0 / 3.0) * eye
    Uh = Q2 / torch.sqrt((Q2 * Q2).sum((-1, -2), keepdim=True) + eps_amp * eps_amp)
    Ch = Qh @ Gh - Gh @ Qh

    mu1 = (Qh * Gh).sum((-1, -2))
    mu2 = (Uh * Gh).sum((-1, -2))
    s = (Ch * Ch).sum((-1, -2))
    kap = 2.0 * (Qh * (Ch @ Ch)).sum((-1, -2))

    # Smooth square-root invariants capture high-slope behavior as s -> 0 near 180 deg
    sq_s_1e2 = torch.sqrt(s + 1.0e-2) - math.sqrt(1.0e-2)
    sq_s_1e3 = torch.sqrt(s + 1.0e-3) - math.sqrt(1.0e-3)
    sq_s_1e4 = torch.sqrt(s + 1.0e-4) - math.sqrt(1.0e-4)
    sq_mp = torch.sqrt((1.0 + mu1).clamp_min(0.0) + 1.0e-3) - math.sqrt(1.0e-3)
    sq_mm = torch.sqrt((1.0 - mu1).clamp_min(0.0) + 1.0e-3) - math.sqrt(1.0e-3)
    # Cross-channel Q-G contractions [B, N, 4, 4]
    cross_qg = torch.einsum("bncij,bndij->bncd", Qh, Gh)

    inv = torch.cat(
        (
            torch.stack(
                (
                    mu1,
                    mu2,
                    s,
                    s * mu1,
                    s * mu2,
                    2.0 * s * s,
                    2.0 * s * s * mu1,
                    2.0 * s * s * mu2,
                    4.0 * s * s * s,
                    kap,
                    sq_s_1e2,
                    sq_s_1e3,
                    sq_s_1e4,
                    sq_mp,
                    sq_mm,
                    s * sq_s_1e3,
                ),
                dim=-1,
            ),
            cross_qg,
        ),
        dim=-1,
    )  # [B, N, 4, 20]
    return amp, inv


class RichSTFBranch(ReadoutBranch):
    """Extends the restored gate ReadoutBranch with a zero-initialized rich STF + l1 residual.

    Supports ablation via `use_root_invariants`:
      - rich_stf: full 20 invariants including regularized sqrt(s), sqrt(1+-mu1)
      - smooth_stf: identical architecture except root invariants are zeroed out
    """

    def _compute_features(self, t2, m2, t1, m1, local):
        amp2, inv2 = _stf_invariants_rich(t2, m2)
        if not getattr(self, "use_root_invariants", True):
            # Zero out the 6 root-based slots (indices 10..15) for controlled ablation
            mask = torch.ones_like(inv2)
            mask[..., 10:16] = 0.0
            inv2 = inv2 * mask

        v1 = t1.transpose(-1, -2)   # [B, N, 8, 3]
        mv1 = m1.transpose(-1, -2)  # [B, N, 8, 3]
        dot1 = (v1 * mv1).sum(-1)
        cross1 = torch.linalg.cross(v1, mv1, dim=-1).square().sum(-1)
        m1_norm = mv1.square().sum(-1)

        G = _packed_l2_to_stf(m2.transpose(-1, -2))  # [B, N, 4, 3, 3]
        Q = _packed_l2_to_stf(t2.transpose(-1, -2))  # [B, N, 4, 3, 3]
        v1_4 = v1[..., :4, :]
        mv1_4 = mv1[..., :4, :]
        vgv = torch.einsum("bnci,bncij,bncj->bnc", v1_4, G, v1_4)
        mqm = torch.einsum("bnci,bncij,bncj->bnc", mv1_4, Q, mv1_4)

        feats = torch.cat(
            (
                local,
                (amp2.unsqueeze(-1) * inv2).flatten(-2),  # 80
                inv2.flatten(-2),                           # 80
                amp2,                                       # 4
                dot1,                                       # 8
                cross1,                                     # 8
                m1_norm,                                    # 8
                vgv,                                        # 4
                mqm,                                        # 4
            ),
            dim=-1,
        )  # 228
        return amp2, inv2, feats

    def orient_energy(self, raw, left, right, *, normalized_features=None):
        base_energy = super().orient_energy(
            raw, left, right, normalized_features=normalized_features
        )
        if normalized_features is None:
            raise ValueError("RichSTFBranch requires normalized_features")

        t2 = self.projections["2"](self.moments(raw, 2))  # [B, N, 5, 4]
        t1 = self.l1_proj(self.moments(raw, 1))           # [B, N, 3, 8]
        x12 = torch.cat((t1.flatten(-2), t2.flatten(-2)), dim=-1)
        remote = left @ (right.mT @ x12) - (left * right).sum(-1, keepdim=True) * x12
        m1_flat, m2_flat = remote.split((t1.shape[-2] * t1.shape[-1], t2.shape[-2] * t2.shape[-1]), dim=-1)
        m1 = m1_flat.reshape_as(t1)
        m2 = m2_flat.reshape_as(t2)

        local = self.local_proj(normalized_features.index_select(-1, self.chemical_gate.scalar_indices))  # [B, N, 32]
        amp2, inv2, feats_live = self._compute_features(t2, m2, t1, m1, local)
        # Exact remote-zero baseline by evaluating with m1 = 0, m2 = 0 (not all-zero inv2!)
        _, _, feats_zero = self._compute_features(
            t2, torch.zeros_like(m2), t1, torch.zeros_like(m1), local
        )

        # Explicit linear term in amp2*inv2 (already 0 when m2=0 because amp2=0) + residual MLP
        lin_term = (amp2.unsqueeze(-1) * inv2 * self.rich_linear).sum((-3, -2, -1))
        mlp_term = (self.rich_mlp(feats_live) - self.rich_mlp(feats_zero)).sum((-2, -1))
        return base_energy + lin_term + mlp_term


def attach_cumulene_arm(model, arm, seed=3407):
    """Attach Cumulene-specific zero-initialized residual arm onto an 80k gate model."""
    if arm == "baseline":
        return []
    if arm not in ("rich_stf", "smooth_stf"):
        raise ValueError(f"Unknown cumulene arm: {arm}")
    energy = model.output_modules[0].internal_energy
    tb = energy.tensor_branch
    if not isinstance(tb, ReadoutBranch) or not hasattr(tb, "chemical_gate"):
        raise ValueError("Expected restored 80k gate ReadoutBranch with chemical_gate")
    ref = tb.stf_weights
    scalar_dim = len(energy.feature_norm.scalar_indices)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        tb.l1_proj = nn.Linear(tb.widths[1], tb.widths[1], bias=False).to(ref)
        nn.init.orthogonal_(tb.l1_proj.weight)
        tb.local_proj = nn.Linear(scalar_dim, 32).to(ref)
        tb.rich_linear = nn.Parameter(ref.new_zeros(4, 20))
        tb.rich_mlp = nn.Sequential(
            nn.Linear(228, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1, bias=False),
        ).to(ref)
        nn.init.zeros_(tb.rich_mlp[-1].weight)

    params = [
        *tb.l1_proj.parameters(),
        *tb.local_proj.parameters(),
        tb.rich_linear,
        *tb.rich_mlp.parameters(),
    ]
    tb.__class__ = RichSTFBranch
    tb.use_root_invariants = (arm == "rich_stf")
    return params
