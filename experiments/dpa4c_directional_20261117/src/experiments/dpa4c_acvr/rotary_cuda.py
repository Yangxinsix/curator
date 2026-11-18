"""Fused ERoPE for both Q/K with deterministic first and second derivatives."""

import torch
import triton
import triton.language as tl


@triton.jit
def _expanded(
    X,
    COS,
    SIN,
    W,
    DX,
    DP,
    OUT,
    A: tl.constexpr,
    M: tl.constexpr,
    L: tl.constexpr,
    F: tl.constexpr,
    SECOND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = ids < 2 * A * M * L * F
    f = ids % F
    l = ids // F % L
    m = ids // (L * F) % M
    a = ids // (M * L * F) % A
    group = ids // (A * M * L * F)
    ix = ((group * A + a) * L + l) * F + f
    ph = (a * M + m) * F + f
    c = tl.load(COS + ph, mask, 0)
    s = tl.load(SIN + ph, mask, 0)
    w = tl.where(group == 0, tl.load(W + m, mask, 0), 1.0)
    x0 = tl.load(X + 2 * ix, mask, 0)
    x1 = tl.load(X + 2 * ix + 1, mask, 0)
    y0 = c * x0 - s * x1
    y1 = s * x0 + c * x1
    if SECOND:
        u0 = tl.load(DX + 2 * ix, mask, 0)
        u1 = tl.load(DX + 2 * ix + 1, mask, 0)
        v = tl.load(DP + ph, mask, 0)
        z0 = c * u0 - s * u1 - v * y1
        z1 = s * u0 + c * u1 + v * y0
    else:
        z0 = y0
        z1 = y1
    tl.store(OUT + 2 * ids, z0 * w, mask)
    tl.store(OUT + 2 * ids + 1, z1 * w, mask)


@triton.jit
def _reduce_directions(
    G,
    COS,
    SIN,
    W,
    DP,
    OUT,
    A: tl.constexpr,
    M: tl.constexpr,
    L: tl.constexpr,
    F: tl.constexpr,
    SECOND: tl.constexpr,
    BM: tl.constexpr,
    BF: tl.constexpr,
):
    group_a_l = tl.program_id(0)
    a = group_a_l // L % A
    group = group_a_l // (L * A)
    l = group_a_l % L
    m = tl.arange(0, BM)[:, None]
    f = tl.program_id(1) * BF + tl.arange(0, BF)[None, :]
    mask = (m < M) & (f < F)
    ph = (a * M + m) * F + f
    out_idx = (((group * A + a) * M + m) * L + l) * F + f
    c = tl.load(COS + ph, mask, 0)
    s = tl.load(SIN + ph, mask, 0)
    g0 = tl.load(G + 2 * out_idx, mask, 0)
    g1 = tl.load(G + 2 * out_idx + 1, mask, 0)
    w = tl.where(group == 0, tl.load(W + m, m < M, 0), 1.0)
    h0 = c * g0 + s * g1
    h1 = c * g1 - s * g0
    if SECOND:
        v = tl.load(DP + ph, mask, 0)
        z0 = v * h1
        z1 = -v * h0
    else:
        z0 = h0
        z1 = h1
    col = tl.program_id(1) * BF + tl.arange(0, BF)
    ix = group_a_l * F + col
    tl.store(OUT + 2 * ix, tl.sum(w * z0, 0), col < F)
    tl.store(OUT + 2 * ix + 1, tl.sum(w * z1, 0), col < F)


@triton.jit
def _reduce_channels(
    X,
    G,
    COS,
    SIN,
    W,
    DX,
    DP,
    OUT,
    A: tl.constexpr,
    M: tl.constexpr,
    L: tl.constexpr,
    F: tl.constexpr,
    SECOND: tl.constexpr,
    BH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    h = tl.arange(0, BH)[:, None]
    ph = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)[None, :]
    a = ph // (M * F)
    m = ph // F % M
    f = ph % F
    group = h // L
    l = h % L
    valid = (ph < A * M * F) & (h < 2 * L)
    gi = (((group * A + a) * M + m) * L + l) * F + f
    xi = ((group * A + a) * L + l) * F + f
    c = tl.load(COS + ph, ph < A * M * F, 0)
    s = tl.load(SIN + ph, ph < A * M * F, 0)
    w = tl.where(group == 0, tl.load(W + m, ph < A * M * F, 0), 1.0)
    g0 = tl.load(G + 2 * gi, valid, 0)
    g1 = tl.load(G + 2 * gi + 1, valid, 0)
    x0 = tl.load(X + 2 * xi, valid, 0)
    x1 = tl.load(X + 2 * xi + 1, valid, 0)
    h0 = c * g0 + s * g1
    h1 = c * g1 - s * g0
    if SECOND:
        u0 = tl.load(DX + 2 * xi, valid, 0)
        u1 = tl.load(DX + 2 * xi + 1, valid, 0)
        v = tl.load(DP + ph, ph < A * M * F, 0)
        value = u0 * h1 - u1 * h0 - v * (x0 * h0 + x1 * h1)
    else:
        value = x0 * h1 - x1 * h0
    target = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + target, tl.sum(w * value, 0), target < A * M * F)


def expanded(x, c, s, w, u=None, v=None):
    _, a, l, k = x.shape
    m, f = c.shape[1:]
    y = x.new_empty(2, a, m, l, k)
    with torch.cuda.device(x.device):
        _expanded[(triton.cdiv(y.numel() // 2, 256),)](
            x, c, s, w, u, v, y, a, m, l, f, u is not None, 256, enable_fp_fusion=False
        )
    return y


def reductions(x, g, c, s, w, u=None, v=None):
    _, a, l, k = x.shape
    m, f = c.shape[1:]
    gx = torch.empty_like(x)
    gp = torch.empty_like(c)
    with torch.cuda.device(x.device):
        _reduce_directions[(2 * a * l, triton.cdiv(f, 16))](
            g,
            c,
            s,
            w,
            v,
            gx,
            a,
            m,
            l,
            f,
            u is not None,
            triton.next_power_of_2(m),
            16,
            enable_fp_fusion=False,
        )
        _reduce_channels[(triton.cdiv(c.numel(), 64),)](
            x,
            g,
            c,
            s,
            w,
            u,
            v,
            gp,
            a,
            m,
            l,
            f,
            u is not None,
            triton.next_power_of_2(2 * l),
            64,
            enable_fp_fusion=False,
        )
    return gx, gp


class _RotaryVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, phase, w, g, c, s):
        ctx.save_for_backward(x, phase, w, g, c, s)
        return reductions(x, g, c, s, w)

    @staticmethod
    def backward(ctx, u, v):
        x, phase, w, g, c, s = ctx.saved_tensors
        u, v = u.contiguous(), v.contiguous()
        if torch.is_grad_enabled():
            # Preserve the graph if diagnostics request derivatives beyond order two.
            with torch.enable_grad():

                def reference(x, ph):
                    vals = x.reshape(*x.shape[:-1], -1, 2).unsqueeze(2)
                    cph, sph = (
                        ph.cos()[None, :, :, None, :],
                        ph.sin()[None, :, :, None, :],
                    )
                    x0, x1 = vals.unbind(-1)
                    y = torch.stack(
                        (cph * x0 - sph * x1, sph * x0 + cph * x1), -1
                    ).flatten(-2)
                    return torch.stack((y[0] * w[None, :, None, None], y[1]))

                _, tangent = torch.func.jvp(reference, (x, phase), (u, v))
                # torch.func.vjp differentiates only the indicated positional inputs.
                _, pullback = torch.func.vjp(
                    lambda xx, pp: torch.func.jvp(reference, (xx, pp), (u, v))[1],
                    x,
                    phase,
                )
                gx, gp = pullback(g)
                return gx, gp, None, tangent, None, None
        gx, gp = reductions(x, g, c, s, w, u, v)
        return gx, gp, None, expanded(x, c, s, w, u, v), None, None


class _Rotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, phase, w):
        c, s = phase.cos(), phase.sin()
        ctx.save_for_backward(x, phase, w, c, s)
        return expanded(x, c, s, w)

    @staticmethod
    def backward(ctx, grad):
        x, phase, w, c, s = ctx.saved_tensors
        gx, gp = _RotaryVJP.apply(x, phase, w, grad.contiguous(), c, s)
        return gx, gp, None


def rotated_factors(q, k, phase, weights):
    """Return separate weighted Q, unweighted K as [atoms, directions, channels]."""
    x = torch.stack((q, k)).contiguous()
    rotated = _Rotary.apply(x, phase.contiguous(), weights.contiguous()).flatten(-2)
    return rotated.unbind(0)
