"""Reproducible Cumulene-only optimization & ablation runner starting from gate80k.

Strictly obeys ACVR split-gradient policy:
  - physical loss (E/F, weights 200/200) updates physical parameters ONLY
    (backbone + internal_energy + tensor_branch + new readout parameters).
  - stationarity residual (weight 30) updates proposal parameters ONLY.
  - never trains on profile/scan labels or charge labels.
  - supports both local RTX PRO 4000 (reference backend) and cluster GPU (native or reference).
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [
    str(ROOT / "experiments" / n)
    for n in (
        "dpa4c_acvr",
        "dpa4c_gemini",
        "dpa4c_tensor",
        "dpa4c_readout",
        "dpa4c_unified",
        "dpa4c_transfer",
        "periodic_qefa/scripts",
    )
]
sys.path.insert(0, str(HERE))

import numpy as np
import torch

# Load unified runner explicitly first so periodic_qefa/curator_source is on sys.path
_spec = importlib.util.spec_from_file_location("unified_run", ROOT / "experiments/dpa4c_unified/run.py")
urun = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(urun)

from curator.data import properties as p
from curator.layer.efa.core import EuclideanFastAttention
from native_spline import NativeSplineLayer
from operator_kan_architectures import OperatorSplineLayer
from soap import SOAP
import fixed_q_batch
import scan
from readout_ext import attach_cumulene_arm

ARMS = ("baseline", "rich_stf", "smooth_stf")
DEFAULT_NPZ = Path("/mnt/c/Users/Administrator/AppData/Local/Temp/acvr-longrange-data/efa_cumulene.npz")
DEFAULT_PROFILE = Path("/mnt/c/Users/Administrator/AppData/Local/Temp/acvr-longrange-data/efa_cumulene_profile.npz")
DEFAULT_CKPT = Path("/tmp/cumulene_80k/gate_ckpt-020000.pt")
DEFAULT_SPLIT = Path("/tmp/cumulene_80k/split.json")
DEFAULT_EREF = Path("/tmp/cumulene_80k/energy_reference.json")
EXPECTED_SPLIT_SHAS = {
    "486f1e01ece07cd2dbd684b7db70ddfc7f499c461f1c7e147e587fd544a09acd",
    "8979beb238a8ef35bf18cc5f5ef941ad71d567ebadc44a208b9312e2e5f758be",
}
EXPECTED_EREF_SPLIT_SHA = "486f1e01ece07cd2dbd684b7db70ddfc7f499c461f1c7e147e587fd544a09acd"
EXPECTED_NPZ_SHA = "5ee897677cdcba26dfe6eee6794633506383be513d92aff4a8584b8950c5a111"
EXPECTED_PROFILE_SHA = "08e4b224428b2178e788ac5741d3c8bab0f2a25da8af22648eac20b083215f26"
EXPECTED_CKPT_SHA = "fd3ec04f4f2cb8d7c15e64ce3413d77f5897b9de911fc662ab3761e8ae4e7de6"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clone_state_cpu(state_dict):
    out = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
        else:
            out[k] = copy.deepcopy(v)
    return out


def atomic_torch_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def atomic_write_text(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def configure_deterministic():
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def geom_angle_tangent(z, positions):
    """Geometry-only terminal CH2 angle and right-CH2 tangent (A/rad), without hard 1.3A cutoff."""
    c = np.flatnonzero(z == 6)
    h = np.flatnonzero(z == 1)
    if len(z) != 13 or len(c) != 9 or len(h) != 4:
        raise ValueError("Expected 13-atom C9H4 cumulene")
    cc_dist = np.linalg.norm(positions[c, None] - positions[c][None], axis=2)
    i, j = np.unravel_index(np.argmax(cc_dist), cc_dist.shape)
    left, right = sorted((int(c[i]), int(c[j])))
    d_left = np.linalg.norm(positions[h] - positions[left], axis=1)
    d_right = np.linalg.norm(positions[h] - positions[right], axis=1)
    assign_right = d_right < d_left
    hleft = h[~assign_right]
    hright = h[assign_right]
    if len(hleft) != 2 or len(hright) != 2:
        raise ValueError("Could not assign two terminal H to each end C")
    axis = positions[right] - positions[left]
    axis /= np.linalg.norm(axis)
    l = positions[hleft[1]] - positions[hleft[0]]
    r = positions[hright[0]] - positions[hright[1]]
    l -= (l @ axis) * axis
    r -= (r @ axis) * axis
    ang = float(np.degrees(np.arctan2(np.cross(l, r) @ axis, l @ r)) % 360.0)
    tg = np.zeros_like(positions, dtype=np.float64)
    tg[hright] = np.cross(axis, positions[hright] - positions[right])
    return ang, tg


def load_cumulene_data(npz_path: Path, split_path: Path, eref_path: Path, profile_path: Path):
    split_sha = sha256(split_path)
    if split_sha not in EXPECTED_SPLIT_SHAS:
        raise ValueError(f"Unexpected split SHA256 for {split_path}: {split_sha}")
    npz_sha = sha256(npz_path)
    if npz_sha != EXPECTED_NPZ_SHA:
        raise ValueError(f"Unexpected NPZ SHA256 for {npz_path}: {npz_sha}")
    prof_sha = sha256(profile_path)
    if prof_sha != EXPECTED_PROFILE_SHA:
        raise ValueError(f"Unexpected profile SHA256 for {profile_path}: {prof_sha}")

    split = json.loads(split_path.read_text())
    tr = np.array(split["train"], dtype=int)
    va = np.array(split["validation"], dtype=int)
    te = np.array(split["test"], dtype=int)
    if len(tr) != 2000 or len(va) != 500 or len(te) != 2473:
        raise ValueError("Unexpected split sizes")
    if set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te):
        raise ValueError("Train/validation/test splits are not disjoint")

    eref_raw = json.loads(eref_path.read_text())
    if eref_raw["fit_split"] != "train_only" or eref_raw["split_sha256"] != EXPECTED_EREF_SPLIT_SHA:
        raise ValueError("Energy reference must match train_only split")
    eref = {int(k): float(v) for k, v in eref_raw["per_element_eV"].items()}

    npz = np.load(npz_path)
    z_all = npz["atomic_numbers"]
    pos_all = npz["positions"]
    e_raw = npz["energy"][:, 0].astype(np.float64)
    comp = sum((z_all == elem).sum(axis=1) * val for elem, val in eref.items())
    e_int = e_raw - comp  # Subtract composition reference in float64
    f_all = npz["forces"].astype(np.float64)

    prof = np.load(profile_path)
    return dict(
        z=z_all,
        pos=pos_all,
        e_int=e_int,
        f=f_all,
        split=split,
        eref=eref,
        split_sha=split_sha,
        npz_sha=npz_sha,
        profile_sha=prof_sha,
        profile=dict(
            z=prof["atomic_numbers"],
            pos=prof["positions"],
            e=prof["energy"].astype(np.float64),
            f=prof["forces"].astype(np.float64),
        ),
    )


def install_reference_backend(model):
    model.representation.inference_backend = "reference"
    for mod in model.modules():
        if isinstance(mod, NativeSplineLayer):
            mod.__class__ = OperatorSplineLayer
    model.output_modules[0].internal_energy.attention.__class__ = EuclideanFastAttention
    return model


def build_model_from_gate80k(ckpt_path: Path, arm: str, seed: int, dtype=torch.float32, use_reference=True):
    ckpt_sha = sha256(ckpt_path)
    if ckpt_sha != EXPECTED_CKPT_SHA:
        raise ValueError(f"Unexpected gate80k checkpoint SHA256: {ckpt_sha}")
    model, cp = urun.restore(ckpt_path)
    if cp["identity"]["system"] != "cumulene" or cp["identity"]["arm"] != "gate" or cp["joint_step"] != 80000:
        raise ValueError("Expected Cumulene 80k gate checkpoint")
    if use_reference:
        install_reference_backend(model)
    fresh = attach_cumulene_arm(model, arm, seed=seed)
    model = model.cuda().to(dtype=dtype)
    return model, cp, fresh, ckpt_sha


def make_batch(z_np, pos_np, device="cuda", dtype=torch.float32, requires_grad=True):
    B, N = pos_np.shape[:2]
    z_t = torch.as_tensor(z_np, device=device, dtype=torch.long)
    pos_t = torch.as_tensor(pos_np, device=device, dtype=dtype)
    if requires_grad:
        pos_t = pos_t.detach().clone().requires_grad_(True)
    return {
        p.atomic_numbers: z_t.reshape(-1),
        p.positions: pos_t.reshape(-1, 3),
        p.image_idx: torch.arange(B, device=device).repeat_interleave(N),
        p.n_atoms: torch.full((B,), N, device=device, dtype=torch.long),
        p.cell: torch.zeros(B, 3, 3, device=device, dtype=dtype),
        p.pbc: torch.zeros(B, 3, device=device, dtype=torch.bool),
        p.total_charge: torch.zeros(B, device=device, dtype=dtype),
        "variational": {
            "condition_geometry_index": torch.arange(B, device=device),
            "condition_state_ptr": torch.arange(B + 1, device=device) * N,
            "boundary_spec": scan.boundaries()["boundary_spec"] * B,
        },
    }


def forward_energy_charge(model, batch_dict, return_partial=True):
    head = model.output_modules[0]
    d = batch_dict
    for mod in model.input_modules:
        d = mod(d)
    d = model.representation(d)
    res = fixed_q_batch.panel_energy_charge(head, d, return_partial=return_partial)
    U, q, dq = res if return_partial else (res[0], res[1], None)
    return U, q, dq, d


def eval_split(model, z_np, pos_np, e_int_np, f_np, batch_size=64):
    dtype = next(model.parameters()).dtype
    U_list, F_list, res_list, q_list = [], [], [], []
    model.eval()
    for s in range(0, len(pos_np), batch_size):
        zb = z_np[s:s + batch_size]
        pb = pos_np[s:s + batch_size]
        B, N = pb.shape[:2]
        batch = make_batch(zb, pb, dtype=dtype, requires_grad=True)
        U, q, dq, d = forward_energy_charge(model, batch, return_partial=True)
        if dq is None:
            dR, dq = torch.autograd.grad(U.sum(), (d[p.positions], q))
        else:
            (dR,) = torch.autograd.grad(U.sum(), d[p.positions])
        resid = dq - dq.mean(1, keepdim=True)
        U_list.append(U.detach().cpu().double().numpy())
        F_list.append((-dR).detach().cpu().double().numpy().reshape(B, N, 3))
        res_list.append(resid.detach().cpu().double().numpy().reshape(B, N))
        q_list.append(q.detach().cpu().double().numpy().reshape(B, N))
    Up = np.concatenate(U_list)
    Fp = np.concatenate(F_list)
    resp = np.concatenate(res_list)
    qp = np.concatenate(q_list)
    e_rmse = float(np.sqrt(np.mean(((Up - e_int_np) / 13.0) ** 2)) * 1000.0)
    f_rmse = float(np.sqrt(np.mean((Fp - f_np) ** 2)) * 1000.0)
    c_mask = z_np[0] == 6
    h_mask = z_np[0] == 1
    c_rmse = float(np.sqrt(np.mean((Fp[:, c_mask] - f_np[:, c_mask]) ** 2)) * 1000.0)
    h_rmse = float(np.sqrt(np.mean((Fp[:, h_mask] - f_np[:, h_mask]) ** 2)) * 1000.0)
    rms_v = float(np.sqrt(np.mean(resp ** 2)))

    # Geometry-only projected torque metrics on general training/validation/test frames
    angs, t_p, t_t = [], [], []
    for k in range(len(pos_np)):
        ang, tg = geom_angle_tangent(z_np[k], pos_np[k].astype(np.float64))
        angs.append(ang)
        t_p.append(np.sum(Fp[k] * tg))
        t_t.append(np.sum(f_np[k] * tg))
    angs = np.array(angs)
    t_p = np.array(t_p)
    t_t = np.array(t_t)
    fold = np.where(angs > 180.0, 360.0 - angs, angs)
    torque_overall = float(np.sqrt(np.mean((t_p - t_t) ** 2)))
    m150 = fold >= 150.0
    m170 = fold >= 170.0
    m175 = fold >= 175.0
    return dict(
        frames=int(len(pos_np)),
        energy_RMSE_meV_atom=e_rmse,
        force_RMSE_meV_A=f_rmse,
        carbon_force_RMSE_meV_A=c_rmse,
        hydrogen_force_RMSE_meV_A=h_rmse,
        torque_RMSE_eV_rad=torque_overall,
        torque_150_180_RMSE_eV_rad=float(np.sqrt(np.mean((t_p[m150] - t_t[m150]) ** 2))) if m150.any() else None,
        torque_170_180_RMSE_eV_rad=float(np.sqrt(np.mean((t_p[m170] - t_t[m170]) ** 2))) if m170.any() else None,
        torque_175_180_RMSE_eV_rad=float(np.sqrt(np.mean((t_p[m175] - t_t[m175]) ** 2))) if m175.any() else None,
        stationarity_RMS_V=rms_v,
        q_transfer_RMS_e=float(np.sqrt(np.mean((qp - qp.mean(1, keepdims=True)) ** 2))),
    )


def eval_profile(model, prof_data, batch_size=64, save_dir: Path | None = None, tag: str = ""):
    dtype = next(model.parameters()).dtype
    zp, pp, ep, fp = prof_data["z"], prof_data["pos"], prof_data["e"], prof_data["f"]
    U_list, F_list = [], []
    model.eval()
    for s in range(0, len(pp), batch_size):
        zb = zp[s:s + batch_size]
        pb = pp[s:s + batch_size]
        B, N = pb.shape[:2]
        batch = make_batch(zb, pb, dtype=dtype, requires_grad=True)
        U, _, _, d = forward_energy_charge(model, batch, return_partial=False)
        (dR,) = torch.autograd.grad(U.sum(), d[p.positions])
        U_list.append(U.detach().cpu().double().numpy())
        F_list.append((-dR).detach().cpu().double().numpy().reshape(B, N, 3))
    Up = np.concatenate(U_list)
    Fp = np.concatenate(F_list)

    angles = np.zeros(len(pp))
    tangents = np.zeros_like(pp, dtype=np.float64)
    for i in range(len(pp)):
        ang, tg = scan.angle_tangent(zp[i], pp[i].astype(np.float64))
        angles[i] = ang
        tangents[i] = tg

    t_pred = np.sum(Fp * tangents, axis=(1, 2))
    t_dft = np.sum(fp * tangents, axis=(1, 2))
    rel_pred = Up - Up[0]
    rel_dft = ep - ep[0]

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        csv_path = save_dir / f"profile_{tag}.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "angle_deg", "rel_dft_eV", "rel_pred_eV", "torque_dft_eV_rad", "torque_pred_eV_rad"])
            for idx in range(len(pp)):
                w.writerow([idx, float(angles[idx]), float(rel_dft[idx]), float(rel_pred[idx]), float(t_dft[idx]), float(t_pred[idx])])

    def bin_rmse(lo, hi, inclusive_hi=False):
        m = (angles >= lo) & (angles <= hi if inclusive_hi else angles < hi)
        return dict(n=int(m.sum()), torque_RMSE_eV_rad=float(np.sqrt(np.mean((t_pred[m] - t_dft[m]) ** 2))))

    return dict(
        frames=int(len(pp)),
        barrier_pred_eV=float(rel_pred[-1]),
        barrier_DFT_eV=float(rel_dft[-1]),
        barrier_error_eV=float(rel_pred[-1] - rel_dft[-1]),
        relative_energy_RMSE_eV=float(np.sqrt(np.mean((rel_pred - rel_dft) ** 2))),
        cartesian_force_RMSE_meV_A=float(np.sqrt(np.mean((Fp - fp) ** 2)) * 1000.0),
        full_torque_RMSE_eV_rad=float(np.sqrt(np.mean((t_pred - t_dft) ** 2))),
        high_angle_150_175=bin_rmse(150.0, 175.0, inclusive_hi=True),
        endpoint_175_180=bin_rmse(175.0, 180.1, inclusive_hi=True),
        bins={
            "150_155": bin_rmse(150.0, 155.0),
            "155_160": bin_rmse(155.0, 160.0),
            "160_165": bin_rmse(160.0, 165.0),
            "165_170": bin_rmse(165.0, 170.0),
            "170_175": bin_rmse(170.0, 175.0, inclusive_hi=True),
            "175_180": bin_rmse(175.0, 180.1, inclusive_hi=True),
        },
        torque_at_174_95_pred=float(t_pred[(angles >= 150.0) & (angles <= 175.0)][-1]),
        torque_at_174_95_DFT=float(t_dft[(angles >= 150.0) & (angles <= 175.0)][-1]),
    )


def preflight(args):
    data = load_cumulene_data(args.npz, args.split, args.eref, args.profile)
    val_idx = np.array(data["split"]["validation"][:8], dtype=int)
    zb = data["z"][val_idx]
    pb = data["pos"][val_idx].astype(np.float64)
    fb = torch.as_tensor(data["f"][val_idx], device="cuda", dtype=torch.float64)

    # 1. Parity at step 0 in fp64 against baseline gate80k
    ref_model, _, _, _ = build_model_from_gate80k(args.ckpt, "baseline", args.seed, dtype=torch.float64, use_reference=True)
    ref_model.eval()
    ref_b = make_batch(zb, pb, dtype=torch.float64, requires_grad=True)
    U_ref, q_ref, _, d_ref = forward_energy_charge(ref_model, ref_b, return_partial=False)
    (dR_ref,) = torch.autograd.grad(U_ref.sum(), d_ref[p.positions])
    F_ref = -dR_ref.reshape(8, 13, 3)

    report = {}
    for arm in ARMS:
        m, _, fresh, ckpt_sha = build_model_from_gate80k(args.ckpt, arm, args.seed, dtype=torch.float64, use_reference=True)
        m.eval()
        b = make_batch(zb, pb, dtype=torch.float64, requires_grad=True)
        U, q, dq, d = forward_energy_charge(m, b, return_partial=True)
        if dq is None:
            dR, dq = torch.autograd.grad(U.sum(), (d[p.positions], q), create_graph=True)
        else:
            (dR,) = torch.autograd.grad(U.sum(), d[p.positions], create_graph=True)
        F_pred = -dR.reshape(8, 13, 3)

        max_dU = float((U - U_ref).abs().max().item())
        max_dF = float((F_pred - F_ref).abs().max().item())
        max_dq = float((q - q_ref).abs().max().item())
        assert max_dU < 1e-12 and max_dF < 1e-11 and max_dq < 1e-12, f"{arm} step-0 parity failed"

        # Perturb fresh weights to test active-weight remote-zero, FD consistency, and SO(3) equivariance
        if fresh:
            with torch.no_grad():
                for prm in fresh:
                    prm.add_(torch.randn_like(prm) * 0.05)

        b_act = make_batch(zb, pb, dtype=torch.float64, requires_grad=True)
        U_act, _, _, d_act = forward_energy_charge(m, b_act, return_partial=False)
        (dR_act,) = torch.autograd.grad(U_act.sum(), d_act[p.positions])
        F_act = -dR_act.reshape(8, 13, 3)

        # Remote-zero baseline test under ACTIVE weights (set left=0, right=0)
        remote_zero_err = 0.0
        if fresh:
            tb = m.output_modules[0].internal_energy.tensor_branch
            for mod in m.input_modules:
                b_rz = mod(make_batch(zb[:2], pb[:2], dtype=torch.float64, requires_grad=False))
            d_rz = m.representation(b_rz)
            e_mod = m.output_modules[0].internal_energy
            raw_rz = d_rz[p.node_embedding].reshape(2, 13, -1)
            norm_rz = e_mod.feature_norm(e_mod.feature_spec.flatten(d_rz)).reshape(2, 13, -1)
            z_left = torch.zeros(2, 13, 1, device="cuda", dtype=torch.float64)
            z_right = torch.zeros(2, 13, 1, device="cuda", dtype=torch.float64)
            e_rz = tb.orient_energy(raw_rz, z_left, z_right, normalized_features=norm_rz)
            remote_zero_err = float(e_rz.abs().max().item())
            assert remote_zero_err < 1e-12, f"{arm} active remote-zero failed: {remote_zero_err}"

        # Finite-difference energy vs force consistency on frame 0 in float64
        eps_fd = 1e-5
        pb_plus = pb[:1].copy(); pb_plus[0, 1, 2] += eps_fd
        pb_minus = pb[:1].copy(); pb_minus[0, 1, 2] -= eps_fd
        Up_fd, _, _, _ = forward_energy_charge(m, make_batch(zb[:1], pb_plus, dtype=torch.float64), return_partial=False)
        Um_fd, _, _, _ = forward_energy_charge(m, make_batch(zb[:1], pb_minus, dtype=torch.float64), return_partial=False)
        fd_force = -float((Up_fd - Um_fd).item() / (2 * eps_fd))
        fd_err = abs(fd_force - float(F_act[0, 1, 2].item()))
        assert fd_err < 1e-7, f"{arm} FD force mismatch {fd_err}"

        # SO(3) rotation equivariance check under ACTIVE weights
        rot_q, _ = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(123), dtype=torch.float64))
        if torch.linalg.det(rot_q) < 0:
            rot_q[:, 0] *= -1
        pb_rot = pb @ rot_q.numpy().T
        b_rot = make_batch(zb, pb_rot, dtype=torch.float64, requires_grad=True)
        U_rot, _, _, d_rot = forward_energy_charge(m, b_rot, return_partial=False)
        (dR_rot,) = torch.autograd.grad(U_rot.sum(), d_rot[p.positions])
        F_rot = -dR_rot.reshape(8, 13, 3)
        rot_e_err = float((U_rot - U_act).abs().max().item())
        rot_f_err = float((F_rot - (F_act @ rot_q.cuda().T)).abs().max().item())
        assert rot_e_err < 1e-7 and rot_f_err < 1e-6, f"{arm} rotation invariance failed: {rot_e_err}, {rot_f_err}"

        # Force-loss gradient check on fresh params & split-gradient isolation
        grad_norm = 0.0
        prop_grad_on_phys = 0.0
        if fresh:
            b_g = make_batch(zb, pb, dtype=torch.float64, requires_grad=True)
            U_g, q_g, dq_g, d_g = forward_energy_charge(m, b_g, return_partial=True)
            if dq_g is None:
                dR_g, _ = torch.autograd.grad(U_g.sum(), (d_g[p.positions], q_g), create_graph=True)
            else:
                (dR_g,) = torch.autograd.grad(U_g.sum(), d_g[p.positions], create_graph=True)
            F_g = -dR_g.reshape(8, 13, 3)
            floss = (F_g - fb).square().mean()
            grads = torch.autograd.grad(floss, fresh, allow_unused=True)
            assert all(g is not None and torch.isfinite(g).all() for g in grads), f"{arm} has unused/nonfinite fresh param!"
            grad_norm = float(sum(g.square().sum() for g in grads).sqrt().item())
            assert grad_norm > 0.0

        # Verify state_dict cloning handles non-tensor _extra_state
        cloned = clone_state_cpu(m.state_dict())
        assert set(cloned.keys()) == set(m.state_dict().keys())

        report[arm] = dict(
            source_ckpt_sha256=ckpt_sha,
            new_params=int(sum(x.numel() for x in fresh)),
            max_step0_dU=max_dU,
            max_step0_dF=max_dF,
            max_step0_dq=max_dq,
            active_remote_zero_err=remote_zero_err,
            active_fd_force_err=fd_err,
            active_rot_e_err=rot_e_err,
            active_rot_f_err=rot_f_err,
            force_grad_norm_on_fresh=grad_norm,
        )
        del m
    return report


def train_arm(args):
    data = load_cumulene_data(args.npz, args.split, args.eref, args.profile)
    out = args.out_root / args.arm
    if out.exists() and any(out.iterdir()) and not args.resume and not args.overwrite:
        raise FileExistsError(f"Output directory {out} is non-empty; pass --resume or --overwrite")
    out.mkdir(parents=True, exist_ok=True)

    model, cp, fresh, ckpt_sha = build_model_from_gate80k(args.ckpt, args.arm, args.seed, dtype=torch.float32, use_reference=True)
    head = model.output_modules[0]

    backbone_params = list(model.representation.parameters())
    fresh_ids = {id(x) for x in fresh}
    base_energy_params = [prm for prm in head.internal_energy.parameters() if id(prm) not in fresh_ids]
    proposal_params = list(head.proposal.parameters())
    physical_params = backbone_params + base_energy_params + list(fresh)

    assert len({id(x) for x in physical_params} & {id(x) for x in proposal_params}) == 0

    opt_groups = [
        dict(params=backbone_params, lr=args.backbone_lr),
        dict(params=base_energy_params, lr=args.energy_lr),
    ]
    if fresh:
        opt_groups.append(dict(params=fresh, lr=args.new_lr))
    phys_opt = torch.optim.Adam(opt_groups)
    prop_opt = SOAP(proposal_params, lr=args.proposal_lr)

    scheduler = None
    if args.cosine_min_factor < 1.0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            phys_opt,
            lr_lambda=lambda st: args.cosine_min_factor + 0.5 * (1.0 - args.cosine_min_factor) * (1.0 + math.cos(math.pi * st / args.steps)),
        )

    tr_idx = np.array(data["split"]["train"], dtype=int)
    val_idx = np.array(data["split"]["validation"], dtype=int)
    test_idx = np.array(data["split"]["test"], dtype=int)
    z_tr, pos_tr, e_tr, f_tr = data["z"][tr_idx], data["pos"][tr_idx], data["e_int"][tr_idx], data["f"][tr_idx]
    z_val, pos_val, e_val, f_val = data["z"][val_idx], data["pos"][val_idx], data["e_int"][val_idx], data["f"][val_idx]
    z_te, pos_te, e_te, f_te = data["z"][test_idx], data["pos"][test_idx], data["e_int"][test_idx], data["f"][test_idx]

    ident = dict(
        schema="acvr-cumulene-opt-v2",
        arm=args.arm,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        backbone_lr=args.backbone_lr,
        energy_lr=args.energy_lr,
        new_lr=args.new_lr if fresh else 0.0,
        proposal_lr=args.proposal_lr,
        warmup_new_only_steps=args.warmup_new_steps if fresh else 0,
        cosine_min_factor=args.cosine_min_factor,
        selection_metric="val_torque_RMSE_eV_rad" if args.select_by == "val_torque" else "val_force_RMSE_meV_A",
        optimizer_reset=True,
        source_checkpoint=str(args.ckpt),
        source_checkpoint_sha256=ckpt_sha,
        source_joint_step=int(cp["joint_step"]),
        split_sha256=data["split_sha"],
        npz_sha256=data["npz_sha"],
        profile_sha256=data["profile_sha"],
        gpu=torch.cuda.get_device_name(0),
    )

    rng = torch.Generator().manual_seed(args.seed + 500000)
    n_train = len(tr_idx)
    perm = torch.randperm(n_train, generator=rng).numpy()
    cursor = 0
    start_step = 0
    prior_elapsed = 0.0
    best_score = float("inf")
    best_step = 0

    resume_path = out / "resume.pt"
    log_path = out / "training.jsonl"
    if args.resume and resume_path.exists():
        rs = torch.load(resume_path, map_location="cpu", weights_only=False)
        if rs["identity"] != ident:
            raise ValueError("Resume identity mismatch")
        model.load_state_dict(rs["model"], strict=True)
        phys_opt.load_state_dict(rs["phys_opt"])
        prop_opt.load_state_dict(rs["prop_opt"])
        if scheduler is not None and rs["scheduler"] is not None:
            scheduler.load_state_dict(rs["scheduler"])
        rng.set_state(rs["rng_state"])
        torch.set_rng_state(rs["torch_rng_state"])
        if torch.cuda.is_available() and rs.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(rs["cuda_rng_state"])
        perm = rs["perm"]
        cursor = int(rs["cursor"])
        start_step = int(rs["step"])
        prior_elapsed = float(rs["elapsed_s"])
        best_score = float(rs["best_score"])
        best_step = int(rs["best_step"])
    else:
        atomic_write_text(out / "config.json", json.dumps(ident, indent=2) + "\n")
        atomic_torch_save(
            dict(identity=ident, model=clone_state_cpu(model.state_dict()), step=0, joint_step=80000),
            out / "ckpt-000000.pt",
        )
        if log_path.exists():
            log_path.unlink()

    t0 = time.monotonic()
    for step in range(start_step + 1, args.steps + 1):
        if cursor + args.batch_size > n_train:
            perm = torch.randperm(n_train, generator=rng).numpy()
            cursor = 0
        sel = perm[cursor:cursor + args.batch_size]
        cursor += args.batch_size

        model.train()
        phys_opt.zero_grad(set_to_none=True)
        prop_opt.zero_grad(set_to_none=True)

        batch = make_batch(z_tr[sel], pos_tr[sel], dtype=torch.float32, requires_grad=True)
        U, q, dq, d = forward_energy_charge(model, batch, return_partial=True)
        if dq is None:
            dR, dq = torch.autograd.grad(U.sum(), (d[p.positions], q), create_graph=True)
        else:
            (dR,) = torch.autograd.grad(U.sum(), d[p.positions], create_graph=True)
        F_pred = -dR.reshape(len(sel), 13, 3)

        eb = torch.as_tensor(e_tr[sel], device="cuda", dtype=torch.float32)
        fb = torch.as_tensor(f_tr[sel], device="cuda", dtype=torch.float32)

        err_e = ((U - eb) / 13.0).square().mean()
        err_f = (F_pred - fb).square().mean()
        phys_loss = 200.0 * (err_e + err_f)
        res_loss = 30.0 * (dq - dq.mean(1, keepdim=True)).square().mean()

        if not (torch.isfinite(phys_loss) and torch.isfinite(res_loss)):
            raise FloatingPointError(f"Non-finite loss at step {step}")

        active_phys = fresh if (fresh and step <= args.warmup_new_steps) else physical_params
        torch.autograd.backward(phys_loss, inputs=active_phys, retain_graph=True)
        torch.autograd.backward(res_loss, inputs=proposal_params)

        torch.nn.utils.clip_grad_norm_(active_phys, 10.0, error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(proposal_params, 10.0, error_if_nonfinite=True)

        phys_opt.step()
        prop_opt.step()
        if scheduler is not None:
            scheduler.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            torch.cuda.synchronize()
            elapsed = prior_elapsed + (time.monotonic() - t0)
            row = dict(
                step=step,
                joint_step=80000 + step,
                elapsed_s=round(elapsed, 2),
                phys_loss=float(phys_loss.detach()),
                res_rms_V=float((res_loss.detach() / 30.0).sqrt()),
            )
            if step % args.eval_every == 0 or step == args.steps:
                state_cpu = clone_state_cpu(model.state_dict())
                model.double()
                val_m = eval_split(model, z_val, pos_val, e_val, f_val)
                prof_m = eval_profile(model, data["profile"], save_dir=out / "curves", tag=f"step{step:06d}")
                row["val"] = val_m
                row["profile_diag"] = {
                    "high_angle_150_175": prof_m["high_angle_150_175"]["torque_RMSE_eV_rad"],
                    "endpoint_175_180": prof_m["endpoint_175_180"]["torque_RMSE_eV_rad"],
                    "full_torque": prof_m["full_torque_RMSE_eV_rad"],
                    "barrier_pred_eV": prof_m["barrier_pred_eV"],
                    "cartesian_force_RMSE_meV_A": prof_m["cartesian_force_RMSE_meV_A"],
                }
                score_val = val_m["torque_RMSE_eV_rad"] if args.select_by == "val_torque" else val_m["force_RMSE_meV_A"]
                if score_val < best_score:
                    best_score = score_val
                    best_step = step
                    atomic_torch_save(
                        dict(identity=ident, model=state_cpu, step=step, joint_step=80000 + step, training_seconds=elapsed),
                        out / "ckpt-best-val.pt",
                    )
                model.float()
                # Save resumable checkpoint atomically
                atomic_torch_save(
                    dict(
                        identity=ident,
                        model=state_cpu,
                        phys_opt=phys_opt.state_dict(),
                        prop_opt=prop_opt.state_dict(),
                        scheduler=scheduler.state_dict() if scheduler is not None else None,
                        rng_state=rng.get_state(),
                        torch_rng_state=torch.get_rng_state(),
                        cuda_rng_state=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                        perm=perm,
                        cursor=cursor,
                        step=step,
                        elapsed_s=elapsed,
                        best_score=best_score,
                        best_step=best_step,
                    ),
                    resume_path,
                )
            with log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            atomic_write_text(out / "status.json", json.dumps(row, indent=2) + "\n")
            print(
                f"[{args.arm}] step={step}/{args.steps} t={elapsed:.1f}s loss={row['phys_loss']:.3f}"
                + (
                    f" val_E={row['val']['energy_RMSE_meV_atom']:.3f} val_F={row['val']['force_RMSE_meV_A']:.2f}"
                    f" val_T={row['val']['torque_RMSE_eV_rad']:.4f} P150={row['profile_diag']['high_angle_150_175']:.4f}"
                    if "val" in row
                    else ""
                ),
                flush=True,
            )

    torch.cuda.synchronize()
    total_s = prior_elapsed + (time.monotonic() - t0)
    final_cpu = clone_state_cpu(model.state_dict())
    atomic_torch_save(
        dict(identity=ident, model=final_cpu, step=args.steps, joint_step=80000 + args.steps, training_seconds=total_s),
        out / "ckpt-final.pt",
    )

    summary = dict(identity=ident, training_seconds=total_s, best_val_step=best_step, best_val_score=best_score)
    for tag, cpath in (("final", out / "ckpt-final.pt"), ("best_val", out / "ckpt-best-val.pt")):
        if not cpath.exists():
            continue
        payload = torch.load(cpath, map_location="cpu", weights_only=False)
        model.double()
        model.load_state_dict(payload["model"], strict=True)
        summary[tag] = dict(
            step=int(payload["step"]),
            checkpoint_sha256=sha256(cpath),
            validation=eval_split(model, z_val, pos_val, e_val, f_val),
            test=eval_split(model, z_te, pos_te, e_te, f_te),
            profile_diag=eval_profile(model, data["profile"], save_dir=out / "curves", tag=tag),
        )
    atomic_write_text(out / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(f"[{args.arm}] DONE in {total_s:.1f}s -> {out / 'summary.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--action", choices=("preflight", "train"), required=True)
    ap.add_argument("--arm", choices=ARMS, default="rich_stf")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--backbone-lr", type=float, default=1e-4)
    ap.add_argument("--energy-lr", type=float, default=1e-5)
    ap.add_argument("--new-lr", type=float, default=1e-3)
    ap.add_argument("--proposal-lr", type=float, default=1e-4)
    ap.add_argument("--warmup-new-steps", type=int, default=0)
    ap.add_argument("--cosine-min-factor", type=float, default=0.1)
    ap.add_argument("--select-by", choices=("val_torque", "val_force"), default="val_torque")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--out-root", type=Path, default=HERE / "runs_local")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    ap.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--eref", type=Path, default=DEFAULT_EREF)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    configure_deterministic()
    if args.action == "preflight":
        args.out_root.mkdir(parents=True, exist_ok=True)
        res = preflight(args)
        atomic_write_text(args.out_root / "preflight.json", json.dumps(res, indent=2) + "\n")
        print(json.dumps(res, indent=2))
    else:
        train_arm(args)


if __name__ == "__main__":
    main()
