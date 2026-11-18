"""Verify all AuMgO ACVR arms on the real slab, separately from fitting any model."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from types import MethodType

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from ase.io.trajectory import Trajectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dpa4c_acvr"))
from model import build_model, ScratchTrainer
from aumgo_arms import AuMgOGraph, install_aumgo_arm
from common import write_json
from curator.data import properties as p
from curator.train.train import tree_to_device
from fixed_q_batch import panel_forward
from compare_pure_speed import compute
from mixed_data import build_data


def outputs(model, trainer, batch, device):
    model.train()
    inp = tree_to_device(deepcopy(batch), device)
    pred = model(inp)["variational_prediction"]
    loss = trainer.supervision({"variational_prediction": pred}, inp)["loss"]
    residual = pred["residuals"]["stationarity"]["values"]
    out = {
        k: v["values"].detach().cpu()
        for k, v in pred["labels"].items()
        if k in {"condition_energy", "condition_forces", "variable_value"}
    }
    out["stationarity"] = residual.detach().cpu()
    for name, obj, params in (
        ("physical_gradient", loss, trainer.physical),
        ("proposal_gradient", 30 * residual.square().mean(), trainer.proposal),
    ):
        grads = torch.autograd.grad(obj, params, retain_graph=True, allow_unused=True)
        out[name] = torch.cat(
            [
                (torch.zeros_like(z) if g is None else g).flatten().detach().cpu()
                for z, g in zip(params, grads)
            ]
        )
    return out


def single(model, b, device):
    pred = model(tree_to_device(deepcopy(b), device))["variational_prediction"]
    return {
        k: v["values"].detach().cpu()
        for k, v in pred["labels"].items()
        if k in {"condition_energy", "condition_forces", "variable_value"}
    }


def main(args):
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = getattr(torch, args.dtype)
    original_ids = json.loads(args.split.read_text())
    ids = (
        original_ids["train"][: args.batch]
        + original_ids["validation"][: max(2, args.batch)]
    )
    with Trajectory(args.trajectory) as source, TemporaryDirectory() as tmp:
        root = Path(tmp)
        with Trajectory(root / "panel.traj", "w") as dest:
            for index in ids:
                dest.write(source[index])
        frame = source[ids[-1]]
        write_json(
            root / "split.json",
            {
                "train": list(range(args.batch)),
                "validation": list(range(args.batch, len(ids))),
                "test": [],
            },
        )
        source_cfg = dict(
            datapath="panel.traj",
            split_file="split.json",
            boundary=dict(type="fixed_charge", total_charge_key="total_charge"),
            energy_reference_file=str(args.reference.resolve()),
        )
        ds = build_data(
            dict(
                sources=[source_cfg],
                energy_convention="internal",
                batch_size=args.batch,
                default_dtype=args.dtype,
            ),
            base_dir=root,
        )
    collate = ds.val_dataloader().collate_fn
    batch = next(iter(ds.val_dataloader()))
    single_batch = collate(
        [ds.datasets["validation"][len(ds.datasets["validation"]) - 1]]
    )
    species = ds.sources[0].species

    def setup(arm, *, native):
        model = build_model(
            species,
            seed=3407,
            periodic_backend="rsa",
            dtype=torch.float64,
            backbone_backend="native"
            if native and args.device == "cuda"
            else "reference",
            native_proposal_spline=native and args.device == "cuda",
        )
        install_aumgo_arm(model, arm)
        model.to(device=args.device, dtype=dtype)
        t = ScratchTrainer(
            model,
            backbone_lr=1e-4,
            energy_lr=1e-5,
            proposal_lr=1e-4,
            energy_weight=200.0,
            force_weight=200.0,
        )
        return model, t

    def check(x, y, *, rtol=4e-8, atol=4e-8):
        norm = torch.linalg.vector_norm(x)
        rel = float(torch.linalg.vector_norm(x - y) / norm.clamp_min(1e-25))
        peak = float((x - y).abs().max())
        if args.dtype == "float32":
            rtol, atol = 1e-3, 3e-5
        if not (rel <= rtol or peak <= atol):
            raise ValueError(f"Mismatch relative L2 {rel} and max absolute {peak}")
        return dict(max_abs=peak, relative_L2=rel)

    rows = {}
    initial = None
    for arm in args.arms:
        model, trainer = setup(arm, native=False)
        names = (
            "condition_energy",
            "condition_forces",
            "variable_value",
            "stationarity",
        )
        baseline = outputs(model, trainer, batch, args.device)
        if initial is None:
            initial = {k: baseline[k] for k in names}
        init_diffs = {k: check(initial[k], baseline[k]) for k in names}
        fast, ft = setup(arm, native=True)
        fast.output_modules[0].forward = MethodType(
            panel_forward, fast.output_modules[0]
        )
        accel = outputs(fast, ft, batch, args.device)
        diffs = {k: check(baseline[k], v) for k, v in accel.items()}
        # An R-dependent log-gain gradient should flow out of the *physical* loss to every supplied group.
        gains = next(
            (
                p
                for k, p in model.output_modules[
                    0
                ].internal_energy.attention.named_parameters()
                if k in {"band_coefficients", "group_log_gain"}
            ),
            None,
        )
        gradient_norm = None
        if gains is not None:
            val = model(tree_to_device(deepcopy(batch), args.device))[
                "variational_prediction"
            ]
            label = val["labels"]
            obj = (
                label["condition_forces"]["values"].square().mean()
                + label["condition_energy"]["values"].square().mean() / 110**2
            )
            grad = torch.autograd.grad(obj, gains)[0]
            gradient_norm = float(grad.norm())
            if not torch.isfinite(grad).all() or gradient_norm == 0:
                raise ValueError("Inactive periodic spectral parameter")
        gain_diffs = {}
        if gains is not None:
            # Exercise a non-uniform trained-like spectrum, including its gradient.
            with torch.no_grad():
                gains.copy_(torch.linspace(-0.6, 0.7, gains.numel()).reshape_as(gains))
                fast.load_state_dict(model.state_dict())
            nonuniform = outputs(model, trainer, batch, args.device)
            nonuniform_fast = outputs(fast, ft, batch, args.device)
            gain_diffs = {
                k: check(v, nonuniform_fast[k]) for k, v in nonuniform.items()
            }
        # Only generic per-condition inference may be used with the larger supercell.
        primitive = single(model, single_batch, args.device)
        moved = deepcopy(single_batch)
        idx = int(np.flatnonzero(frame.numbers == 79)[0])
        moved[p.positions][idx] += moved[p.cell].reshape(3, 3)[0]
        wrapped = single(model, moved, args.device)
        wrapping = {k: check(v, wrapped[k], atol=5e-8) for k, v in primitive.items()}
        from dataclasses import asdict
        from curator.data.charge_conditions import ChargeBoundary
        from curator.data.variational_data import canonical_boundary_hash

        repeated = deepcopy(single_batch)
        tiled = frame.repeat((2, 1, 1))
        n = len(tiled)
        for k in (p.positions, p.atomic_numbers):
            base = tiled.positions if k == p.positions else tiled.numbers
            repeated[k] = torch.as_tensor(base, dtype=repeated[k].dtype)
        repeated[p.cell] = torch.as_tensor(tiled.cell.array, dtype=dtype).reshape_as(
            single_batch[p.cell]
        )
        repeated[p.n_atoms] = torch.full_like(repeated[p.n_atoms], n)
        repeated[p.image_idx] = torch.zeros(n, dtype=torch.long)
        spec = asdict(
            ChargeBoundary(type="fixed_charge", total_charge=0.0).build(tiled)[0]
        )
        var = repeated["variational"]
        var["boundary_spec"] = (spec,)
        var["boundary_hash"] = (canonical_boundary_hash(spec),)
        var["condition_state_ptr"] = torch.tensor([0, n], dtype=torch.long)
        supercell = single(model, repeated, args.device)
        replication = {}
        for k, v in primitive.items():
            target = 2 * v if k == "condition_energy" else v.repeat(2)
            replication[k] = check(target, supercell[k], atol=2e-7)
        rows[arm] = dict(
            zero_gain_outputs=init_diffs,
            native_against_general=diffs,
            gain_gradient_norm=gradient_norm,
            nonuniform_gain_parity=gain_diffs,
            single_Au_wrap=wrapping,
            repeat_2x1x1=replication,
        )
        write_json(
            args.output, dict(ids=ids, device=args.device, dtype=args.dtype, arms=rows)
        )
        print(arm, rows[arm], flush=True)
    if args.graph:
        model, t = setup("A4", native=True)
        graph = AuMgOGraph(t, max_shapes=2)
        intervals = []
        routed_loss_deltas = []
        train_batch = next(iter(ds.train_dataloader()))
        for i in range(4):
            step_batch = batch if i % 2 == 0 else train_batch
            eager_batch = tree_to_device(deepcopy(step_batch), args.device)
            eager = model(eager_batch)
            expected_phys = float(t.supervision(eager, eager_batch)["loss"].detach())
            expected_rms = float(
                eager["variational_prediction"]["residuals"]["stationarity"]["values"]
                .square()
                .mean()
                .detach()
                .sqrt()
            )
            del eager, eager_batch
            torch.cuda.synchronize()
            start = time.perf_counter()
            stat = graph.step(deepcopy(step_batch), "joint")
            torch.cuda.synchronize()
            intervals.append(time.perf_counter() - start)
            diff = {
                "physical": abs(stat["physical_loss"] - expected_phys),
                "stationarity": abs(stat["stationarity_rms_V"] - expected_rms),
            }
            if (
                diff["physical"] > max(1.0e-5, abs(expected_phys) * 2e-4)
                or diff["stationarity"] > 1.0e-7
            ):
                raise ValueError(
                    f"Graph loss differs from generic evaluation before update {i}: {diff}"
                )
            routed_loss_deltas.append(diff)
        if graph.fallbacks or not graph.replays:
            raise ValueError("Periodic joint graph did not replay")
        pure_model, pure_trainer = setup("A0", native=True)
        pure_graph = AuMgOGraph(pure_trainer, pure=True, max_shapes=2)
        pure_wall = []
        for test_batch in (batch, train_batch, batch, train_batch):
            transported = tree_to_device(deepcopy(test_batch), args.device)
            energy, force = compute(pure_model, transported, False, True, False)
            y = transported["variational"]["labels"]
            e_loss = (
                ((energy.flatten() - y["condition_energy"]["values"]) / 110)
                .square()
                .mean()
            )
            f_loss = (force.flatten() - y["condition_forces"]["values"]).square().mean()
            initial_loss = float((200 * (e_loss + f_loss)).detach())
            del e_loss, f_loss, energy, force, transported
            torch.cuda.synchronize()
            ts = time.perf_counter()
            actual = pure_graph.step(deepcopy(test_batch), "joint")["physical_loss"]
            torch.cuda.synchronize()
            pure_wall.append(time.perf_counter() - ts)
            if abs(actual - initial_loss) > max(1.0e-5, abs(initial_loss) * 2e-4):
                raise ValueError("Pure graph changed supervised E/F loss")
        if pure_graph.fallbacks or not pure_graph.replays:
            raise ValueError("Pure periodic small graph did not replay")
        write_json(
            args.output,
            dict(
                ids=ids,
                device=args.device,
                dtype=args.dtype,
                arms=rows,
                pure_graph=dict(
                    replays=pure_graph.replays,
                    captures=pure_graph.captures,
                    wall_seconds=pure_wall,
                ),
                graph=dict(
                    replays=graph.replays,
                    captures=graph.captures,
                    final=stat,
                    wall_seconds=intervals,
                    graph_vs_generic=routed_loss_deltas,
                ),
            ),
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("trajectory", "split", "reference", "output"):
        ap.add_argument("--" + k, type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--arms", nargs="+", default=["A1", "A2", "A3", "A4"])
    ap.add_argument("--graph", action="store_true")
    with torch.jit.optimized_execution(False):
        main(ap.parse_args())
