"""Verify the fixed-shape molecular ACVR and frozen/unfrozen CUDA Graph updates on real SA-SA training geometries."""

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import MethodType

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dpa4c_acvr"))
import torch
from train_pilot import build_model, data_panel
from graph_joint import JointGraph, eligible
from fixed_q_batch import panel_forward
from lightning_joint import make_groups
from cache import output_head
from check_small_native import cast
from curator.train.train import tree_to_device
from curator.data import properties as p
from common import write_json


def differences(a, b):
    diff = (a - b).double()
    return dict(
        max_abs=diff.abs().max().item(),
        relative_l2=(diff.norm() / a.double().norm().clamp_min(1e-15)).item(),
    )


def vectors(m, trainer, batch):
    batch = tree_to_device(batch, "cuda")
    v = m(batch)["variational_prediction"]
    r = v["residuals"]["stationarity"]["values"]
    loss = trainer.supervision({"variational_prediction": v}, batch)["loss"]
    result = {name: lab["values"].detach().cpu() for name, lab in v["labels"].items()}
    result["stationarity"] = r.detach().cpu()
    for key, obj, params in (
        ("physical_grad", loss, trainer.physical),
        ("proposal_grad", 30 * r.square().mean(), trainer.proposal),
    ):
        grad = torch.autograd.grad(obj, params, retain_graph=True, allow_unused=True)
        result[key] = torch.cat(
            [
                (g if g is not None else torch.zeros_like(p)).detach().cpu().flatten()
                for p, g in zip(params, grad)
            ]
        )
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("trajectory", "split", "reference", "output"):
        ap.add_argument("--" + key, type=Path, required=True)
    args = ap.parse_args()
    args.batch, args.seed = 4, 3407
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    split = json.loads(args.split.read_text())
    ids = {"train": split["train"][:16], "validation": split["validation"][:1]}
    with TemporaryDirectory() as raw:
        tmp = Path(raw)
        args.panel = tmp / "panel.json"
        args.panel.write_text(json.dumps(ids))
        ds = data_panel(args, tmp)
    collate = ds.train_dataloader().collate_fn
    samples = ds.datasets["train"]
    batches = [collate([samples[i + j] for j in range(4)]) for i in range(0, 16, 4)]
    for b in batches:
        assert (
            b[p.n_atoms].eq(14).all() and b[p.total_charge].eq(0).all() and eligible(b)
        )

    def make(fast, dtype):
        model = build_model(
            ds.sources[0].species,
            seed=args.seed,
            backbone_backend="native" if fast else "reference",
            native_proposal_spline=fast,
            native_qefa=fast,
        ).to(device="cuda", dtype=dtype)
        if fast:
            head = output_head(model)
            head.forward = MethodType(panel_forward, head)
        return model, make_groups(model)

    rows = {}
    for dtype in (torch.float64, torch.float32):
        reference, rt = make(False, dtype)
        model, tr = make(True, dtype)
        model.load_state_dict(reference.state_dict(), strict=True)
        aa, bb = [], []
        for batch in batches[:2]:
            a = vectors(reference, rt, cast(batch, dtype))
            b = vectors(model, tr, cast(batch, dtype))
            aa.append(a)
            bb.append(b)
        diffs = {}
        for name in aa[0]:
            x = torch.cat([a[name] for a in aa])
            y = torch.cat([b[name] for b in bb])
            diffs[name] = differences(x, y)
            if dtype == torch.float64:
                torch.testing.assert_close(x, y, atol=1e-8, rtol=1e-8)
            else:
                assert (
                    diffs[name]["relative_l2"] < 3e-4 or diffs[name]["max_abs"] < 1e-8
                ), (name, diffs[name])
        rows[str(dtype)] = diffs
        del reference, model, rt, tr
    # Both capture modes must agree with eager routing and leave the Adam
    # clock untouched for the frozen backbone. The final update releases it.
    model, tr = make(True, torch.float32)
    ref, rt = make(True, torch.float32)
    ref.load_state_dict(model.state_dict())
    graph = JointGraph(tr)
    before = [p.detach().clone() for p in tr.backbone]
    graph_steps = []
    for i, frozen in enumerate((True, True, False)):
        batch = cast(batches[i], torch.float32)
        physical = rt.physical
        if frozen:
            rt.physical = rt.energy
        try:
            eager = rt.step(batch, "joint")
        finally:
            rt.physical = physical
        got = graph.step(batch, "joint", freeze_backbone=frozen)
        assert tr.physics_optimizer.state[tr.backbone[0]].get("step", 0) == (
            0 if frozen else 1
        )
        delta = max(
            float((p.detach() - q).abs().max()) for p, q in zip(tr.backbone, before)
        )
        assert (delta == 0) == frozen
        for key in got:
            torch.testing.assert_close(got[key], eager[key], atol=3e-6, rtol=3e-4)
        graph_steps.append(
            dict(frozen=frozen, graph=got, eager=eager, backbone_max_delta=delta)
        )
    assert graph.replays == 3 and graph.fallbacks == 0
    write_json(
        args.output,
        dict(
            device=torch.cuda.get_device_name(),
            frames=ids["train"],
            seed=args.seed,
            description="Real SA-SA fixed Q=0 training frames only. Two independent batches, native single-pass CUDA vs generic Curator + DPA reference for E/F/q/residual and full parameter gradients. Graph/eager optimizer comparison freezes backbone for two updates, then enables it.",
            comparisons=rows,
            graph_steps=graph_steps,
            captures=graph.captures,
            replays=graph.replays,
            fallbacks=graph.fallbacks,
        ),
    )
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
