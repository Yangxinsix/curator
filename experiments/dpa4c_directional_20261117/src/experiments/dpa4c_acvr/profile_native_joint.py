"""Compare native Small training and isolate the cost of the ACVR head and losses."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory
from time import perf_counter
from types import MethodType
from unittest.mock import patch

from check_small_native import cast
from compare_pure_speed import pure_update
from train_pilot import HERE, ScratchTrainer, build_model, data_panel
from fixed_q_batch import panel_forward
import fixed_q_batch as panel
from cache import output_head
from common import write_json
from curator.train.train import tree_to_device
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("trajectory", "split", "reference", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--compare-qefa", action="store_true")
    args = parser.parse_args()
    args.panel = HERE / "pure_local/source_indices.json"
    args.batch, args.seed = 4, 3407
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ck = torch.load(HERE / "sabs_warm/final.pt", map_location="cpu", weights_only=False)
    with TemporaryDirectory() as tmp:
        ds = data_panel(args, Path(tmp))
    data = ds.datasets["train"]
    collate = ds.train_dataloader().collate_fn
    order = torch.randperm(
        len(data), generator=torch.Generator().manual_seed(args.seed)
    ).tolist()

    def batch(i, dtype=torch.float32):
        idx = i % (len(data) // 4) * 4
        return cast(collate([data[j] for j in order[idx : idx + 4]]), dtype)

    def setup(spline, dtype=torch.float32):
        m = build_model(
            **(
                ck["model_options"]
                | dict(backbone_backend="native", native_proposal_spline=(args.compare_qefa or spline), native_qefa=(args.compare_qefa and spline))
            )
        )
        m.load_state_dict(ck["model"], strict=True)
        m.to(device="cuda", dtype=dtype)
        output_head(m).forward = MethodType(panel_forward, output_head(m))
        return ScratchTrainer(
            m,
            backbone_lr=1e-4,
            energy_lr=1e-5,
            proposal_lr=1e-4,
            energy_weight=200.0,
            force_weight=200.0,
        )

    result = {
        "device": torch.cuda.get_device_name(),
        "protocol": "Native DPA4C-Small, RTX GPU, SA-BS 4 x 13, same FP32 checkpoint and batch order, collation/transfer and rebuilt neighbors included; no TF32. Pure trains energy and force with identical backbone LR, loss scales and Adam. Joint adds q-EFA readout, one-pass proposal, original stationarity and SOAP. Three methods rotated across 45 steps with first nine discarded. Stage measurements use separate synchronized instrumentation and arithmetic means; phases outside measured regions included in remainder. No checkpoint writes.",
    }

    def outputs(t, i, dtype):
        b = tree_to_device(batch(i, dtype), "cuda")
        v = t.model(b)["variational_prediction"]
        r = v["residuals"]["stationarity"]["values"]
        loss = t.supervision({"variational_prediction": v}, b)["loss"]
        out = {
            k: v["labels"][k]["values"]
            for k in ["condition_energy", "condition_forces", "variable_value"]
        }
        out["stationarity"] = r
        for label, cost, ps in [
            ("physical", loss, t.physical),
            ("proposal", 30 * r.square().mean(), t.proposal),
        ]:
            grad = torch.autograd.grad(cost, ps, retain_graph=True, allow_unused=True)
            out[label + "_grads"] = torch.cat(
                [
                    (g if g is not None else torch.zeros_like(p)).flatten()
                    for g, p in zip(grad, ps)
                ]
            )
        return {k: x.detach().double().reshape(-1).cpu() for k, x in out.items()}

    if args.compare_qefa:
        result["protocol"] += " Both joint models use native proposal splines. The treatment enables molecular ERoPE and factorized CUDA communication."
    from kan_proposal import bspline_basis
    from native_spline import CubicBasis

    z = torch.tensor(
        [-1, -0.999, -0.63, -0.001, 0, 0.001, 0.73, 0.999, 1],
        device="cuda",
        requires_grad=True,
    )
    knots = torch.arange(-4, 5, device="cuda", dtype=z.dtype)
    y = bspline_basis(z, knots)
    native = CubicBasis.apply(z, 0)
    w = torch.arange(y.numel(), device="cuda", dtype=y.dtype).reshape_as(y).cos()
    diffs = []
    for derivative in range(3):
        diffs.append(float((y - native).abs().max().detach()))
        if derivative < 2:
            y = torch.autograd.grad((y * w).sum(), z, create_graph=True)[0]
            native = torch.autograd.grad((native * w).sum(), z, create_graph=True)[0]
            w = torch.ones_like(z)
    assert max(diffs) < 1e-12
    result["spline_basis_max_abs_F64_value_first_second"] = diffs
    for dtype in [torch.float64, torch.float32]:
        stores = []
        for optimized in [False, True]:
            t = setup(optimized, dtype)
            rows = [outputs(t, i, dtype) for i in [1, 5, 11]]
            stores.append({k: torch.cat([row[k] for row in rows]) for k in rows[0]})
        diff = {}
        for k, x in stores[0].items():
            delta = stores[1][k] - x
            diff[k] = dict(
                rmse=float(delta.square().mean().sqrt()),
                relative_L2=float(delta.norm() / x.norm()),
            )
        print(dtype, diff, flush=True)
        assert max(v["relative_L2"] for v in diff.values()) < (
            1.0e-8 if dtype == torch.float64 else 0.003
        )
        result[str(dtype) + "_parity"] = diff
    new_name = "joint_cuda_qefa" if args.compare_qefa else "joint_cuda_spline"
    trainers = {
        key: setup(key == new_name)
        for key in ("pure", "joint", new_name)
    }

    def run(key, i):
        t = trainers[key]
        b = batch(i)
        if key == "pure":
            return pure_update(t.model, t, b)
        return t.step(b, "joint")

    times = defaultdict(list)
    for i in range(45):
        names = list(trainers)
        rot = i % 3
        for key in names[rot:] + names[:rot]:
            torch.cuda.synchronize()
            start = perf_counter()
            run(key, i)
            torch.cuda.synchronize()
            times[key].append((perf_counter() - start) * 1000)
    result["timings"] = {
        k: dict(median_ms=median(v[9:]), mean_ms=mean(v[9:]), steps_ms=v)
        for k, v in times.items()
    }
    print(
        "timings", {k: v["median_ms"] for k, v in result["timings"].items()}, flush=True
    )
    write_json(args.output, result)

    def profile(key):
        t = trainers[key]
        samples = defaultdict(list)
        stages = {}

        def measured(name, fn, *args, **kwargs):
            torch.cuda.synchronize()
            start = perf_counter()
            value = fn(*args, **kwargs)
            torch.cuda.synchronize()
            stages[name] = stages.get(name, 0.0) + (perf_counter() - start) * 1000
            return value

        def wrap(stack, obj, attr, name):
            fn = getattr(obj, attr)
            stack.enter_context(
                patch.object(obj, attr, lambda *a, **kw: measured(name, fn, *a, **kw))
            )

        true_grad = torch.autograd.grad

        def grad(*a, **kw):
            return measured("initial_derivatives_R_or_Rq", true_grad, *a, **kw)

        true_backward = torch.autograd.backward

        def backward(*a, **kw):
            name = (
                "proposal_backward"
                if kw.get("inputs") is t.proposal
                else "physical_backward"
            )
            return measured(name, true_backward, *a, **kw)

        for i in range(20):
            stages = {}
            head = output_head(t.model)
            with ExitStack() as stack:
                for obj, attr, name in [
                    (t.model.representation, "forward", "backbone_forward"),
                    (panel, "panel_energy_charge", "charge_head_forward"),
                    (t.physics_optimizer, "step", "Adam"),
                    (t.solver_optimizer, "step", "SOAP"),
                ]:
                    wrap(stack, obj, attr, name)
                stack.enter_context(patch.object(torch.autograd, "grad", grad))
                stack.enter_context(patch.object(torch.autograd, "backward", backward))
                total = perf_counter()
                run(key, i)
                torch.cuda.synchronize()
                elapsed = (perf_counter() - total) * 1000
            stages["other"] = elapsed - sum(stages.values())
            stages["total"] = elapsed
            for name, time in stages.items():
                samples[name].append(time)
        return {k: mean(v) for k, v in samples.items()}

    result["stage_mean_ms"] = {key: profile(key) for key in trainers}
    print("stages", result["stage_mean_ms"], flush=True)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
