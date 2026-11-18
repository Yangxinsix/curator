"""End-to-end molecular ACVR timing, gradient and trajectory comparison."""

import argparse
import math
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory
from time import perf_counter
from types import MethodType

from train_pilot import HERE, build_model, data_panel
from fixed_q_batch import panel_forward
from lightning_joint import make_groups
from compare_pure_speed import pure_update
from check_small_native import cast
from cache import output_head
from common import write_json
from graph_joint import JointGraph
from curator.train.train import tree_to_device
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trajectory", "reference", "split", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    args.seed, args.batch, args.panel = 3407, 4, HERE / "pure_local/source_indices.json"
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with TemporaryDirectory() as tmp:
        ds = data_panel(args, Path(tmp))
    collate = ds.train_dataloader().collate_fn
    ck = torch.load(HERE / "sabs_warm/final.pt", weights_only=False, map_location="cpu")
    gen = torch.Generator().manual_seed(args.seed)
    train = ds.datasets["train"]
    val = ds.datasets["validation"]
    ids = [
        j for _ in range(3) for j in torch.randperm(len(train), generator=gen).tolist()
    ]

    def get(i):
        return cast(collate([train[ids[4 * i + j]] for j in range(4)]), torch.float32)

    def setup(native):
        m = build_model(
            **(
                ck["model_options"]
                | dict(
                    backbone_backend="native",
                    native_proposal_spline=True,
                    native_qefa=native,
                )
            )
        )
        m.load_state_dict(ck["model"])
        m = m.cuda().float()
        output_head(m).forward = MethodType(panel_forward, output_head(m))
        return make_groups(m)

    keys = ("pure", "pure_graph", "eager", "fused_eager", "graph", "fused_graph")
    trainers = {k: setup(k in ("fused_eager", "fused_graph")) for k in keys}
    graphs = {
        k: JointGraph(trainers[k], pure=k == "pure_graph")
        for k in ("pure_graph", "graph", "fused_graph")
    }
    stats = {k: [] for k in keys}
    outputs = {k: [] for k in keys if not k.startswith("pure")}
    for i in range(51):
        for key in keys[i % len(keys) :] + keys[: i % len(keys)]:
            torch.cuda.synchronize()
            start = perf_counter()
            batch = get(i)
            t = trainers[key]
            if key in graphs:
                z = graphs[key].step(batch, "joint")
            elif key == "pure":
                pure_update(t.model, t, batch)
            else:
                z = t.step(batch, "joint")
            torch.cuda.synchronize()
            stats[key].append(1000 * (perf_counter() - start))
            if not key.startswith("pure"):
                outputs[key].append(z)
    result = dict(
        device=torch.cuda.get_device_name(),
        protocol="4 x 13 SA-BS, same FP32 checkpoint and same shuffled minibatches across 51 sequential updates per arm. Interleaved wall timing includes CPU collation, GPU transfer, neighbor rebuild, gradient clipping, Adam and SOAP. First 11 omitted from steady medians; measured capture, warmup and first optimizer updates included in all-step totals. Pure and pure_graph train native Small with the same E/F Adam update only. ACVR heads use native KAN splines. Graph uses static shape, live geometry/topology/targets/parameters, and original fixed-Q stationarity. Fused uses molecular rotary CUDA and the explicit derivative of the same readout; communication GEMMs stay factorized. Validation uses all 32 held-out frames.",
    )
    result["timings"] = {
        k: dict(
            median_ms=median(v[11:]),
            mean_ms=mean(v[11:]),
            all_step_seconds=sum(v) / 1000,
            first_step_ms=v[0],
            steps_ms=v,
        )
        for k, v in stats.items()
    }
    result["graph_usage"] = {
        k: dict(captures=g.captures, replays=g.replays, fallbacks=g.fallbacks)
        for k, g in graphs.items()
    }

    def row(t):
        out = {k: [] for k in ("energy", "force", "q", "stationarity")}
        for i in range(0, len(val), 4):
            b = tree_to_device(
                cast(
                    collate([val[j] for j in range(i, min(len(val), i + 4))]),
                    torch.float32,
                ),
                "cuda",
            )
            pred = t.model(b)["variational_prediction"]
            labs = pred["labels"]
            out["energy"].append(
                (
                    labs["condition_energy"]["values"]
                    - b["variational"]["labels"]["condition_energy"]["values"]
                )
                .detach()
                .double()
                / 13
            )
            out["force"].append(
                (
                    labs["condition_forces"]["values"]
                    - b["variational"]["labels"]["condition_forces"]["values"]
                )
                .detach()
                .double()
            )
            out["q"].append(labs["variable_value"]["values"].detach().double())
            out["stationarity"].append(
                pred["residuals"]["stationarity"]["values"].detach().double()
            )
        return {k: torch.cat(v) for k, v in out.items()}

    results = {k: row(trainers[k]) for k in keys if not k.startswith("pure")}

    def rms(v):
        return float(v.square().mean().sqrt())

    result["validation"] = {
        k: {n: rms(v) for n, v in d.items() if n != "q"} for k, d in results.items()
    }
    result["validation_output_difference_vs_eager"] = {
        k: {n: rms(v - results["eager"][n]) for n, v in d.items()}
        for k, d in results.items()
    }
    result["max_param_abs_after_51_vs_eager"] = {}
    for key in results:
        result["max_param_abs_after_51_vs_eager"][key] = {
            group: max(
                float((p - q).detach().abs().max())
                for p, q in zip(
                    getattr(trainers[key], group), getattr(trainers["eager"], group)
                )
            )
            for group in ("energy", "backbone", "proposal")
        }
    result["pure_graph_backbone_max_abs_vs_pure_after_51"] = max(
        float((p - q).detach().abs().max())
        for p, q in zip(trainers["pure_graph"].backbone, trainers["pure"].backbone)
    )
    for g in graphs.values():
        assert g.replays == 51 and not g.fallbacks
    for rows in outputs.values():
        assert all(math.isfinite(v) for row in rows for v in row.values())
    write_json(args.output, result)
    print(
        "times",
        {k: round(v["median_ms"], 2) for k, v in result["timings"].items()},
        result["graph_usage"],
        flush=True,
    )
    print("validation", result["validation"], flush=True)


if __name__ == "__main__":
    main()
