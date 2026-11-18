"""Check Small native CUDA against the identical SA-BS checkpoint and batches."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType

from compare_pure_speed import compute
from statistics import median
from time import perf_counter
from train_pilot import HERE, data_panel, build_model, ScratchTrainer
from fixed_q_batch import panel_forward
from cache import output_head
from common import write_json
from curator.data import properties as p
from curator.train.train import tree_to_device
import torch


def cast(x, dtype):
    if isinstance(x, torch.Tensor):
        return x.to(dtype) if x.is_floating_point() else x
    if isinstance(x, dict):
        return {k: cast(v, dtype) for k, v in x.items()}
    if isinstance(x, list):
        return [cast(v, dtype) for v in x]
    if isinstance(x, tuple):
        return tuple(cast(v, dtype) for v in x)
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("trajectory", "split", "reference", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
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

    def batch(i):
        idx = i % (len(data) // args.batch) * args.batch
        return collate([data[j] for j in order[idx : idx + args.batch]])

    batches = [batch(j) for j in range(12)]
    result = {
        "device": torch.cuda.get_device_name(),
        "protocol": "Same trained Small weights, same SA-BS nonperiodic fixed net charge -1, 4 x 13 atoms; no TF32. Accuracy on three batches. Inference includes device transfer and neighbor construction, excludes collation. Training includes collation, transfer, neighbor construction, Adam E/F and SOAP stationarity, one proposal pass. Median of 40 synchronized steps after 9 warmups; strictly paired same batch, alternating first backend across successive pairs, fresh weights per backend. No checkpoint writes.",
        "parity": {},
        "timings": {},
    }

    def make(backend, dtype):
        m = build_model(**(ck["model_options"] | {"backbone_backend": backend}))
        m.load_state_dict(ck["model"], strict=True)
        m = m.to(device="cuda", dtype=dtype)
        output_head(m).forward = MethodType(panel_forward, output_head(m))
        return m

    def trainer(m):
        return ScratchTrainer(
            m,
            backbone_lr=1e-4,
            energy_lr=1e-5,
            proposal_lr=1e-4,
            energy_weight=200.0,
            force_weight=200.0,
        )

    for dtype in [torch.float64, torch.float32]:
        ref = {}
        for backend in ["reference", "native"]:
            m = make(backend, dtype)
            t = trainer(m)
            outputs = {}
            for i in [3, 7, 11]:
                b = tree_to_device(cast(batches[i], dtype), "cuda")
                v = m(b)["variational_prediction"]
                feats = m.representation
                geom = dict(b)
                for layer in m.input_modules:
                    geom = layer(geom)
                geom = feats(geom)
                labels = v["labels"]
                residual = v["residuals"]["stationarity"]["values"]
                row = {
                    k: labels[k]["values"]
                    for k in ["condition_energy", "condition_forces", "variable_value"]
                }
                row.update(
                    scalar=geom[p.node_feat],
                    vector=geom[p.node_vect],
                    stationarity=residual,
                )
                loss = t.supervision({"variational_prediction": v}, b)["loss"]
                for name, cost, params in [
                    ("physical", loss, t.physical),
                    ("proposal", 30 * residual.square().mean(), t.proposal),
                ]:
                    grads = torch.autograd.grad(
                        cost, params, retain_graph=True, allow_unused=True
                    )
                    row[name + "_grads"] = torch.cat(
                        [
                            (g if g is not None else torch.zeros_like(param)).flatten()
                            for g, param in zip(grads, params)
                        ]
                    )
                for k, x in row.items():
                    outputs.setdefault(k, []).append(
                        x.detach().cpu().double().reshape(-1)
                    )
            vals = {k: torch.cat(v) for k, v in outputs.items()}
            if backend == "reference":
                ref = vals
            else:
                differences = {}
                for k, x in ref.items():
                    delta = vals[k] - x
                    differences[k] = {
                        "RMSE": float(delta.square().mean().sqrt()),
                        "max_abs": float(delta.abs().max()),
                        "relative_L2": float(
                            delta.norm() / x.norm().clamp_min(1.0e-30)
                        ),
                    }
                result["parity"][str(dtype)] = differences
                print(dtype, "parity", differences, flush=True)
                write_json(args.output, result)
                assert all(
                    v["relative_L2"] < (1e-8 if dtype == torch.float64 else 0.005)
                    for v in differences.values()
                )
        del t, m

    def compare(tag, operations):
        samples = {name: [] for name in operations}
        for i in range(49):
            names = list(operations)
            if i % 2:
                names.reverse()
            for name in names:
                torch.cuda.synchronize()
                start = perf_counter()
                operations[name](i)
                torch.cuda.synchronize()
                samples[name].append((perf_counter() - start) * 1000)
        for name, times in samples.items():
            key = f"{name}_{tag}"
            result["timings"][key] = dict(median_ms=median(times[9:]), steps_ms=times)
            print(key, result["timings"][key]["median_ms"], flush=True)
        write_json(args.output, result)

    models = {
        name: make(name, torch.float32).eval().requires_grad_(False)
        for name in ("reference", "native")
    }
    fp32_batches = [cast(b, torch.float32) for b in batches]
    for joint in (False, True):
        for forces in (False, True):

            def inference(model, i):
                with torch.set_grad_enabled(forces):
                    return compute(
                        model,
                        tree_to_device(fp32_batches[i % len(batches)], "cuda"),
                        joint,
                        forces,
                        False,
                    )

            tag = f"{'acvr' if joint else 'pure'}_{'EF' if forces else 'E'}"
            compare(
                tag,
                {
                    name: (lambda i, m=model: inference(m, i))
                    for name, model in models.items()
                },
            )
    trainers = {name: trainer(make(name, torch.float32).train()) for name in models}
    compare(
        "acvr_joint_step",
        {
            name: (lambda i, t=t: t.step(cast(batch(i), torch.float32), "joint"))
            for name, t in trainers.items()
        },
    )


if __name__ == "__main__":
    main()
