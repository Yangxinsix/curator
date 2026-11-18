"""Compare pure Small and Small+ACVR on the same molecular batches and device."""

import argparse
import os
import time
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory
from types import MethodType

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from train_pilot import HERE, ScratchTrainer, build_model, data_panel
from fixed_q_batch import panel_energy_charge, panel_forward
from cache import output_head
from common import write_json
from curator.data import properties as p
from curator.train.train import tree_to_device


def compute(model, batch, joint, forces, training):
    data = batch.copy()
    for layer in model.input_modules:
        data = layer(data)
    data = model.representation(data)
    if joint:
        energy, _ = panel_energy_charge(output_head(model), data)
    else:
        energy = data[p.energy]
    force = None
    if forces:
        force = -torch.autograd.grad(
            energy.sum(), data[p.positions], create_graph=training
        )[0]
    return energy, force


def pure_update(model, trainer, batch):
    batch = tree_to_device(batch, "cuda")
    trainer.physics_optimizer.zero_grad(set_to_none=True)
    energy, force = compute(model, batch, False, True, True)
    target = batch["variational"]["labels"]
    e = (
        (
            (energy.flatten() - target["condition_energy"]["values"].flatten())
            / batch[p.n_atoms]
        )
        .square()
        .mean()
    )
    f = (
        (force.reshape(-1) - target["condition_forces"]["values"].reshape(-1))
        .square()
        .mean()
    )
    (200.0 * (e + f)).backward()
    torch.nn.utils.clip_grad_norm_(trainer.backbone, 10.0, error_if_nonfinite=True)
    trainer.physics_optimizer.step()


def timed(operation, iterations):
    times = []
    for i in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        operation(i)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return dict(median_ms=median(times[5:]), mean_ms=mean(times[5:]), steps_ms=times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trajectory", "split", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batch-size", dest="batch", type=int, default=4)
    parser.add_argument("--fp32-training-only", action="store_true")
    args = parser.parse_args()
    args.seed = 3407
    args.panel = HERE / "pure_local/source_indices.json"
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    # Leave float32 matmul strictly FP32 for the precision comparison.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ck = torch.load(HERE / "sabs_warm/final.pt", map_location="cpu", weights_only=False)
    with TemporaryDirectory() as tmp:
        data = data_panel(args, Path(tmp))
    samples = data.datasets["train"]
    order = torch.randperm(
        len(samples), generator=torch.Generator().manual_seed(args.seed)
    ).tolist()
    collate = data.train_dataloader().collate_fn
    if args.batch <= 0 or len(samples) % args.batch:
        parser.error("The training panel must contain whole equal-sized batches.")
    batches = [
        collate([samples[j] for j in order[i : i + args.batch]])
        for i in range(0, len(order), args.batch)
    ]
    results = dict(
        protocol=f"Nonperiodic SA-BS {args.batch} x 13 atoms, zero field, fixed Q=-1, same split/order and identical Small representation weights at initialization, CPU threads=2, CUDA deterministic, TF32 disabled. Inference excludes CPU collation, moves the batch to GPU and rebuilds neighbors, with no parameter gradients or stationarity diagnostics. Training includes CPU collation, transfer, neighbor rebuild, Adam E/F updates, with SOAP stationarity updates in ACVR only. Same EF weighting, physical optimizer and norm clipping, independent warmup. Exclude first 5 steps. No checkpoint changes.",
        device=torch.cuda.get_device_name(),
        results={},
    )

    def cast(value, dtype):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.to(dtype)
        if isinstance(value, dict):
            return {key: cast(v, dtype) for key, v in value.items()}
        if isinstance(value, list):
            return [cast(v, dtype) for v in value]
        if isinstance(value, tuple):
            return tuple(cast(v, dtype) for v in value)
        return value

    def make(dtype):
        model = build_model(**ck["model_options"])
        model.load_state_dict(ck["model"], strict=True)
        model = model.to(device="cuda", dtype=dtype)
        output_head(model).forward = MethodType(panel_forward, output_head(model))
        return model

    parity = {}
    dtypes = (
        (torch.float32,) if args.fp32_training_only else (torch.float64, torch.float32)
    )
    for dtype in dtypes:
        name = str(dtype).split(".")[-1]
        selected = (
            batches
            if dtype == torch.float64
            else [cast(batch, dtype) for batch in batches]
        )
        values = []
        for joint in (False, True):
            tag = "acvr" if joint else "pure"
            model = make(dtype).requires_grad_(False).eval()
            for forces in () if args.fp32_training_only else (False, True):

                def inference(i):
                    batch = tree_to_device(selected[i % len(selected)], "cuda")
                    with torch.set_grad_enabled(forces):
                        return compute(model, batch, joint, forces, False)

                key = f"{name}_{tag}_" + ("EF" if forces else "E")
                result = timed(inference, 23)
                print(key, result["median_ms"], flush=True)
                results["results"][key] = result
            if joint and not args.fp32_training_only:
                model.requires_grad_(True).train()
                t = ScratchTrainer(
                    model,
                    backbone_lr=1e-4,
                    energy_lr=1e-5,
                    proposal_lr=1e-4,
                    energy_weight=200.0,
                    force_weight=200.0,
                )
                for i in (3, 7, 11):
                    i %= len(selected)
                    batch = tree_to_device(selected[i], "cuda")
                    v = model(batch)["variational_prediction"]
                    labs = v["labels"]
                    r = v["residuals"]["stationarity"]["values"]
                    loss = t.supervision({"variational_prediction": v}, batch)["loss"]
                    vs = [
                        labs[k]["values"]
                        for k in (
                            "condition_energy",
                            "condition_forces",
                            "variable_value",
                        )
                    ] + [r]
                    for obj, group in (
                        (loss, t.physical),
                        (30 * r.square().mean(), t.proposal),
                    ):
                        gs = torch.autograd.grad(
                            obj, group, allow_unused=True, retain_graph=True
                        )
                        vs.append(
                            torch.cat(
                                [
                                    (
                                        g if g is not None else torch.zeros_like(z)
                                    ).flatten()
                                    for g, z in zip(gs, group)
                                ]
                            )
                        )
                    values.append([x.detach().cpu().double() for x in vs])
                del t
            # Same local checkpoint weights for both training timings.
            model = make(dtype).train()
            trainer = ScratchTrainer(
                model,
                backbone_lr=1e-4,
                energy_lr=1e-5,
                proposal_lr=1e-4,
                energy_weight=200.0,
                force_weight=200.0,
            )

            def update(i):
                start = (i % len(batches)) * args.batch
                ids = order[start : start + args.batch]
                batch = collate([samples[j] for j in ids])
                if dtype != torch.float64:
                    batch = cast(batch, dtype)
                if joint:
                    return trainer.step(batch, "joint")
                return pure_update(model, trainer, batch)

            key = f"{name}_{tag}_train"
            result = timed(update, 23)
            print(key, result["median_ms"], flush=True)
            results["results"][key] = result
            del trainer, model
            write_json(args.output, results)
        parity[name] = values
    if args.fp32_training_only:
        return
    check = {}
    for j, key in enumerate(
        ("E_eV", "F_eV_A", "q_e", "stationarity_V", "physics_grads", "proposal_grads")
    ):
        x = torch.cat([v[j].reshape(-1) for v in parity["float64"]])
        y = torch.cat([v[j].reshape(-1) for v in parity["float32"]])
        check[key] = dict(
            max_abs=float((x - y).abs().max()),
            relative_L2=float((x - y).norm() / x.norm()),
            RMSE=float((x - y).square().mean().sqrt()),
        )
    results[f"fp32_against_fp64_{3 * args.batch}_frames"] = check
    write_json(args.output, results)


if __name__ == "__main__":
    main()
