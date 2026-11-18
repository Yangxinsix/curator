"""Numerical and timing probe for the 4 x 13 atom SA-BS, zero-field, fixed-Q panel.

Run with --trajectory, --split, --reference and --output pointing to the original
SA-BS sources and a fresh JSON output. This is a specialist training benchmark.
"""

import argparse
import os
import statistics
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from train_pilot import HERE, ScratchTrainer, build_model, data_panel
from common import write_json
from curator.data import properties as p
from curator.train.train import tree_to_device
from cache import output_head


from fixed_q_batch import panel_forward


def test(mode, setup, get):
    m, h, t = setup()
    if mode == "batch":
        h.forward = MethodType(panel_forward, h)
    out = []
    for idx in [3, 7, 11]:
        b = tree_to_device(get(idx), "cuda")
        v = m(b)["variational_prediction"]
        loss = t.supervision({"variational_prediction": v}, b)["loss"]
        r = v["residuals"]["stationarity"]["values"]
        groups = []
        for obj, params in [(loss, t.physical), (30 * r.square().mean(), t.proposal)]:
            g = torch.autograd.grad(obj, params, allow_unused=True, retain_graph=True)
            groups.append(
                torch.cat(
                    [
                        (torch.zeros_like(z) if x is None else x)
                        .detach()
                        .cpu()
                        .reshape(-1)
                        for x, z in zip(g, params)
                    ]
                )
            )
        out.append(
            [
                v["labels"][k]["values"].detach().cpu()
                for k in ("condition_energy", "condition_forces", "variable_value")
            ]
            + [r.detach().cpu()]
            + groups
        )
    return out


def bench(mode, setup, get):
    m, h, t = setup()
    if mode == "batch":
        h.forward = MethodType(panel_forward, h)
    times = []
    for i in range(17):
        torch.cuda.synchronize()
        start = time.perf_counter()
        t.step(get(i), "joint")
        torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - start))
    return {
        "times_ms": times,
        "median_ms": statistics.median(times[5:]),
        "mean_ms": statistics.mean(times[5:]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trajectory", "split", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.seed, args.batch = 3407, 4
    args.panel = HERE / "pure_local/source_indices.json"
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with TemporaryDirectory() as directory:
        data = data_panel(args, Path(directory))
    samples = data.datasets["train"]
    # data_panel uses only a single fixed-total-charge recipe, ASE energies/forces
    # and no applied field. This probe deliberately targets the known SA-BS fold.
    if any(
        float(samples[i].atoms[p.total_charge]) != -1
        or int(samples[i].atoms[p.n_atoms]) != 13
        for i in range(len(samples))
    ):
        raise ValueError("Expected the original SA-BS fixed-Q=-1, 13-atom panel")
    checkpoint = torch.load(
        HERE / "sabs_warm/final.pt", weights_only=False, map_location="cpu"
    )
    order = torch.randperm(
        len(samples), generator=torch.Generator().manual_seed(args.seed)
    ).split(args.batch)
    collate = data.train_dataloader().collate_fn

    def get(i):
        return collate([samples[j] for j in order[i % len(order)].tolist()])

    def setup():
        model = build_model(**checkpoint["model_options"])
        model.load_state_dict(checkpoint["model"], strict=True)
        model = model.cuda()
        trainer = ScratchTrainer(
            model,
            backbone_lr=1e-4,
            energy_lr=1e-5,
            proposal_lr=1e-4,
            energy_weight=200.0,
            force_weight=200.0,
        )
        return model, output_head(model), trainer

    ref = test("original", setup, get)
    pred = test("batch", setup, get)
    diffs = {}
    for k, name in enumerate(
        [
            "energy",
            "force",
            "charge",
            "stationarity",
            "physical_parameter_gradient",
            "proposal_parameter_gradient",
        ]
    ):
        x = torch.cat([a[k].flatten() for a in ref])
        y = torch.cat([a[k].flatten() for a in pred])
        diffs[name] = dict(
            max_abs=float((x - y).abs().max()), rel_L2=float((x - y).norm() / x.norm())
        )
        torch.testing.assert_close(x, y, atol=1e-9, rtol=1e-9)
    print(diffs, flush=True)
    results = {}
    for mode in ("original", "batch"):
        results[mode] = bench(mode, setup, get)
        print(mode, results[mode], flush=True)
    write_json(
        args.output,
        dict(
            protocol="Isolated prototype on 4x13 SA-BS nonperiodic, fixed net charge -1, zero voltage, FP64 RTX PRO 4000 Blackwell; identical starting checkpoint, effective batch, two loss routes and optimizers. Full training update including collation, 17 updates, first 5 omitted. Parity at shared weights across three disjoint batches (12 training frames). No weights saved. Native per-geometry condition compiler/packaging is specialized for this panel in the prototype; batched factorized attention, reuse live coefficients, one joint positional/charge derivative, no atom-square kernel.",
            differences=diffs,
            timing=results,
        ),
    )
