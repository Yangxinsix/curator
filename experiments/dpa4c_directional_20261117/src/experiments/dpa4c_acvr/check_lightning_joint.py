"""Verify Lightning manual optimisation reproduces Small+ACVR joint updates."""

import argparse
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType

from train_pilot import HERE, build_model, data_panel
from fixed_q_batch import panel_forward
from check_small_native import cast
from cache import output_head
from common import write_json
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


from lightning_joint import JointTask, make_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trajectory", "reference", "split", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cuda-graph-joint", action="store_true")
    args = parser.parse_args()
    args.panel = HERE / "pure_local/source_indices.json"
    args.seed, args.batch = 3407, 4
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with TemporaryDirectory() as tmp:
        ds = data_panel(args, Path(tmp))
    collate = ds.train_dataloader().collate_fn
    data = ds.datasets["train"]
    ids = torch.randperm(
        len(data), generator=torch.Generator().manual_seed(args.seed)
    ).tolist()
    batches = [
        cast(collate([data[i] for i in ids[start : start + 4]]), torch.float32)
        for start in range(0, 24, 4)
    ]
    ck = torch.load(HERE / "sabs_warm/final.pt", weights_only=False, map_location="cpu")

    def make():
        m = build_model(
            **(
                ck["model_options"]
                | dict(
                    backbone_backend="native",
                    native_proposal_spline=True,
                    native_qefa=args.cuda_graph_joint,
                )
            )
        )
        m.load_state_dict(ck["model"], strict=True)
        output_head(m).forward = MethodType(panel_forward, output_head(m))
        return m.float().cuda()

    reference = make_groups(make())
    initial = {k: v.detach().clone() for k, v in reference.model.named_parameters()}
    if args.cuda_graph_joint:
        from graph_joint import JointGraph
        reference_graph = JointGraph(reference)
    for batch in deepcopy(batches):
        if args.cuda_graph_joint:
            reference_graph.step(batch, "joint")
        else:
            reference.step(batch, "joint")
    graph_model = make()
    if args.cuda_graph_joint:
        del output_head(graph_model).forward  # Use the ordinary model for any fallback.
    task = JointTask(graph_model, cuda_graph=args.cuda_graph_joint)
    fit = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="32-true",
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        deterministic=True,
    )
    fit.fit(
        task,
        train_dataloaders=DataLoader(
            deepcopy(batches), batch_size=None, collate_fn=lambda b: b
        ),
    )
    # Single-device Lightning moves the fitted module back to CPU on teardown.
    task.to(reference.physical[0].device)
    diff = {}
    updates = {}
    for group, params in (
        ("backbone", reference.backbone),
        ("energy", reference.energy),
        ("proposal", reference.proposal),
    ):
        names = [
            k
            for k, p in reference.model.named_parameters()
            if any(p is member for member in params)
        ]
        a = dict(reference.model.named_parameters())
        b = dict(task.model.named_parameters())
        diff[group] = max(float((a[k] - b[k]).detach().abs().max()) for k in names)
        updates[group] = max(
            float((b[k] - initial[k]).detach().abs().max()) for k in names
        )
    print("step comparison", fit.global_step, diff, updates, flush=True)
    assert all(d < 1.0e-6 for d in diff.values()) and all(
        d > 0 for d in updates.values()
    )
    result = dict(
        lightning=pl.__version__,
        batches=len(batches),
        optimizer_steps=fit.global_step,
        device=torch.cuda.get_device_name(),
        dtype="float32",
        max_parameter_difference_vs_original=diff,
        max_parameter_update_vs_input=updates,
        backbone_backend="native",
        native_proposal_spline=True,
        protocol="Single-GPU FP32 manual optimization, same checkpoint and same 6 batches as original joint loop. E/F backward into energy and backbone, stationarity into proposal, disjoint optimizer updates after both gradients, fixed-Q SA-BS one-pass batch path.",
    )
    if args.cuda_graph_joint:
        result["protocol"] += " Lightning graph replay is compared to standalone graph replay; eager equivalence is checked in the separate matched benchmark."
        result["graph"] = dict(
            captures=task.joint_graph.captures, replays=task.joint_graph.replays
        )
        assert task.joint_graph.replays == 6 and fit.global_step == 12
    print(result, flush=True)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
