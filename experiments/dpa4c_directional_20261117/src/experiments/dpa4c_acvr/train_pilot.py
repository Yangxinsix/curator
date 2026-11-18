"""Train and validate DPA4C-Small + ACVR on the saved SA-BS training and validation panels."""
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from ase.data import atomic_numbers
from ase.io.trajectory import Trajectory

from model import ScratchTrainer, build_model, calibrate_training_inputs
from common import sha, write_json
from curator.data import properties
from equilibrium import _forward, _read_only, model_inputs
from mixed_data import build_data

HERE = Path(__file__).resolve().parent


def data_panel(args, directory):
    split = json.loads(args.split.read_text())
    ids = json.loads(args.panel.read_text())
    offset = 0
    local_split = {"train": [], "validation": [], "test": []}
    with Trajectory(args.trajectory) as source, Trajectory(directory / "subset.traj", "w") as dest:
        for key in ("train", "validation"):
            valid_ids = set(split[key])
            if len(set(ids[key])) != len(ids[key]) or not set(ids[key]) <= valid_ids:
                raise ValueError(f"{key} IDs do not match the original split")
            for idx in ids[key]:
                dest.write(source[idx])
            local_split[key] = list(range(offset, offset + len(ids[key])))
            offset += len(ids[key])
    write_json(directory / "split.json", local_split)
    src = dict(datapath="subset.traj", split_file="split.json",
               boundary=dict(type="fixed_charge", total_charge_key="total_charge"),
               energy_reference_file=str(args.reference.resolve()))
    return build_data(dict(sources=[src], default_dtype=getattr(args,"dtype","float64"), energy_convention="internal",
                           batch_size=args.batch, val_batch_size=args.batch, seed=args.seed), base_dir=directory)


def snapshot(groups):
    return {k: [p.detach().clone() for p in ps] for k, ps in groups.items()}


def deltas(groups, previous):
    return {k: float(torch.stack([(p.detach()-q).abs().max() for p, q in zip(ps, previous[k])]).max())
            for k, ps in groups.items()}


def probe(model, data, device):
    """Native evaluation uses no labels. Only the offline oracle enters converged mode."""
    indices = np.linspace(0, len(data.datasets["validation"])-1,
                          min(8, len(data.datasets["validation"])), dtype=int)
    collate = data.val_dataloader().collate_fn
    rows = []
    with _read_only(model, device) as head:
        for j, index in enumerate(indices):
            sample = data.datasets["validation"][int(index)]
            batch = collate([sample])
            calls = []
            hook = head.proposal.register_forward_hook(lambda *args: calls.append(1))
            try:
                pred = _forward(model, head, batch, device, "proposal")
            finally:
                hook.remove()
            if len(calls) != 1:
                raise RuntimeError(f"Expected one proposal evaluation, saw {len(calls)}")
            exact = _forward(model, head, batch, device, "equilibrium")
            delta = (pred["q"]-exact["q"]).square().mean().sqrt()
            transfer = (exact["q"]-exact["q"].mean()).square().mean().sqrt()
            row = dict(validation_slot=int(index), proposal_calls=len(calls),
                       max_constraint_e=pred["constraint_max"],
                       max_newton_residual_V=float(exact["stationarity"].abs().max()),
                       transfer_rms_e=float(transfer), q_error_rms_e=float(delta),
                       rel_transfer_percent=float(100*delta/transfer) if transfer > 0 else None)
            if j == 0:
                direction = torch.randn(batch[properties.positions].shape,
                                        generator=torch.Generator().manual_seed(617), dtype=torch.float64)
                direction /= direction.norm()
                eps = 1e-4
                energies = []
                for sign in (-1, 1):
                    moved = model_inputs(batch, "cpu")
                    moved[properties.positions] += sign*eps*direction
                    energies.append(_forward(model, head, moved, device, "proposal")["energy"])
                fd = (energies[1]-energies[0])/(2*eps)
                analytic = float(-(pred["forces"]*direction).sum())
                row["directional_force_check"] = dict(analytic_dE_ds_eV_A=analytic, fd_eV_A=fd,
                                                        absolute_error_eV_A=abs(fd-analytic))
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("trajectory", "split", "reference", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--panel", type=Path, default=HERE/"pure_local/source_indices.json")
    p.add_argument("--backbone", type=Path, help="Small representation state_dict in absolute eV")
    p.add_argument("--backbone-backend", choices=("reference", "auto", "native"), default="reference")
    p.add_argument("--native-proposal-spline", action="store_true")
    p.add_argument("--native-qefa", action="store_true", help="Use differentiable molecular CUDA communication")
    p.add_argument("--cuda-graph-joint", action="store_true", help="Replay eligible SA-BS joint updates with CUDA Graphs")
    p.add_argument("--dtype", choices=("float32","float64"), default="float64")
    p.add_argument("--steps", type=int, nargs=3, default=[16, 24, 96], metavar=("ENERGY", "PROPOSAL", "JOINT"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--batch", type=int, default=4)
    args = p.parse_args()
    if args.cuda_graph_joint and (not args.device.startswith("cuda") or args.backbone_backend!="native"):
        p.error("--cuda-graph-joint requires a CUDA device and --backbone-backend native")
    if min(args.steps) < 1 or args.batch < 1:
        p.error("Steps and batch size must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/"final.pt").exists():
        p.error("Use a new output directory")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    with TemporaryDirectory() as tmp:
        data = data_panel(args, Path(tmp))
    species = data.sources[0].species
    references = json.loads(args.reference.read_text())["per_element_eV"]
    state = None
    if args.backbone:
        state = torch.load(args.backbone, weights_only=True, map_location="cpu")
        # Local Curator weights emit absolute energy. ACVR data subtracts elemental references.
        state["atomic_bias"] -= state["atomic_bias"].new_tensor([references[str(atomic_numbers[z])] for z in species])
    model = build_model(species, seed=args.seed, backbone_state=state,
                        backbone_backend=args.backbone_backend, native_proposal_spline=args.native_proposal_spline,
                        native_qefa=args.native_qefa).to(device=device,dtype=getattr(torch,args.dtype))
    trainer = ScratchTrainer(model, backbone_lr=1e-4, energy_lr=1e-5, proposal_lr=1e-4,
                             energy_weight=200., force_weight=200.)
    if args.cuda_graph_joint:
        from graph_joint import JointGraph
        joint_graph=JointGraph(trainer)
    groups = {"backbone": trainer.backbone, "energy": trainer.energy, "proposal": trainer.proposal}
    train = data.train_dataloader()
    train_it = iter(train)

    def next_batch():
        nonlocal train_it
        try:
            return next(train_it)
        except StopIteration:
            train_it = iter(train)
            return next(train_it)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    val = list(data.val_dataloader())
    results = dict(seed=args.seed, device=torch.cuda.get_device_name(device) if device.type=="cuda" else str(device),
                   dtype=args.dtype, backbone_backend=args.backbone_backend,
                   native_proposal_spline=args.native_proposal_spline,
                   native_qefa=args.native_qefa,
                   batch=args.batch, steps=args.steps, train=len(data.datasets["train"]),
                   validation=len(data.datasets["validation"]), species=species, timing="Wall time for batch collection and optimizer step; synchronized; median/mean exclude first update of each phase.",
                   parameters={k:sum(p.numel() for p in v) for k,v in groups.items()},
                   source_sha256={k:sha(path)
                                  for k,path in vars(args).items() if isinstance(path,Path) and path.is_file()},
                   initial=trainer.validate(val), phases=[])
    for phase, n in zip(("energy_warmup","proposal_warmup","joint"),args.steps):
        if phase == "proposal_warmup":
            calib = data.train_dataloader().collate_fn
            examples = data.datasets["train"]
            batches = [calib([examples[i] for i in range(j, min(j+args.batch,len(examples)))])
                       for j in range(0,min(64,len(examples)),args.batch)]
            calibrate_training_inputs(model, batches)
        prev = snapshot(groups)
        timings = []
        if device.type=="cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for step in range(n):
            sync()
            start = time.perf_counter()
            metrics = joint_graph.step(next_batch(), phase) if args.cuda_graph_joint else trainer.step(next_batch(), phase)
            sync()
            timings.append(time.perf_counter()-start)
            if not np.isfinite(list(metrics.values())).all():
                raise ValueError(f"Nonfinite optimization metrics in {phase}")
        used = timings[1:] or timings
        change = deltas(groups,prev)
        active = {"energy_warmup":{"backbone","energy"},"proposal_warmup":{"proposal"},"joint":set(groups)}[phase]
        for key in groups:
            if (key not in active and change[key] != 0) or (n>=2 and key in active and change[key] == 0):
                raise RuntimeError(f"Unexpected {key} parameter update in {phase}: {change}")
        record = dict(phase=phase, updates=n, last_batch=metrics, max_parameter_changes=change,
                      mean_step_s=float(np.mean(used)), median_step_s=float(np.median(used)),
                      timed_steps_s=timings, peak_cuda_MiB=torch.cuda.max_memory_allocated(device)/2**20
                      if device.type=="cuda" else None, validation=trainer.validate(val))
        results["phases"].append(record)
        print(phase,record["validation"],round(record["mean_step_s"],4), flush=True)
    if args.cuda_graph_joint:
        results["cuda_graph"] = dict(captures=joint_graph.captures,replays=joint_graph.replays,fallbacks=joint_graph.fallbacks)
    if args.dtype=="float32":
        # Newton convergence and finite differences use FP64 at the trained FP32 weights.
        with TemporaryDirectory() as tmp:
            oracle_data=data_panel(argparse.Namespace(**(vars(args)|{"dtype":"float64"})),Path(tmp))
        oracle_options=dict(species=species, seed=args.seed, backbone_backend=args.backbone_backend,
                            native_proposal_spline=args.native_proposal_spline,native_qefa=args.native_qefa)
        oracle_model=build_model(**oracle_options).to(device)
        oracle_model.load_state_dict(model.state_dict())
        results["oracle"] = probe(oracle_model,oracle_data,device)
    else:
        results["oracle"] = probe(model,data,device)
    tmp_path = args.out/"final.tmp"
    torch.save(dict(model_options=dict(species=species, seed=args.seed, native_qefa=args.native_qefa,
                                        native_proposal_spline=args.native_proposal_spline), model=model.state_dict(),
                    per_element_reference_eV=references,steps=args.steps), tmp_path)
    tmp_path.replace(args.out/"final.pt")
    write_json(args.out/"metrics.json", results)


if __name__ == "__main__":
    main()
