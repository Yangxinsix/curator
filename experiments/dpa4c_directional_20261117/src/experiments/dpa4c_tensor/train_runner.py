"""Paired continuations of the seven 20k DPA4C + ACVR checkpoints."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dpa4c_gemini"))
import torch
from train import BATCH, JOINT, SEED, AuMgOGraph, Batches, ScratchTrainer, atomic_save, configure
from train import make_model, paths as primary_paths
from extension_run import SOURCES, Stream, build, inputs
from extension_molecular import update
from graph_joint import JointGraph
from mixed_data import build_data
from common import sha
from evaluation import evaluate, load_sasa_dipoles
from extension_dipoles import aggregate, dipole, references
from equilibrium import _forward, _read_only
from score import panel, write_json
from tensor_model import install_variant

SYSTEMS = ("aumgo", "sasa", *SOURCES)
ARMS = ("old", "qk2", "orient2", "orient1", "orient12")
ROOT = Path("/data1/home/hzxy05/work/yangxin/acvr")
DEFAULT_SOURCE = ROOT / "experiments/dpa4c_gemini_20261113/runs"
DEFAULT_SCAN = ROOT / "datasets/longrange_benchmarks_20261009/efa_cumulene/cumulene_profile.traj"


def load_data(system):
    trajectory, splits, reference = inputs(system) if system in SOURCES else primary_paths(system)
    split = json.loads(splits.read_text())
    ref = json.loads(reference.read_text())
    if (sha(trajectory) != ref["source_sha256"] or sha(splits) != ref["split_sha256"]
            or ref["fit_split"] != "train_only" or not set(ref["fit_indices"]) <= set(split["train"])):
        raise ValueError("Dataset or split differs from original train-only reference")
    source = dict(datapath=str(trajectory), split_file=str(splits), energy_reference_file=str(reference),
                  boundary=dict(type="fixed_charge", total_charge_key="total_charge"))
    ds = build_data(dict(energy_convention="internal", default_dtype="float64", batch_size=BATCH,
                         sources=[source]))
    return ds, split, ref["split_sha256"]


def source_arm(system):
    return {"aumgo": "A3", "sasa": "B2"}.get(system, "acvr")


def make_old(system, species):
    if system in SOURCES:
        return build(system, species, "acvr", SEED, "B2")
    return make_model(system, source_arm(system), SEED)


def load_tensor_checkpoint(checkpoint_path, *, device="cuda", dtype=torch.float64):
    """Reconstruct our checkpoint from its recorded model/source identity, without a dataset."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ident = payload["identity"]
    if ident["schema"] != 1 or ident["arm"] not in ARMS or ident["system"] not in SYSTEMS:
        raise ValueError("Unrecognized tensor experiment checkpoint")
    model = make_old(ident["system"], ident["species"]).to(device=device, dtype=dtype)
    install_variant(model, ident["arm"], seed=ident["seed"], exclude_self=True)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval()


def restore_origin(args, ds, split_sha):
    """Load both optimizers on the *original* parameter lists before attaching fresh heads."""
    sys_name = args.system
    arm = source_arm(sys_name)
    dest = args.source_runs / sys_name / str(SEED) / arm
    ckpt, resume = dest / f"ckpt-{JOINT:06d}.pt", dest / "resume.pt"
    source_sha = sha(ckpt)
    test = json.loads((dest / "test_final.json").read_text())
    if test["checkpoint_sha256"] != source_sha:
        raise ValueError(f"Source weight digest disagrees with completed independent-test receipt: {dest}")
    state = torch.load(resume, map_location="cpu", weights_only=False)
    base = torch.load(ckpt, map_location="cpu", weights_only=False)
    if state["step"] != JOINT or base["joint_step"] != JOINT:
        raise ValueError("Expected the final 20k joint-update model and its optimizer state")
    if sys_name in SOURCES:
        src = state["identity"]
        if (sys_name, SEED, arm, JOINT, "B2", split_sha) != (
                src["system"], src["seed"], src["arm"], src["budget"], src["winner"], src["split_sha256"]):
            raise ValueError("Molecular source identity mismatch")
    else:
        cfg = json.loads((dest / "config.json").read_text())
        if (state["identity"] != [sys_name, SEED, arm, JOINT] or cfg["train_split_sha256"] != split_sha):
            raise ValueError("Original A3/B2 source identity mismatch")
    if state["model"].keys() != base["model"].keys() or any(
            not torch.equal(v, base["model"][k]) for k, v in state["model"].items() if torch.is_tensor(v)):
        raise ValueError("Weights in source optimizer checkpoint differ from the verified endpoint")
    species = list(ds.sources[0].species)
    model = make_old(sys_name, species).cuda().float()
    model.load_state_dict(base["model"], strict=True)
    trainer = ScratchTrainer(model, backbone_lr=1e-4, energy_lr=1e-5, proposal_lr=1e-4,
                             energy_weight=200., force_weight=200.)
    adam, soap = trainer.physics_optimizer, trainer.solver_optimizer
    head_group = adam.param_groups[1]
    if sys_name == "aumgo":
        gain = model.output_modules[0].internal_energy.attention.group_log_gain
        head_group["params"] = [p for p in head_group["params"] if p is not gain]
        adam.add_param_group(dict(params=[gain], lr=0.003))
    adam.load_state_dict(state["adam"])
    # SOAP's nested preconditioner state is migrated by the torch optimizer loader.
    soap.load_state_dict(state["soap"])
    original = dict(model.named_parameters())
    install_variant(model, args.arm, seed=args.seed, exclude_self=True)
    current = dict(model.named_parameters())
    if any(current.get(name) is not param for name, param in original.items()):
        raise ValueError("A variant replaced a shared parameter; optimizer continuation is unsafe")
    fresh = [(k, v) for k, v in current.items() if k not in original]
    if bool(fresh) != (args.arm != "old"):
        raise ValueError("Unexpected trainable parameter inventory")
    # Re-read groups after load_state_dict(), which replaces group dictionaries.
    adam.param_groups[1]["params"].extend(v for _, v in fresh)
    trainer.energy.extend(v for _, v in fresh)
    trainer.physical = trainer.energy + trainer.backbone
    physical_ids = [id(p) for g in adam.param_groups for p in g["params"]]
    if sorted(physical_ids) != sorted(map(id, trainer.physical)):
        raise ValueError("Incomplete or duplicated physical parameter")
    ident = dict(schema=1, system=sys_name, arm=args.arm, source_arm=arm, source_seed=SEED,
                 seed=args.seed, species=species, source_joint_step=JOINT, budget=args.steps,
                 split_sha256=split_sha, source_checkpoint_sha256=source_sha, source_resume_sha256=sha(resume))
    return model, trainer, ident, [k for k, _ in fresh]


def fit(args, ds, split_sha, out):
    model, trainer, ident, names = restore_origin(args, ds, split_sha)
    adam, soap = trainer.physics_optimizer, trainer.solver_optimizer
    graph = (AuMgOGraph if args.system == "aumgo" else JointGraph)(trainer)
    full = not (args.system in SOURCES)
    stream = (Batches if full else Stream)(ds, args.seed + 300000)
    done, elapsed = 0, 0.
    resume = out / "resume.pt"
    if resume.exists():
        st = torch.load(resume, weights_only=False, map_location="cpu")
        if st["identity"] != ident:
            raise ValueError("Tensor resume identity mismatch")
        model.load_state_dict(st["model"], strict=True)
        adam.load_state_dict(st["adam"])
        soap.load_state_dict(st["soap"])
        done, elapsed = st["step"], st["seconds"]
        del st
    def snap(file, step, sec):
        atomic_save(dict(identity=ident, model=model.state_dict(), joint_step=JOINT + step,
                         added_steps=step, training_seconds=sec), out / file)
    if done == 0:
        snap("ckpt-000000.pt", 0, 0.)
    write_json(out / "config.json", dict(identity=ident, additional_parameters=names, batch=BATCH,
               optimizer_lrs=[g["lr"] for g in adam.param_groups], proposal_lr=soap.param_groups[0]["lr"],
               loss="200*(E_per_atom_MSE + force_MSE) to physics; 30*stationarity_MSE to proposal",
               training_frames=len(ds.datasets["train"]), warmup=0, original_optimizer_state_restored=True,
               gpu=torch.cuda.get_device_name(), started_stream_at=JOINT, exclude_self=True))
    if done >= args.steps:
        if not (out / "done.json").exists():
            write_json(out / "done.json", dict(identity=ident, steps=done, training_seconds=elapsed))
        return
    model.train()
    wall_points = [s for s in (300, 600, 900) if not (out / f"wall-{s:06d}.pt").exists()]
    start = time.monotonic()
    for step in range(done + 1, args.steps + 1):
        b = stream.get(JOINT + step - 1)
        batches = [b] if full else b
        if len(batches) == 1 and graph.eligible(batches[0]):
            metrics = graph.step(batches[0], "joint")
        elif full:
            raise ValueError("Source fixed-shape graph unexpectedly unsupported")
        else:
            metrics = update(model, trainer, batches, "acvr")
        if step == 1 or step % 100 == 0 or step == args.steps:
            torch.cuda.synchronize()
            secs = elapsed + time.monotonic() - start
            row = dict(step=step, joint_step=JOINT + step, training_seconds=secs, **metrics)
            with (out / "training.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            write_json(out / "status.json", row)
            if step == 1 or step % 1000 == 0:
                print(json.dumps(row), flush=True)
        secs = elapsed + time.monotonic() - start
        if wall_points and secs >= wall_points[0]:
            snap(f"wall-{wall_points.pop(0):06d}.pt", step, secs)
        if step in (1000, 5000, 10000) or step == args.steps:
            snap(f"ckpt-{step:06d}.pt", step, secs)
        if step % 1000 == 0 or step == args.steps:
            atomic_save(dict(identity=ident, model=model.state_dict(), adam=adam.state_dict(),
                             soap=soap.state_dict(), step=step, seconds=secs), resume)
    write_json(out / "done.json", dict(identity=ident, steps=step, training_seconds=secs))


def score(args, ds, split, split_sha, out, fold):
    checkpoint = out / f"ckpt-{args.steps:06d}.pt"
    ids = [int(i) for i in split[fold]]
    slots = {i: slot for slot, i in enumerate(ids)}
    ids = sorted(ids)
    collate = ds.val_dataloader().collate_fn
    def batches(indices):
        for idx in indices:
            sample = ds.datasets[fold][slots[idx]]
            if int(sample.meta["source_index"]) != idx:
                raise ValueError("Held-out source ID mismatch")
            yield idx, collate([sample])
    jobs = [(checkpoint, out / (fold + "_final.json"), ids, 32)]
    if fold == "validation":
        pts = sorted(out.glob("ckpt-*.pt")) + sorted(out.glob("wall-*.pt"))
        jobs += [(c, out / f"validation_{c.stem}.json", panel(ids, 32), 8) for c in pts]
    for ck, dest, frame_ids, n_newton in jobs:
        digest = sha(ck)
        if dest.exists():
            prev = json.loads(dest.read_text())
            if (prev["checkpoint_sha256"] == digest and prev["identity"]["split_sha256"] == split_sha
                    and [r["source_index"] for r in prev["frames"]] == frame_ids):
                continue
        payload = torch.load(ck, weights_only=False, map_location="cpu")
        identity = payload["identity"]
        if (identity["arm"], identity["system"], identity["budget"], identity["seed"], identity["split_sha256"]) != (
            args.arm, args.system, args.steps, args.seed, split_sha):
            raise ValueError("Evaluation identity mismatch")
        model = load_tensor_checkpoint(ck)
        sasa = load_sasa_dipoles() if args.system == "sasa" else None
        if sasa is not None and not set(frame_ids) <= sasa.keys():
            raise ValueError("SA-SA DFT dipole coverage incomplete")
        report = evaluate(model, batches(frame_ids), newton_ids=panel(frame_ids, n_newton),
                          dipoles_D=sasa, device="cuda")
        if args.system == "sabs":
            refs = references(fold)
            if not set(frame_ids) <= refs.keys():
                raise ValueError("SA-BS DFT dipole coverage incomplete")
            with _read_only(model, "cuda") as head:
                for row, (idx, batch) in zip(report["frames"], batches(frame_ids), strict=True):
                    q = _forward(model, head, batch, "cuda", "proposal")["q"]
                    row["dipole_D"] = dipole(q, batch, idx, refs)
            report["summary"]["dipole"] = aggregate(report["frames"])
        report.update(identity=identity, split=fold, checkpoint=ck.name, checkpoint_sha256=digest,
                      joint_step=payload["joint_step"], added_steps=payload["added_steps"],
                      training_seconds=payload["training_seconds"])
        write_json(dest, report)
        print(dest, report["summary"]["proposal"], flush=True)
        del model, payload


def scan(args, out):
    if args.system != "cumulene":
        raise ValueError("The torsion benchmark is defined for cumulene only")
    from scan import score_scan
    files = [out / f"ckpt-{args.steps:06d}.pt"]
    if args.arm == "old":
        files.insert(0, out / "ckpt-000000.pt")
    for ck in files:
        model = load_tensor_checkpoint(ck)
        score_scan(model, trajectory=args.scan_trajectory, out_dir=out / f"torsion_{ck.stem}",
                   checkpoint_path=ck, device="cuda")
        del model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", required=True, choices=SYSTEMS)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--stage", choices=("all", "train", "validation", "test", "scan"), default="all")
    ap.add_argument("--source-runs", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out-root", type=Path, default=Path("tensor_runs"))
    ap.add_argument("--scan-trajectory", type=Path, default=DEFAULT_SCAN)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--steps", type=int, default=JOINT)
    args = ap.parse_args()
    if args.steps < 1:
        ap.error("additional steps must be positive")
    configure()
    out = args.out_root / args.system / str(args.seed) / args.arm
    out.mkdir(parents=True, exist_ok=True)
    ds, split, split_sha = load_data(args.system)
    if args.stage in ("all", "train"):
        fit(args, ds, split_sha, out)
        gc.collect()
        torch.cuda.empty_cache()
    for fold in ("validation", "test"):
        if args.stage in ("all", fold):
            score(args, ds, split, split_sha, out, fold)
    if args.stage == "scan" or (args.stage == "all" and args.system == "cumulene"):
        scan(args, out)
    if args.stage == "all":
        write_json(out / "complete.json", dict(system=args.system, arm=args.arm, seed=args.seed,
               steps=args.steps, independent_test=True, validation=True, torsion=args.system=="cumulene"))


if __name__ == "__main__":
    main()
