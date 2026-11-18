"""Score original held-out folds from saved DPA4C checkpoints with fixed Newton panels."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def write_json(path, obj):
    payload = json.dumps(obj, allow_nan=False, indent=2) + "\n"
    with NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as f:
        f.write(payload)
        name = f.name
    os.replace(name, path)


def panel(ordered_ids, count):
    n = min(len(ordered_ids), count)
    if n < 2:
        return ordered_ids[:n]
    return [ordered_ids[round(i * (len(ordered_ids)-1)/(n-1))] for i in range(n)]


def run(args):
    import torch
    from train import load_data, make_model, run_dir
    from evaluation import evaluate, load_sasa_dipoles

    dest = Path(run_dir(args.system, args.seed, args.arm))
    final = dest / f"ckpt-{args.joint_steps:06d}.pt"
    if not final.is_file():
        raise FileNotFoundError(f"Final checkpoint required for {args.stage} scoring: {final}")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", 2)) // 2))
    cpu = torch.device("cpu")
    device = torch.device("cuda") if torch.cuda.is_available() else cpu
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    curve_stage = args.stage == "validation"
    checkpoints = [final]
    if curve_stage:
        checkpoints = sorted((p for p in dest.glob("ckpt-??????.pt")
                              if int(p.stem[5:]) <= args.joint_steps), key=lambda p: p.name)
        if (dest / "initial.pt").is_file():
            checkpoints.insert(0, dest / "initial.pt")
        checkpoints.extend(sorted(dest.glob("wall-??????.pt")))
    data, splits = load_data(args.system)
    fold = splits[args.stage]
    ids = sorted(map(int, fold))
    slots = {int(source_id): i for i, source_id in enumerate(fold)}
    heldout = data.datasets[args.stage]
    if len(ids) != len(slots) or len(ids) != len(heldout) or not ids:
        raise ValueError(f"{args.stage} IDs do not match the prepared dataset")
    collate = data.val_dataloader().collate_fn
    pure = args.arm in ("A0", "B0")
    labels = load_sasa_dipoles() if args.system == "sasa" and not pure else None
    curve = []
    wall_points = {}
    final_point = None
    model = None

    def batches(which):
        for idx in which:
            item = heldout[slots[idx]]
            if int(item.meta["source_index"]) != idx:
                raise ValueError(f"Prepared source ID disagrees with {args.stage} split at {idx}")
            yield idx, collate([item])

    for checkpoint in checkpoints:
        sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        evaluations = []
        if curve_stage:
            curve_ids = panel(ids, 32)
            point = dest / f"validation_{checkpoint.stem}.json"
            evaluations.append((point, curve_ids, panel(curve_ids, 8)))
        if checkpoint == final:
            evaluations.append((dest / (args.stage + "_final.json"), ids, panel(ids, 32)))
        loaded = None
        for output, frame_ids, newton_ids in evaluations:
            if pure:
                newton_ids = []
            identity = dict(system=args.system, seed=args.seed, arm=args.arm, split=args.stage,
                            checkpoint=checkpoint.name, checkpoint_sha256=sha,
                            newton_source_ids=newton_ids)
            existing = None
            if output.exists():
                try:
                    previous = json.loads(output.read_text())
                    if (all(previous.get(k) == v for k, v in identity.items())
                            and [r["source_index"] for r in previous["frames"]] == frame_ids
                            and previous.get("gpu")
                            and previous["summary"]["frames"] == len(frame_ids)):
                        existing = previous
                except (ValueError, KeyError, TypeError):
                    pass
            if existing is None:
                if model is None:
                    model = make_model(args.system, args.arm, args.seed).double().to(device).eval()
                if loaded is None:
                    loaded = torch.load(checkpoint, map_location=cpu, weights_only=False)
                    model.load_state_dict(loaded["model"], strict=True)
                print(f"Evaluating {args.system}/{args.seed}/{args.arm} {checkpoint.name} "
                      f"on {len(frame_ids)} {args.stage} structures with {len(newton_ids)} Newton solves",
                      flush=True)
                # Expose just the eligible evaluation fold's labels to the scorer.
                refs = {i: labels[i] for i in frame_ids if i in labels} if labels is not None else None
                existing = evaluate(model, batches(frame_ids), pure=pure, newton_ids=newton_ids,
                                    dipoles_D=refs, device=device)
                existing.update(identity, ef_step=int(loaded["ef_step"]),
                                training_seconds=float(loaded["training_seconds"]),
                                gpu=str(loaded["gpu"]), joint_step=int(loaded["joint_step"]))
                write_json(output, existing)
            if curve_stage and output == point:
                snapshot = {k: existing[k] for k in ("checkpoint", "checkpoint_sha256", "ef_step",
                                                     "joint_step", "training_seconds", "gpu", "summary")}
                if checkpoint.stem.startswith("wall-"):
                    cutoff = int(checkpoint.stem[5:])
                    if snapshot["training_seconds"] < cutoff:
                        raise ValueError(f"Wall checkpoint claims to reach {cutoff}s at {snapshot['training_seconds']}s")
                    wall_points[cutoff] = dict(snapshot, budget_seconds=cutoff, status="reached_budget")
                else:
                    curve.append(snapshot)
                    write_json(dest / "validation_curve.json",
                               dict(system=args.system, arm=args.arm, seed=args.seed, split=args.stage,
                                    source_ids=frame_ids, newton_source_ids=newton_ids, points=curve))
                    if checkpoint == final:
                        final_point = snapshot
    if curve_stage:
        from train import WALL_CHECKPOINTS
        for cutoff in WALL_CHECKPOINTS:
            if cutoff not in wall_points:
                if final_point["training_seconds"] > cutoff:
                    wall_points[cutoff] = dict(budget_seconds=cutoff, status="missing_wall_checkpoint",
                                               gpu=final_point["gpu"])
                else:
                    wall_points[cutoff] = dict(final_point, budget_seconds=cutoff, status="finished_within_budget")
        write_json(dest / "validation_wall_curve.json",
                   dict(system=args.system, arm=args.arm, seed=args.seed, split=args.stage,
                        source_ids=panel(ids, 32), newton_source_ids=[] if pure else panel(panel(ids, 32), 8),
                        points=[wall_points[t] for t in sorted(wall_points)]))
    print(f"Completed {args.stage} for {args.system}/{args.seed}/{args.arm}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("aumgo", "sasa"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--stage", choices=("validation", "test"), required=True)
    parser.add_argument("--joint-steps", type=int, default=20000)
    args = parser.parse_args()
    prefix = {"aumgo": "A", "sasa": "B"}[args.system]
    if args.arm not in [prefix + str(i) for i in range(5)] or args.joint_steps < 1:
        parser.error("Arm must belong to the specified system and joint-steps must be positive")
    run(args)
