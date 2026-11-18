"""Train or evaluate fixed DPA4C molecular winners with pure and official LES controls."""

import argparse
import json
import os
from pathlib import Path
import time

import torch

from extension_molecular import (
    build_les_model,
    build_model,
    length_groups,
    p,
    physical_ef,
    physical_optimizer,
    tree_to_device,
    update,
)
from train import (
    BATCH,
    FREEZE,
    JOINT,
    SEED,
    WARM_BASE,
    WARM_EF,
    WARM_PROPOSAL,
    SAVES,
    ScratchTrainer,
    atomic_save,
    calibrate_training_inputs,
    configure,
    run_dir,
)
from common import sha
from mixed_data import build_data
from check_small_native import cast
from graph_joint import JointGraph
from evaluation import evaluate
from equilibrium import _read_only, _forward
from extension_dipoles import references, dipole, aggregate
from score import panel, write_json

SOURCES = dict(
    nacl="nacl",
    sabs="acdb_sa_bs",
    cumulene="efa_cumulene",
    dimers="efa_dimers",
    sadma="acdb_sa_dma",
)


def inputs(tag):
    name = SOURCES[tag]
    if local := os.environ.get("DPA_LOCAL_INPUTS"):
        tmp = Path(local)
        if tag == "nacl":
            return tuple(
                tmp / "acvr-nacl-sasa-nov13" / s
                for s in ("nacl.traj", "nacl-split.json", "nacl-eref.json")
            )
        return tuple(
            tmp / "acvr-tracks-nov13" / (name + "." + s)
            for s in ("traj", "split", "ref")
        )
    root = Path("/data1/home/hzxy05/work/yangxin/acvr")
    if tag in ("sabs", "sadma"):
        base = (
            root
            / "experiments/proposal_scratch_campaign_20261105/derived_inputs"
            / name
        )
        return base / "all.traj", base / "split.json", base / "energy_reference.json"
    base = root / "datasets/longrange_benchmarks_20261009" / name
    return (
        base / "all.traj",
        base / "split.json",
        root
        / "experiments/acvr_longrange_20261009/initialization"
        / name
        / "energy_reference.json",
    )


class Stream:
    """Return the *same* original four examples regardless of the model's shape grouping."""

    def __init__(self, data, seed):
        self.dataset = data.datasets["train"]
        self.seed = seed
        self.epoch = None
        self.collate = data.train_dataloader().collate_fn

    def get(self, step):
        samples = []
        for i in range(step * BATCH, (step + 1) * BATCH):
            epoch, offset = divmod(i, len(self.dataset))
            if epoch != self.epoch:
                self.perm = torch.randperm(
                    len(self.dataset),
                    generator=torch.Generator().manual_seed(self.seed + epoch),
                ).tolist()
                self.epoch = epoch
            samples.append(self.dataset[self.perm[offset]])
        return [cast(b, torch.float32) for b in length_groups(samples, self.collate)]


def build(tag, species, arm, seed, selection):
    if arm == "les":
        return build_les_model(species, seed=seed)
    model = build_model(
        species,
        seed=seed,
        backbone_backend="native",
        native_qefa=True,
        native_proposal_spline=True,
    )
    if arm == "acvr" and selection in ("B2", "B4"):
        en = model.output_modules[0].internal_energy
        en.set_scales(driving_scale=2 * en.driving_scale.item())
    return model


def fit(args, data, model, out, winner):
    base = args.arm == "base"
    acvr = args.arm == "acvr"
    method = "pure" if base else args.arm
    steps = args.base_steps if base else args.joint_steps
    if base and args.seed != SEED:
        raise ValueError("Each system has one common training-only seed-3407 backbone")
    origin = run_dir(args.system, SEED, "base") / "backbone.pt"
    identity = dict(
        system=args.system,
        arm=args.arm,
        seed=args.seed,
        budget=steps,
        winner=winner,
        split_sha256=sha(inputs(args.system)[1]),
        base_sha256=sha(origin) if not base else None,
    )
    if (out / "done.json").is_file() and json.loads((out / "done.json").read_text())[
        "identity"
    ] == identity:
        return
    if not base:
        model.representation.load_state_dict(
            torch.load(origin, weights_only=True, map_location="cpu")
        )
    graph = None
    if acvr or method == "pure":
        groups = ScratchTrainer(
            model,
            backbone_lr=1e-4,
            energy_lr=1e-5,
            proposal_lr=1e-4,
            energy_weight=200.0,
            force_weight=200.0,
        )
        graph = JointGraph(groups, pure=method == "pure")
        adam = groups.physics_optimizer
        trainer = groups if acvr else adam
        soap = groups.solver_optimizer if acvr else None
    else:
        trainer = adam = physical_optimizer(
            model, method=method, backbone_lr=1e-4, head_lr=1e-5
        )
        soap = None
    resume = out / "resume.pt"
    done = 0
    elapsed = 0.0

    def save_snapshot(file, step, secs):
        atomic_save(
            dict(
                identity=identity,
                model=model.state_dict(),
                joint_step=step,
                training_seconds=secs,
            ),
            file,
        )

    if resume.is_file():
        state = torch.load(resume, weights_only=False, map_location="cpu")
        if state["identity"] != identity:
            raise ValueError("Changed extension experiment identity")
        model.load_state_dict(state["model"])
        adam.load_state_dict(state["adam"])
        if soap:
            soap.load_state_dict(state["soap"])
        done, elapsed = state["step"], state["seconds"]
    else:
        start = time.monotonic()
        calib = Stream(data, SEED + 70000)
        calib_batches = [b for i in range(64 // BATCH) for b in calib.get(i)]
        if base:
            descriptors = []
            with torch.no_grad():
                for batch in calib_batches:
                    batch = tree_to_device(batch, "cuda")
                    for layer in model.input_modules:
                        batch = layer(batch)
                    descriptors.append(
                        model.representation(batch)[p.node_feat].double().detach()
                    )
                std, mean = torch.std_mean(
                    torch.cat(descriptors), dim=0, unbiased=False
                )
                model.representation.set_calibration(
                    mean, torch.where(std >= 1e-3, std, torch.ones_like(std))
                )
        else:
            save_snapshot(out / "initial.pt", -WARM_EF, 0.0)
            ef = Stream(data, args.seed + 110000)
            for i in range(WARM_EF):
                update(
                    model,
                    trainer,
                    ef.get(i),
                    method,
                    phase="energy_warmup" if acvr else "joint",
                )
            if acvr:
                calibrate_training_inputs(model, calib_batches)
                solver = Stream(data, args.seed + 160000)
                for i in range(WARM_PROPOSAL):
                    update(
                        model, trainer, solver.get(i), method, phase="proposal_warmup"
                    )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        if not base:
            save_snapshot(out / "ckpt-000000.pt", 0, elapsed)
    stream = Stream(data, args.seed + (20000 if base else 300000))
    start = time.monotonic()
    write_json(
        out / "config.json",
        dict(
            identity=identity,
            method=method,
            actual_batch=BATCH,
            full_train_frames=len(data.datasets["train"]),
            per_geometry_EF=True,
            stationarity_per_atom=True,
            frozen_joint_steps=FREEZE if acvr and winner in ("B3", "B4") else 0,
            beta_multiplier_at_initialization=2
            if acvr and winner in ("B2", "B4")
            else 1,
            ef_warmup=0 if base else WARM_EF,
            proposal_warmup=WARM_PROPOSAL if acvr else 0,
            backend="native DPA4C, batched factorized molecular ACVR / official monopole LES",
            les_commit="35b971a650cf30adb84fef7138761e81aa47b3a2"
            if method == "les"
            else None,
        ),
    )
    for step in range(done + 1, steps + 1):
        batches = stream.get(step - 1)
        freeze = acvr and winner in ("B3", "B4") and step <= FREEZE
        if graph is not None and len(batches) == 1 and graph.eligible(batches[0]):
            record = graph.step(batches[0], "joint", freeze_backbone=freeze)
            record.update(
                frames=BATCH,
                atoms=int(batches[0][p.n_atoms].sum()),
                length_groups=1,
                graph_replays=graph.replays,
            )
            if not acvr:
                record["stationarity_rms_V"] = None
        else:
            record = update(model, trainer, batches, method, freeze_backbone=freeze)
        sec = elapsed + time.monotonic() - start
        if step == 1 or step % 100 == 0 or step == steps:
            row = dict(step=step, training_seconds=sec, **record)
            with (out / "training.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            write_json(out / "status.json", row)
            if step % 1000 == 0 or step == 1:
                print(json.dumps(row), flush=True)
        if not base and (step in SAVES or step == steps):
            save_snapshot(out / f"ckpt-{step:06d}.pt", step, sec)
        if step % 1000 == 0 or step == steps:
            atomic_save(
                dict(
                    identity=identity,
                    model=model.state_dict(),
                    adam=adam.state_dict(),
                    soap=soap.state_dict() if soap else None,
                    step=step,
                    seconds=sec,
                ),
                resume,
            )
    if base:
        atomic_save(model.representation.state_dict(), out / "backbone.pt")
    write_json(out / "done.json", dict(identity=identity, steps=steps, seconds=sec))


def score(args, data, split, model, out, winner):
    if args.arm == "base":
        raise ValueError("Fully trained control evaluation uses arm=pure")
    fold = args.stage
    want_dipoles = args.system == "sabs" and args.arm in ("acvr", "les")
    ck = out / f"ckpt-{args.joint_steps:06d}.pt"
    digest = sha(ck)
    result = out / (fold + "_final.json")
    if result.is_file():
        d = json.loads(result.read_text())
        if (
            d.get("checkpoint_sha256") == digest
            and d["summary"]["frames"] == len(split[fold])
            and (not want_dipoles or "dipole" in d["summary"])
        ):
            return
    state = torch.load(ck, weights_only=False, map_location="cpu")
    if state["identity"]["winner"] != winner:
        raise ValueError("Changed selected SA-SA ACVR configuration")
    model.load_state_dict(state["model"])
    model.eval()
    collate = data.val_dataloader().collate_fn
    ids = list(map(int, split[fold]))
    refs = references(fold) if want_dipoles else {}
    if want_dipoles and not set(ids) <= refs.keys():
        raise ValueError("Incomplete SA-BS held-out DFT dipole coverage")

    def batches():
        for j, idx in enumerate(ids):
            sample = data.datasets[fold][j]
            if idx != sample.meta["source_index"]:
                raise ValueError("Mismatched frame identity")
            yield idx, collate([sample])

    if args.arm != "les":
        report = evaluate(
            model,
            batches(),
            pure=args.arm == "pure",
            newton_ids=panel(sorted(ids), 32) if args.arm == "acvr" else [],
            device="cuda",
        )
        if want_dipoles:
            with _read_only(model, "cuda") as head:
                for row, (idx, b) in zip(report["frames"], batches(), strict=True):
                    z = _forward(model, head, b, "cuda", "proposal")["q"]
                    row["dipole_D"] = dipole(z, b, idx, refs)
    else:
        rows = []
        for idx, b in batches():
            y = b["variational"]["labels"]
            n = int(b[p.n_atoms])
            e, f, z = physical_ef(
                model, tree_to_device(b, "cuda"), method="les", create_graph=False
            )
            rows.append(
                dict(
                    source_index=idx,
                    atoms=n,
                    les=dict(
                        energy_error_meV_atom=(
                            e.item() - y["condition_energy"]["values"].item()
                        )
                        / n
                        * 1000,
                        force_sse_meV2_A2=float(
                            (
                                (
                                    f.detach().cpu().reshape(-1)
                                    - y["condition_forces"]["values"]
                                )
                                * 1000
                            )
                            .square()
                            .sum()
                        ),
                        latent_charge_sum_e=z.detach().sum().item(),
                    ),
                )
            )
            if want_dipoles:
                rows[-1]["dipole_D"] = dipole(z, b, idx, refs)
        total = sum(r["atoms"] for r in rows)
        report = dict(
            frames=rows,
            summary=dict(
                frames=len(rows),
                atoms=total,
                les=dict(
                    energy_RMSE_meV_atom=(
                        sum(r["les"]["energy_error_meV_atom"] ** 2 for r in rows)
                        / len(rows)
                    )
                    ** 0.5,
                    force_RMSE_meV_A=(
                        sum(r["les"]["force_sse_meV2_A2"] for r in rows) / (3 * total)
                    )
                    ** 0.5,
                ),
            ),
        )
    if want_dipoles:
        report["summary"]["dipole"] = aggregate(report["frames"])
    report.update(
        identity=state["identity"],
        split=fold,
        checkpoint=ck.name,
        checkpoint_sha256=digest,
    )
    write_json(result, report)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", required=True, choices=SOURCES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--arm", choices=("base", "pure", "acvr", "les"), required=True)
    ap.add_argument("--stage", choices=("train", "validation", "test"), default="train")
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--base-steps", type=int, default=WARM_BASE)
    ap.add_argument("--joint-steps", type=int, default=JOINT)
    args = ap.parse_args()
    if min(args.base_steps, args.joint_steps) < 1:
        ap.error("require positive step counts")
    manifest = json.loads(args.selection.read_text())
    if manifest["stage"] != "select-test":
        raise ValueError("Need the frozen validation decision")
    winner = manifest["systems"]["sasa"]["chosen_acvr"]
    if winner not in ("B1", "B2", "B3", "B4"):
        raise ValueError("Invalid molecular configuration")
    configure()
    trajectory, splits, reference = inputs(args.system)
    split = json.loads(splits.read_text())
    ref = json.loads(reference.read_text())
    if (
        ref["source_sha256"] != sha(trajectory)
        or ref["split_sha256"] != sha(splits)
        or ref["fit_split"] != "train_only"
        or not set(ref["fit_indices"]) <= set(split["train"])
    ):
        raise ValueError("Invalid training source or elemental reference")
    ds = build_data(
        dict(
            energy_convention="internal",
            default_dtype="float64",
            batch_size=BATCH,
            sources=[
                dict(
                    datapath=str(trajectory),
                    split_file=str(splits),
                    energy_reference_file=str(reference),
                    boundary=dict(type="fixed_charge", total_charge_key="total_charge"),
                )
            ],
        )
    )
    out = run_dir(args.system, args.seed, args.arm)
    out.mkdir(parents=True, exist_ok=True)
    model = build(
        args.system, ds.sources[0].species, args.arm, args.seed, winner
    ).cuda()
    if args.stage == "train":
        fit(args, ds, model.float(), out, winner)
    else:
        score(args, ds, split, model.double(), out, winner)


if __name__ == "__main__":
    main()
