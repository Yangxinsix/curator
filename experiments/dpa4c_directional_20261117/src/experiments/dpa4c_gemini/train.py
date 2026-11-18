"""Full-split paired DPA4C-Small screen, including the common pure local warm start."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import MethodType

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dpa4c_acvr"))
import numpy as np
import torch
from torch.utils._pytree import tree_map
from model import build_model, calibrate_training_inputs, ScratchTrainer
from aumgo_arms import AuMgOGraph, install_aumgo_arm
from cache import output_head
from common import sha, write_json
from curator.data import properties as p
from curator.train.train import tree_to_device
from fixed_q_batch import panel_forward
from graph_joint import JointGraph
from mixed_data import build_data

SEED = 3407
BATCH = 4
WARM_BASE = 20000
WARM_EF = 16
WARM_PROPOSAL = 64
JOINT = 20000
FREEZE = 256
WALL_CHECKPOINTS = (300, 600, 900)
SAVES = {0, 64, 256, 512, 1000, 2000, 5000, 10000, 15000, JOINT}
SPECIES = {"aumgo": ["O", "Mg", "Al", "Au"], "sasa": ["H", "O", "S"]}


def paths(system):
    if os.environ.get("DPA_LOCAL_INPUTS"):
        root = Path(os.environ["DPA_LOCAL_INPUTS"])
        if system == "aumgo":
            return tuple(root/"acvr-tracks-nov13"/("aumgo."+suffix)
                         for suffix in ("traj", "split", "ref"))
        return tuple(root/"acvr-q-dipole-1112"/name
                     for name in ("sasa.traj", "sasa-split.json", "sasa-eref.json"))
    root = Path("/data1/home/hzxy05/work/yangxin/acvr")
    if system == "aumgo":
        data = root/"datasets/longrange_benchmarks_20261009/aumgo"
        return data/"all.traj", data/"split.json", root/"experiments/acvr_longrange_20261009/initialization/aumgo/energy_reference.json"
    data = root/"experiments/proposal_scratch_campaign_20261105/derived_inputs/acdb_sa_sa"
    return data/"all.traj", data/"split.json", data/"energy_reference.json"


def load_data(system):
    traj, split_file, ref_file = paths(system)
    split = json.loads(split_file.read_text())
    ref = json.loads(ref_file.read_text())
    if (ref["fit_split"] != "train_only" or not set(ref["fit_indices"]) <= set(split["train"])
            or sha(traj) != ref["source_sha256"] or sha(split_file) != ref["split_sha256"]):
        raise ValueError("Training references or saved split differ from the original benchmark")
    source = dict(datapath=str(traj), split_file=str(split_file), energy_reference_file=str(ref_file),
                  boundary=dict(type="fixed_charge", total_charge_key="total_charge"))
    data = build_data(dict(energy_convention="internal", sources=[source], default_dtype="float64",
                           batch_size=BATCH, val_batch_size=BATCH, seed=SEED))
    if set(data.sources[0].species) != set(SPECIES[system]):
        raise ValueError("Unexpected chemical species")
    return data, split


def make_model(system, arm, seed):
    torch.set_default_dtype(torch.float64)
    periodic = system == "aumgo"
    model = build_model(SPECIES[system], seed=seed, periodic_backend="rsa" if periodic else "legacy",
                        backbone_backend="native", native_proposal_spline=True, native_qefa=not periodic)
    if periodic:
        install_aumgo_arm(model, "A0" if arm == "base" else arm)
    elif arm in ("B2", "B4"):
        energy = output_head(model).internal_energy
        energy.set_scales(driving_scale=2 * energy.driving_scale.item())
    return model


def run_dir(system, seed, arm):
    return Path(os.environ.get("DPA_RUNS", "runs"))/system/str(seed)/arm


def atomic_save(state, file):
    tmp = file.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(file)


def configure():
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class Batches:
    """Each phase has a dedicated deterministic shuffled stream, including across restart."""
    def __init__(self, data, seed):
        self.dataset = data.datasets["train"]
        self.collate = data.train_dataloader().collate_fn
        self.seed = seed
        self.epoch = -1

    def get(self, step):
        n = len(self.dataset)
        start = step * BATCH
        ids = []
        for i in range(start, start+BATCH):
            epoch, offset = divmod(i, n)
            if epoch != self.epoch:
                self.order = torch.randperm(n, generator=torch.Generator().manual_seed(self.seed+epoch)).tolist()
                self.epoch = epoch
            ids.append(self.order[offset])
        return tree_map(lambda x: x.float() if torch.is_tensor(x) and x.is_floating_point() else x,
                        self.collate([self.dataset[i] for i in ids]))


def train(args):
    configure()
    system, seed, arm = args.system, args.seed, args.arm
    family = "A" if system == "aumgo" else "B"
    if arm != "base" and arm not in [family+str(i) for i in range(5)]:
        raise ValueError("Arm and system differ")
    pure = arm in ("base", family+"0")
    out = run_dir(system, seed, arm)
    out.mkdir(parents=True, exist_ok=True)
    if (out/"done.json").is_file():
        return
    print("loading", system, seed, arm, torch.cuda.get_device_name(), flush=True)
    data, split = load_data(system)
    model = make_model(system, arm, seed).cuda().float()
    origin = run_dir(system, SEED, "base")/"backbone.pt"
    if arm != "base":
        model.representation.load_state_dict(torch.load(origin, weights_only=True, map_location="cpu"))
    head = output_head(model)
    trainer = ScratchTrainer(model, backbone_lr=1e-4, energy_lr=1e-5, proposal_lr=1e-4,
                             energy_weight=200., force_weight=200.)
    # Match the prior periodic experiment's spectrum LR in every learnable-gain arm.
    if arm in ("A2","A3","A4"):
        name = "band_coefficients" if arm == "A4" else "group_log_gain"
        gain = getattr(head.internal_energy.attention, name)
        group = trainer.physics_optimizer.param_groups[1]
        group["params"] = [p for p in group["params"] if p is not gain]
        trainer.physics_optimizer.add_param_group(dict(params=[gain], lr=0.003))
    graph = (AuMgOGraph if system=="aumgo" else JointGraph)(trainer, pure=pure)
    budget = args.base_steps if arm=="base" else args.joint_steps
    resume_file = out/"resume.pt"
    done = 0
    elapsed = 0.
    if resume_file.exists():
        state = torch.load(resume_file, map_location="cpu", weights_only=False)
        if state["identity"] != [system, seed, arm, budget]:
            raise ValueError("Resume budget or identity changed")
        model.load_state_dict(state["model"])
        trainer.physics_optimizer.load_state_dict(state["adam"])
        trainer.solver_optimizer.load_state_dict(state["soap"])
        done, elapsed = state["step"], state["seconds"]
    def save_model(filename, step, seconds):
        atomic_save(dict(model=model.state_dict(), system=system, arm=arm, seed=seed,
                         joint_step=step, ef_step=step+(WARM_EF if arm!="base" else 0), gpu=torch.cuda.get_device_name(),
                         training_seconds=seconds), out/filename)
    steps = WARM_BASE if arm=="base" else JOINT
    config = dict(system=system, seed=seed, arm=arm, batch=BATCH, dtype="float32", gpu=torch.cuda.get_device_name(),
                  pure_base_seed=SEED, base_sha256=sha(origin) if arm!="base" else None,
                  train_split_sha256=sha(paths(system)[1]), split_sizes={k:len(v) for k,v in split.items()},
                  budget=budget, ef_warmup=0 if arm=="base" else WARM_EF,
                  soap_warmup=WARM_PROPOSAL if not pure else 0,
                  frozen_joint_updates=FREEZE if arm in ("B3","B4") else 0,
                  energy_lr=1e-5, backbone_lr=1e-4, proposal_lr=1e-4,
                  spectrum_lr=0.003 if arm in ("A2","A3","A4") else None,
                  expected_default_steps=steps)
    write_json(out/"config.json", config)
    if resume_file.exists():
        head.forward = MethodType(panel_forward, head)
    else:
        start = time.monotonic()
        if arm == "base":
            # Before the DPA calibration the emitted descriptor is the raw descriptor.
            stream = Batches(data, SEED+70000)
            descriptors = []
            with torch.no_grad():
                for i in range(64//BATCH):
                    batch = tree_to_device(stream.get(i), "cuda")
                    for module in model.input_modules:
                        batch = module(batch)
                    descriptors.append(model.representation(batch)[p.node_feat].detach().double())
                features = torch.cat(descriptors)
                std, mean = torch.std_mean(features, dim=0, unbiased=False)
                model.representation.set_calibration(mean, torch.where(std>=1e-3,std,torch.ones_like(std)))
        else:
            save_model("initial.pt", -WARM_EF, 0.)
            ef = Batches(data, seed+110000)
            for i in range(WARM_EF):
                if pure:
                    graph.step(ef.get(i),"joint")
                else:
                    trainer.step(ef.get(i), "energy_warmup")
            if not pure:
                calib = Batches(data, SEED+70000)
                calibrate_training_inputs(model, [calib.get(i) for i in range(64//BATCH)])
                # Use the batched head only after all input-calibration hooks have fired.
                head.forward = MethodType(panel_forward, head)
                solver = Batches(data, seed+160000)
                for i in range(WARM_PROPOSAL):
                    trainer.step(solver.get(i), "proposal_warmup")
        head.forward = MethodType(panel_forward, head)
        torch.cuda.synchronize()
        elapsed = time.monotonic()-start
        if arm!="base":
            save_model("ckpt-000000.pt",0,elapsed)
    stream = Batches(data, seed+(20000 if arm=="base" else 300000))
    started = time.monotonic()
    def seconds():
        return elapsed+time.monotonic()-started
    wall_pending = [cutoff for cutoff in WALL_CHECKPOINTS
                   if not (out/f"wall-{cutoff:06d}.pt").is_file() and arm != "base"]
    for step in range(done+1,budget+1):
        batch = stream.get(step-1)
        metrics = graph.step(batch, "joint", freeze_backbone=arm in ("B3","B4") and step<=FREEZE)
        if graph.fallbacks or not all(np.isfinite(v) for v in metrics.values()):
            raise RuntimeError(f"Invalid optimized step at {step}: {metrics}, fallbacks={graph.fallbacks}")
        if step==1 or step%100==0 or step==budget:
            row = dict(step=step, training_seconds=seconds(), captures=graph.captures,
                       replays=graph.replays, **metrics)
            if pure:
                row.pop("stationarity_rms_V", None)
            with (out/"training.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False)+"\n")
            write_json(out/"status.json", row)
            if step<=100 or step%1000==0:
                print(system, arm, json.dumps(row), flush=True)
        if wall_pending and seconds() >= wall_pending[0]:
            cutoff = wall_pending.pop(0)
            save_model(f"wall-{cutoff:06d}.pt", step, seconds())
        if arm!="base" and (step in SAVES or step==budget):
            save_model(f"ckpt-{step:06d}.pt",step,seconds())
        if step%1000==0 or step==budget:
            atomic_save(dict(identity=[system,seed,arm,budget], model=model.state_dict(),
                             adam=trainer.physics_optimizer.state_dict(), soap=trainer.solver_optimizer.state_dict(),
                             step=step, seconds=seconds()), resume_file)
    if arm=="base":
        atomic_save(model.representation.state_dict(), out/"backbone.pt")
    write_json(out/"done.json", dict(updates=budget, training_seconds=seconds(), captures=graph.captures,
                                    replays=graph.replays, fallbacks=graph.fallbacks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=tuple(SPECIES))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--base-steps", type=int, default=WARM_BASE)
    parser.add_argument("--joint-steps", type=int, default=JOINT)
    train(parser.parse_args())
