"""DPA4C+LES E/F and mixed-length ACVR gradients on real NaCl and SA-BS batches."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from extension_molecular import (
    build_model,
    build_les_model,
    update,
    physical_ef,
    physical_optimizer,
    length_groups,
    p,
    tree_to_device,
)
from train_pilot import data_panel
from ase.io.trajectory import Trajectory
from common import write_json
from lightning_joint import make_groups


def check(tag, args):
    split = json.loads(args.split.read_text())
    # Two of each size for NaCl; two independent batches for fixed-size SA-BS.
    chosen = {}
    with Trajectory(args.trajectory) as traj:
        for i in split["train"]:
            size = len(traj[i])
            group = chosen.setdefault(size, [])
            if len(group) < 4:
                group.append(i)
            if (
                tag == "nacl"
                and len(chosen) == 2
                and all(len(ids) >= 2 for ids in chosen.values())
            ):
                break
            if tag == "sabs" and sum(map(len, chosen.values())) == 4:
                break
    train_ids = [i for ids in chosen.values() for i in ids[: 2 if tag == "nacl" else 4]]
    with TemporaryDirectory() as tmp:
        args.panel = Path(tmp) / "panel.json"
        args.panel.write_text(
            json.dumps(dict(train=train_ids, validation=split["validation"][:1]))
        )
        ds = data_panel(args, Path(tmp))
    train = list(ds.datasets["train"])
    collate = ds.train_dataloader().collate_fn
    species = ds.sources[0].species
    grouped = length_groups(train, collate)
    mixed = collate(train)
    torch.set_default_dtype(torch.float64)
    base = build_model(
        species,
        seed=args.seed,
        backbone_backend="native",
        native_proposal_spline=True,
        native_qefa=True,
    ).cuda()
    base_state = {
        k: v.detach().cpu().clone() if torch.is_tensor(v) else v
        for k, v in base.representation.state_dict().items()
    }
    ref = build_model(
        species,
        seed=args.seed,
        backbone_backend="native",
        native_proposal_spline=True,
        native_qefa=True,
    ).cuda()
    ref.load_state_dict(base.state_dict())
    native, opt = make_groups(ref), make_groups(base)
    standard = native.step(mixed, "joint")
    grouped_update = update(base, opt, grouped, "acvr")
    for k in ("physical_loss", "stationarity_rms_V"):
        torch.testing.assert_close(
            standard[k], grouped_update[k], atol=3e-10, rtol=5e-9
        )
    delta = max(
        (q.detach() - p.detach()).abs().max().item()
        for q, p in zip(ref.parameters(), base.parameters())
    )
    assert delta < 1e-8, delta
    les = build_les_model(species, seed=args.seed, backbone_state=base_state).cuda()
    adam = physical_optimizer(les, method="les", backbone_lr=1e-4, head_lr=1e-5)
    b = tree_to_device(mixed, "cuda")
    energy, force, q = physical_ef(les, b, method="les", create_graph=True)
    labels = mixed["variational"]["labels"]
    joint = energy.square().mean() + force.square().mean()
    derivatives = torch.autograd.grad(
        joint, tuple(les.output_modules[0].parameters()), allow_unused=False
    )
    assert all(torch.isfinite(t).all() for t in derivatives)
    baseline = [p.detach().clone() for p in les.output_modules[0].parameters()]
    sum_by_structure = [float(z.detach().sum()) for z in q.split(b[p.n_atoms].tolist())]
    rep = update(les, adam, grouped, "les")
    assert (
        max(
            (p.detach() - prev).abs().max().item()
            for p, prev in zip(les.output_modules[0].parameters(), baseline)
        )
        > 0
    )
    # Independent position finite difference for total native DPA plus official LES.
    direction = torch.randn_like(b[p.positions])
    direction /= direction.norm()

    def shift(sign):
        moved = {
            **mixed,
            p.positions: mixed[p.positions].clone() + sign * 1e-5 * direction.cpu(),
        }
        return (
            physical_ef(
                les, tree_to_device(moved, "cuda"), method="les", create_graph=False
            )[0]
            .sum()
            .item()
        )

    _, f, _ = physical_ef(
        les, tree_to_device(mixed, "cuda"), method="les", create_graph=False
    )
    fd = (shift(1) - shift(-1)) / 2e-5
    analytic = -(f * direction).sum().item()
    assert abs(fd - analytic) < 1e-6, (fd, analytic)
    return dict(
        training_source_indices=train_ids,
        atoms=mixed[p.n_atoms].tolist(),
        total_charge_e=mixed[p.total_charge].tolist(),
        original_acvr_loss=standard,
        segmented_acvr_loss=grouped_update,
        max_acvr_parameter_difference_after_step=delta,
        les_update=rep,
        latent_charge_sums_initial=sum_by_structure,
        les_directional_force_abs_eV_A=abs(fd - analytic),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output",
        "nacl-trajectory",
        "nacl-split",
        "nacl-reference",
        "sabs-trajectory",
        "sabs-split",
        "sabs-reference",
    ):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.manual_seed(3407)
    torch.use_deterministic_algorithms(True)
    ans = {}
    for tag in ("nacl", "sabs"):
        args = argparse.Namespace(
            seed=3407,
            batch=4,
            **{
                k: getattr(a, tag + "_" + k)
                for k in ("trajectory", "split", "reference")
            },
        )
        ans[tag] = check(tag, args)
    write_json(
        a.output,
        dict(
            device=torch.cuda.get_device_name(),
            dtype="float64",
            official_les_commit="35b971a650cf30adb84fef7138761e81aa47b3a2",
            results=ans,
        ),
    )
    print(json.dumps(ans), flush=True)
