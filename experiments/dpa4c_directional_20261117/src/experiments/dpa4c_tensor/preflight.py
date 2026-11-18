"""Parity and force-gradient checks on the exact trained molecular and periodic parents."""
import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dpa4c_gemini"))
import torch
from extension_run import inputs, build as build_extension
from train import configure, make_model, paths
from train_pilot import data_panel
from equilibrium import _read_only, _forward, model_inputs
from fixed_q_batch import panel_energy_charge
from curator.data import properties as p
from tensor_model import install_variant


def frame(system, root):
    trajectory, split, ref = paths(system) if system == "aumgo" else inputs(system)
    split_ids = json.loads(split.read_text())
    panel = root / "ids.json"
    panel.write_text(json.dumps({x:split_ids[x][:2] for x in ("train", "validation")}))
    data = data_panel(argparse.Namespace(trajectory=trajectory, split=split, reference=ref,
                                         panel=panel, seed=3407, batch=1), root)
    batch = data.val_dataloader().collate_fn([data.datasets["validation"][0]])
    return data, batch


def native(model, batch, mode):
    with _read_only(model, "cuda") as head:
        out = _forward(model, head, batch, "cuda", mode)
    return tuple(torch.as_tensor(out[k], dtype=torch.float64) for k in ("energy", "forces", "q"))


def fast(model, batch):
    data = model_inputs(batch, "cuda")
    for layer in model.input_modules:
        data = layer(data)
    data = model.representation(data)
    energy, q, _ = panel_energy_charge(model.output_modules[0], data, return_partial=True)
    f, = torch.autograd.grad(-energy.sum(), data[p.positions], create_graph=True)
    return energy.reshape(()), f, q.flatten()


def compare(a, b, *, tol=2e-7):
    out = {}
    for k, x, y in zip(("energy", "force", "charge"), a, b, strict=True):
        x, y = x.detach().cpu(), y.detach().cpu()
        out[k] = (x-y).abs().max().item()
        torch.testing.assert_close(x, y, atol=tol, rtol=1e-9)
    return out


def inspect(system, folder, weights, arms):
    ds, batch = frame(system, folder)
    model_factory = (lambda: make_model("aumgo","A3",3407)) if system=="aumgo" else (
        lambda: build_extension("cumulene", ds.sources[0].species, "acvr", 3407,"B2"))
    initial = torch.load(weights, map_location="cpu", weights_only=False)
    result = {}
    old_pred = old_zero = None
    for arm in ["old", *arms]:
        model = model_factory().double().cuda()
        model.load_state_dict(initial["model"], strict=True)
        base_names = set(dict(model.named_parameters()))
        install_variant(model, arm, seed=3407, exclude_self=True)
        new = [(k,v) for k,v in model.named_parameters() if k not in base_names]
        predicted, clamped = (native(model, batch, mode) for mode in ("proposal","uniform"))
        if old_pred is None:
            old_pred,old_zero = predicted,clamped
        rec = result[arm] = dict(initial=compare(predicted,old_pred), initial_clamped=compare(clamped,old_zero))
        f = fast(model,batch)
        rec["initial_fast"] = compare(predicted,f)
        if not new:
            assert arm=="old"
            continue
        assert sum(p.numel() for _,p in new)>0
        # A force-only target exercises the mixed position/parameter derivative.
        target = batch["variational"]["labels"]["condition_forces"]["values"].to(f[1]).reshape_as(f[1])
        loss = (f[1]-target).square().mean()
        grads = torch.autograd.grad(loss, [v for _,v in new], allow_unused=True)
        rec["gradient_norms"] = {k:None if g is None else float(g.norm()) for (k,_),g in zip(new,grads)}
        active = [(k,v,g) for (k,v),g in zip(new,grads) if g is not None and bool(g.abs().max()>0)]
        assert active and all(g is None or g.isfinite().all() for g in grads)
        # Activate the initially identity-preserving branch to catch dead train/eval paths.
        with torch.no_grad():
            for k,v,g in active:
                v.add_(0.2 * g/g.norm())
        activated = native(model,batch,"proposal")
        clamped1 = native(model,batch,"uniform")
        rec["active_fast"] = compare(activated,fast(model,batch))
        rec["active_dE"] = float(activated[0]-predicted[0])
        rec["active_dF_max"] = float((activated[1]-predicted[1]).abs().max())
        rec["active_clamped_dE"] = float(clamped1[0]-clamped[0])
        if arm.startswith("orient"):
            torch.testing.assert_close(activated[2],predicted[2],atol=1e-12,rtol=1e-12)
            assert abs(rec["active_clamped_dE"])>1e-12, rec
            assert abs(rec["active_clamped_dE"]-rec["active_dE"])<1e-7
        else:
            assert abs(rec["active_clamped_dE"])<1e-8 and rec["active_dF_max"]>1e-13, rec
        # Rotation not in the Lebedev symmetry group; tolerances include finite quadrature.
        rot = torch.tensor([[.36,-.48,.8],[.8,.60,0.],[-.48,.64,.60]],dtype=batch[p.positions].dtype)
        rb = dict(batch)
        rb[p.positions] = batch[p.positions] @ rot.T
        if p.cell in batch:
            rb[p.cell] = batch[p.cell] @ rot.T
        refrot = (activated[0],activated[1] @ rot.double().T,activated[2])
        rec["active_rotation"] = compare(native(model,rb,"proposal"),refrot,tol=1e-5)
        print(system,arm,rec,flush=True)
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cumulene", type=Path)
    cli.add_argument("--aumgo", type=Path)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    configure()
    report={}
    for system,ck in (("cumulene",args.cumulene),("aumgo",args.aumgo)):
        if ck:
            with TemporaryDirectory() as tmp:
                report[system]=inspect(system,Path(tmp),ck,["qk2","orient2","orient12"])
            args.out.write_text(json.dumps(report,indent=2)+"\n")
