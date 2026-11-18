"""Compare the optimized molecular kernels and the unchanged periodic path at fixed weights."""

import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from train_pilot import HERE, build_model, data_panel
import torch
from ase import Atoms
from cache import output_head
from common import write_json
from curator.data import properties as p
from curator.data.charge_conditions import ChargeBoundary
from curator.data.variational_data import canonical_boundary_hash
from curator.train.train import tree_to_device
from graph_joint import eligible


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("trajectory", "reference", "split", "output"):
        ap.add_argument("--" + key, type=Path, required=True)
    args = ap.parse_args()
    args.seed, args.batch, args.panel = 3407, 1, HERE / "pure_local/source_indices.json"
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with TemporaryDirectory() as tmp:
        ds = data_panel(args, Path(tmp))
    base = ds.val_dataloader().collate_fn([ds.datasets["validation"][0]])
    ck = torch.load(HERE / "sabs_warm/final.pt", weights_only=False, map_location="cpu")
    opts = ck["model_options"] | {
        "periodic_backend": "rsa",
        "backbone_backend": "native",
        "native_proposal_spline": True,
    }
    models = []
    for fused in (False, True):
        model = build_model(**opts, native_qefa=fused).cuda().eval()
        model.load_state_dict(ck["model"], strict=True)
        models.append(model)
    frame = Atoms(
        numbers=base[p.atomic_numbers].tolist(), positions=base[p.positions].tolist()
    )
    n = len(frame)
    opts = dict(
        type="fixed_potential",
        groups={"first": [0], "rest": list(range(1, n))},
        potential={"first": 0.2, "rest": -0.15},
    )
    rows = {}
    for case in ("Q", "QV", "V", "periodic_RSA_Q"):
        b = deepcopy(base)
        if case in ("QV", "V"):
            spec = asdict(
                ChargeBoundary(
                    **opts, total_charge=-1.0 if case == "QV" else None
                ).build(frame)[0]
            )
            b["variational"]["boundary_spec"] = (spec,)
            b["variational"]["boundary_hash"] = (canonical_boundary_hash(spec),)
            if case == "V":
                b.pop(p.total_charge)
        if case == "periodic_RSA_Q":
            b[p.cell] = torch.eye(3).reshape_as(b[p.cell]) * 24.0
            b[p.pbc] = torch.ones_like(b[p.pbc])
        eligible_now = eligible(b)
        assert eligible_now == (case == "Q")
        pred = []
        for model in models:
            calls = []
            hook = output_head(model).proposal.register_forward_hook(
                lambda *args: calls.append(1)
            )
            try:
                v = model(tree_to_device(deepcopy(b), "cuda"))["variational_prediction"]
            finally:
                hook.remove()
            assert len(calls) == 1
            pred.append(
                {
                    k: v["labels"][k]["values"].detach()
                    for k in ("condition_energy", "condition_forces", "variable_value")
                }
                | {"stationarity": v["residuals"]["stationarity"]["values"].detach()}
            )
        diffs = {k: float((t - pred[0][k]).abs().max()) for k, t in pred[1].items()}
        assert max(diffs.values()) < 1e-9
        rows[case] = dict(
            graph_eligible=eligible_now, max_abs_differences=diffs, proposal_calls=1
        )
    write_json(
        args.output,
        dict(
            protocol="One SA-BS validation geometry on GPU, identical FP64 weights. Two groups at +0.20/-0.15 V for QV/V. Periodic RSA uses a synthetic 24 Angstrom cubic cell. Compare ordinary per-condition inference with the new molecular kernels enabled/disabled. No test of accuracy against labels.",
            cases=rows,
        ),
    )


if __name__ == "__main__":
    main()
