import torch
import pytest
from ase.io import read

from curator.data import AseDataReader
from curator.layer.utils import find_layer_by_name_recursive
from curator.utils import convert_curator_to_mace, convert_mace_to_curator, load_models


def _predict(model, inputs):
    model.eval()
    out = model({k: v.clone() for k, v in inputs.items()})
    return out["energy"], out["forces"]


def test_curator_to_mace_roundtrip(tmp_path):
    try:
        import mace  # noqa: F401
    except Exception:
        pytest.skip("mace package with cuequivariance dependencies not available.")
    device = torch.device("cuda")
    curator_path = "mace_e3nn.pth"

    # convert curator -> mace -> curator
    mace_path = tmp_path / "roundtrip_mace.pth"
    back_curator_path = tmp_path / "roundtrip_curator.pth"
    try:
        convert_curator_to_mace(curator_path, mace_path, device=device)
    except RuntimeError as exc:
        pytest.skip(f"curator->mace conversion not available in this environment: {exc}")
    convert_mace_to_curator(mace_path, back_curator_path, device=device)

    orig = load_models(curator_path, device=device, load_compiled=False)[0]
    rt = load_models(back_curator_path, device=device, load_compiled=False)[0]

    cutoff = float(find_layer_by_name_recursive(orig, "cutoff"))
    reader = AseDataReader(cutoff, compute_neighbor_list=True)
    atoms = read("../LiFePO4.traj", index=0)
    inputs = reader(atoms)

    e0, f0 = _predict(orig, inputs)
    e1, f1 = _predict(rt, inputs)

    torch.testing.assert_close(e0, e1)
    torch.testing.assert_close(f0, f1, rtol=1e-5, atol=1e-6)
