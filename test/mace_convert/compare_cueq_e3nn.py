import torch
from ase.io import read

from curator.data import AseDataReader
from curator.layer.utils import find_layer_by_name_recursive
from curator.utils import load_models


def _prepare_inputs(model, traj="../LiFePO4.traj"):
    cutoff = float(find_layer_by_name_recursive(model, "cutoff"))
    reader = AseDataReader(cutoff, compute_neighbor_list=True)
    atoms = read(traj, index=0)
    inputs = reader(atoms)
    return {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}


def _predict(model, inputs):
    model.eval()
    out = model({k: v.clone() for k, v in inputs.items()})
    return out["energy"], out["forces"]


def compare(model_path_a, model_path_b, device="cuda"):
    m_a = load_models(model_path_a, device=device, load_compiled=False)[0]
    m_b = load_models(model_path_b, device=device, load_compiled=False)[0]

    inputs = _prepare_inputs(m_a)

    e_a, f_a = _predict(m_a, inputs)
    e_b, f_b = _predict(m_b, inputs)

    torch.testing.assert_close(e_a, e_b)
    torch.testing.assert_close(f_a, f_b, rtol=1e-5, atol=1e-6)
    print(f"PASS: {model_path_a} == {model_path_b}")


def main():
    compare("mace_cueq.pth", "mace_cueq_e3nn.pth")
    compare("mace_e3nn.pth", "mace_e3nn_cueq.pth")


if __name__ == "__main__":
    main()
