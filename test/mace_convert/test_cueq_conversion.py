import torch
import pytest
from ase.io import read

from curator.data import AseDataReader
from curator.layer._cuequivariance_wrapper import IS_CUET_AVAILABLE
from curator.layer.utils import find_layer_by_name_recursive
from curator.utils import convert_cueq_to_e3nn, convert_e3nn_to_cueq, load_models


def _get_inputs(model, traj_path):
    cutoff = float(find_layer_by_name_recursive(model, "cutoff"))
    reader = AseDataReader(cutoff, compute_neighbor_list=True)
    atoms = read(traj_path, index=0)
    inputs = reader(atoms)
    return {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}


def _predict(model, inputs):
    model.eval()
    out = model({k: v.clone() for k, v in inputs.items()})
    return out["energy"], out["forces"]


@pytest.fixture(scope="module")
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for cuequivariance tests.")
    if not IS_CUET_AVAILABLE:
        pytest.skip("cuequivariance not available; cannot run cueq backend tests.")
    return torch.device("cuda")


def test_e3nn_to_cueq_matches_outputs(cuda_device):
    device = cuda_device
    model = load_models("test/mace_convert/mace_e3nn.pth", device=device, load_compiled=False)[0].to(device)
    inputs = _get_inputs(model, "test/LiFePO4.traj")

    e_energy, e_forces = _predict(model, inputs)

    cueq_model = convert_e3nn_to_cueq(model)
    c_energy, c_forces = _predict(cueq_model, inputs)

    torch.testing.assert_close(c_energy, e_energy)
    torch.testing.assert_close(c_forces, e_forces, rtol=1e-5, atol=1e-6)


def test_cueq_to_e3nn_matches_outputs(cuda_device):
    device = cuda_device
    model = load_models("test/mace_convert/mace_cueq.pth", device=device, load_compiled=False)[0].to(device)
    inputs = _get_inputs(model, "test/LiFePO4.traj")

    cueq_energy, cueq_forces = _predict(model, inputs)

    e3nn_model = convert_cueq_to_e3nn(model)
    e_energy, e_forces = _predict(e3nn_model, inputs)

    torch.testing.assert_close(e_energy, cueq_energy)
    torch.testing.assert_close(e_forces, cueq_forces, rtol=1e-5, atol=1e-6)
