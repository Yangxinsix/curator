"""Build both differentiable CUDA extensions on the training partition."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dpa4c_acvr"))
import model
import torch
from curator.model.dpa4c_small import _extension
from native_spline import extension

print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability(), flush=True)
_extension()
extension()
print("Native DPA4C-Small and ACVR spline extensions loaded", flush=True)
