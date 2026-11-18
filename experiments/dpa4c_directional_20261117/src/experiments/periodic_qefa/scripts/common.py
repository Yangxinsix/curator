"""Frozen inputs and native loading for the shared-proposal GPU campaign."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('ACVR_SOURCE', '/data1/home/hzxy05/work/yangxin/curator-acvr'))
sys.path.insert(0, str(SOURCE))


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def initialize():
    import torch
    import curator

    if not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required; CPU fallback is disabled")
    if not Path(curator.__file__).resolve().is_relative_to(SOURCE):
        raise RuntimeError("Curator must come from the frozen canonical-source snapshot")
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, expected in manifest["source_files"].items():
        if sha(SOURCE / name) != expected:
            raise RuntimeError("Frozen source changed: " + name)
    for name, expected in manifest["input_files"].items():
        if sha(name) != expected:
            raise RuntimeError("Frozen input changed: " + name)
    for name, expected in manifest.get("experiment_files", {}).items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("Experiment source changed: " + name)
    if sha(manifest["initial_checkpoint"]) != manifest["initial_checkpoint_sha256"]:
        raise RuntimeError("Initial checkpoint changed")
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("GPU_CAMPAIGN_VERIFIED", {
        "gpu": torch.cuda.get_device_name(),
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "job_id": os.environ.get("ACVR_JOB_ID"),
        "source": str(SOURCE), "checkpoint_sha256": manifest["initial_checkpoint_sha256"],
    }, flush=True)


def load_model(path):
    """Restore selected state through Curator, with JIT tensors loaded on CPU."""
    from unittest.mock import patch
    import torch
    from omegaconf import OmegaConf
    from curator.commands.train import _load_model

    native = torch.jit.load

    def cpu_load(*args, **kwargs):
        if len(args) < 2 and kwargs.get("map_location") is None:
            kwargs["map_location"] = "cpu"
        return native(*args, **kwargs)

    with patch("torch.jit.load", side_effect=cpu_load):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = OmegaConf.create({"model": {}, "model_path": {
            "path": str(path), "mode": "model"}})
        model = _load_model(config).model
    return model, payload
