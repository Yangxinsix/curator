#!/usr/bin/env python3
"""
Benchmark eager vs TorchScripted (compiled) Curator model on the LiFePO4 example.

This uses the reference checkpoint and trajectory under curator/test, scripts the
model with e3nn.util.jit.script, and reports average forward-pass latency.
"""

import argparse
import time
from pathlib import Path
import contextlib

import torch
import os
import warnings
from ase.io import Trajectory
from e3nn.util.jit import script

from curator.data import AseDataReader
from curator.layer.utils import find_layer_by_name_recursive
from curator.utils import load_model

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.suppress_errors = True
warnings.filterwarnings("ignore", message=".*weights_only.*")
warnings.filterwarnings("ignore", message="cuequivariance is not available.*")
DEFAULT_CUTOFF = 5.0
CUDA_BROKEN = False


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as exc:
            print(f"CUDA reported but not usable ({exc}); falling back to CPU.")
            global CUDA_BROKEN
            CUDA_BROKEN = True
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    return torch.device("cpu")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    if device.type == "cuda":
        if precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "fp16":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
    if device.type == "cpu" and precision == "bf16":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    print(f"Requested precision {precision} not supported on {device}; using fp32.")
    return contextlib.nullcontext()


def _infer_cutoff(model: torch.nn.Module) -> float:
    layer = find_layer_by_name_recursive(model, "cutoff")
    if layer is None:
        return DEFAULT_CUTOFF
    try:
        return float(layer)
    except Exception:
        return DEFAULT_CUTOFF


def _clone_inputs(inputs: dict) -> dict:
    cloned = {}
    for k, v in inputs.items():
        cloned[k] = v.clone() if isinstance(v, torch.Tensor) else v
    return cloned


def _set_energy_only(model: torch.nn.Module) -> torch.nn.Module:
    for module in model.modules():
        if hasattr(module, "model_outputs") and isinstance(module.model_outputs, list):
            filtered = [m for m in module.model_outputs if m == "energy"]
            module.model_outputs = filtered if filtered else ["energy"]
    if hasattr(model, "collect_outputs"):
        model.collect_outputs()
    if hasattr(model, "model_outputs"):
        model.model_outputs = ["energy"]
    return model


def _prepare_inputs(traj_path: Path, cutoff: float, device: torch.device, frame: int, supercell: tuple[int, int, int]) -> dict:
    atoms = Trajectory(traj_path)[frame].copy()
    if supercell != (1, 1, 1):
        atoms = atoms.repeat(supercell)
    reader = AseDataReader(cutoff=cutoff, compute_neighbor_list=True)
    base = reader(atoms)
    return {k: v.to(device) for k, v in base.items()}


def _has_forces(model: torch.nn.Module, base_inputs: dict) -> bool:
    outputs = model(_clone_inputs(base_inputs))
    return isinstance(outputs, dict) and "forces" in outputs


def _benchmark(model: torch.nn.Module, base_inputs: dict, device: torch.device, warmup: int, repeat: int, precision: str) -> float:
    model.eval()
    # warmup to remove one-time overheads
    with _autocast_context(device, precision):
        for _ in range(warmup):
            model(_clone_inputs(base_inputs))
        _synchronize(device)

    start = time.perf_counter()
    with _autocast_context(device, precision):
        for _ in range(repeat):
            model(_clone_inputs(base_inputs))
        _synchronize(device)

    return (time.perf_counter() - start) / repeat


def _run_with_retry(fn, device: torch.device, precision: str):
    try:
        return fn(precision)
    except RuntimeError as exc:
        if device.type == "cuda" and "out of memory" in str(exc):
            torch.cuda.empty_cache()
            if precision != "bf16":
                print("OOM detected; retrying with bf16.")
                return fn("bf16")
            if precision != "fp16":
                print("OOM detected; retrying with fp16.")
                return fn("fp16")
        raise


def main():
    parser = argparse.ArgumentParser(description="Benchmark scripted Curator model latency.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations before timing.")
    parser.add_argument("--repeat", type=int, default=20, help="Number of timed forward passes.")
    parser.add_argument("--frame", type=int, default=0, help="Trajectory frame index to benchmark.")
    parser.add_argument(
        "--energy-only",
        action="store_true",
        help="Benchmark energy-only outputs (skip forces/stress) for both eager and scripted.",
    )
    parser.add_argument(
        "--supercell",
        type=int,
        nargs=3,
        default=[1, 1, 1],
        metavar=("NX", "NY", "NZ"),
        help="Repeat the base structure to build a larger test system (e.g., 2 2 2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Force device selection (default auto: CUDA if available, else CPU).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
        help="Precision for inference (uses autocast when supported).",
    )
    parser.add_argument(
        "--compile-eager",
        action="store_true",
        help="Apply torch.compile to the eager model before benchmarking.",
    )
    parser.add_argument(
        "--compile-script",
        action="store_true",
        help="Apply torch.compile to the scripted model (experimental).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "best_model.ckpt"
    traj_path = repo_root / "LiFePO4.traj"

    device = _select_device() if args.device == "auto" else torch.device(args.device)
    eager_model = load_model(model_path, device=device, load_compiled=False)
    if args.compile_eager:
        if device.type != "cuda":
            print("Skipping torch.compile on eager model for CPU; enable CUDA to use compile.")
        else:
            try:
                eager_model = torch.compile(eager_model, mode="reduce-overhead", fullgraph=False, dynamic=True, backend="inductor")
            except Exception as exc:
                print(f"torch.compile failed on eager model ({exc}); falling back to uncompiled.")
    if args.energy_only:
        eager_model = _set_energy_only(eager_model)
    cutoff = _infer_cutoff(eager_model)

    inputs = _prepare_inputs(traj_path, cutoff, device, args.frame, tuple(args.supercell))

    eager_has_forces = _has_forces(eager_model, inputs)

    energy_only = False
    scripted_source = load_model(model_path, device=device, load_compiled=False)
    if args.energy_only:
        scripted_source = _set_energy_only(scripted_source)
    try:
        scripted_model = script(scripted_source)
        if args.compile_script:
            if device.type != "cuda":
                print("Skipping torch.compile on scripted model for CPU; enable CUDA to use compile.")
            else:
                try:
                    scripted_model = torch.compile(scripted_model, mode="reduce-overhead", fullgraph=False, dynamic=True, backend="inductor")
                except Exception as exc:
                    print(f"torch.compile failed on scripted model ({exc}); using scripted-only.")
        scripted_latency = _run_with_retry(
            lambda prec: _benchmark(scripted_model, inputs, device, args.warmup, args.repeat, prec),
            device,
            args.precision,
        )
    except RuntimeError as exc:
        print(f"Scripted model failed with full outputs ({exc}); retrying energy-only.")
        energy_only = True
        scripted_source = _set_energy_only(load_model(model_path, device=device, load_compiled=False))
        scripted_model = script(scripted_source)
        scripted_latency = _run_with_retry(
            lambda prec: _benchmark(scripted_model, inputs, device, args.warmup, args.repeat, prec),
            device,
            args.precision,
        )
        eager_model = _set_energy_only(load_model(model_path, device=device, load_compiled=False))

    scripted_has_forces = _has_forces(scripted_model, inputs)
    eager_latency = _run_with_retry(
        lambda prec: _benchmark(eager_model, inputs, device, args.warmup, args.repeat, prec),
        device,
        args.precision,
    )

    print(f"Device: {device.type}")
    print(f"Precision: {args.precision}")
    print(f"torch.compile (eager): {args.compile_eager}; torch.compile (script): {args.compile_script}")
    print(f"Cutoff used: {cutoff:.2f} Angstrom")
    print(f"Supercell: {tuple(args.supercell)}; total atoms: {inputs['n_atoms'].item()}")
    print(f"Eager includes forces:    {eager_has_forces}")
    print(f"Scripted includes forces: {scripted_has_forces}")
    print(f"Eager latency:    {eager_latency * 1000:.3f} ms")
    print(f"Scripted latency: {scripted_latency * 1000:.3f} ms")
    if energy_only:
        print("Note: Benchmarked energy-only path because TorchScript failed on full outputs.")
    if scripted_latency > 0:
        print(f"Speedup (eager/scripted): {eager_latency / scripted_latency:.2f}x")


if __name__ == "__main__":
    main()
