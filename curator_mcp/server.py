"""Minimal MCP server exposing selected CURATOR CLI commands.

This module is intentionally a thin wrapper around existing console commands.
Long-running commands must run in subprocesses so their stdout/stderr cannot
pollute the MCP stdio protocol.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence


def _console_command(command_name: str, module_name: str, function_name: str) -> List[str]:
    executable = shutil.which(command_name)
    if executable:
        return [executable]

    snippet = (
        "import sys; "
        f"from {module_name} import {function_name}; "
        f"raise SystemExit({function_name}(sys.argv[1:]) or 0)"
    )
    return [sys.executable, "-c", snippet]


def _run_command(
    command: Sequence[str],
    *,
    timeout_sec: int,
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "status": "timeout",
            "command": list(command),
            "cwd": cwd or os.getcwd(),
            "timeout_sec": timeout_sec,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }

    return {
        "ok": completed.returncode == 0,
        "status": "completed" if completed.returncode == 0 else "failed",
        "command": list(command),
        "cwd": cwd or os.getcwd(),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def curator_evaluate_help(timeout_sec: int = 10) -> Dict[str, Any]:
    """Return ``curator-evaluate --help`` output through the MCP wrapper."""
    command = _console_command(
        "curator-evaluate",
        "curator.commands.evaluate",
        "evaluate_main",
    )
    return _run_command([*command, "--help"], timeout_sec=timeout_sec)


def curator_evaluate(
    datapath: List[str],
    model_path: List[str],
    device: str = "cpu",
    out: str = "evaluate",
    no_plot: bool = True,
    save_data: bool = False,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
    timeout_sec: int = 3600,
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    """Run ``curator-evaluate`` and return captured process output.

    Args:
        datapath: One or more dataset paths passed to ``curator-evaluate --data``.
        model_path: One or more model checkpoint/run paths passed to
            ``curator-evaluate --model``.
        device: Device string used by CURATOR, e.g. ``cpu`` or ``cuda``.
        out: Output directory for evaluator artifacts.
        no_plot: Disable parity/diagnostic plotting when true.
        save_data: Save raw prediction arrays when true.
        batch_size: Evaluation batch size.
        num_workers: DataLoader worker count.
        pin_memory: Enable DataLoader pinned memory when true.
        timeout_sec: Subprocess timeout in seconds.
        cwd: Optional working directory for the command.
    """
    if not datapath:
        raise ValueError("datapath must contain at least one path.")
    if not model_path:
        raise ValueError("model_path must contain at least one path.")

    command = _console_command(
        "curator-evaluate",
        "curator.commands.evaluate",
        "evaluate_main",
    )
    command.extend(["--data", *datapath, "--model", *model_path])
    command.extend(["--device", device, "--out", out])
    command.extend(["--batch-size", str(batch_size), "--num-workers", str(num_workers)])
    if no_plot:
        command.append("--no-plot")
    if save_data:
        command.append("--save-data")
    if pin_memory:
        command.append("--pin-memory")

    return _run_command(command, timeout_sec=timeout_sec, cwd=cwd)


def fetch_model(
    model_id: str,
    backend: Optional[str] = None,
    resource: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    elements: Optional[List[str]] = None,
    require: Optional[Dict[str, Any]] = None,
    out: str = "model_fetch",
    cache_dir: Optional[str] = None,
    registry_path: Optional[str] = None,
    download: bool = True,
    expected_sha256: Optional[str] = None,
    hash_resource: bool = False,
    probe: bool = False,
    device: str = "cpu",
    timeout_sec: int = 300,
) -> Dict[str, Any]:
    """Resolve a pretrained MLIP into a Curator adapter spec and manifest.

    The returned ``adapter_spec`` can be passed to Curator model-loading paths,
    while ``artifacts.manifest`` records provenance, requirements, capabilities,
    and any probe result.
    """
    from curator.model.foundation import fetch_model as fetch_model_runner

    return fetch_model_runner(
        model_id,
        backend=backend,
        resource=resource,
        params=params,
        elements=elements,
        require=require,
        out=out,
        cache_dir=cache_dir,
        registry_path=registry_path,
        download=download,
        expected_sha256=expected_sha256,
        hash_resource=hash_resource,
        probe=probe,
        device=device,
        timeout_sec=timeout_sec,
    )


def list_foundation_models(
    elements: Optional[List[str]] = None,
    require: Optional[Dict[str, Any]] = None,
    backend: Optional[str] = None,
    registry_path: Optional[str] = None,
    potential_only: bool = True,
    include_builtin: bool = True,
    include_dynamic: bool = False,
    discovery_timeout_sec: int = 10,
    discovery_limit: int = 64,
    out: Optional[str] = None,
) -> Dict[str, Any]:
    """List candidate pretrained MLIPs known to Curator's foundation registry."""
    from curator.model.foundation import list_foundation_models as list_models_runner

    return list_models_runner(
        elements=elements,
        require=require,
        backend=backend,
        registry_path=registry_path,
        potential_only=potential_only,
        include_builtin=include_builtin,
        include_dynamic=include_dynamic,
        discovery_timeout_sec=discovery_timeout_sec,
        discovery_limit=discovery_limit,
        out=out,
    )


def inject_uncertainty_model(
    model_path: str,
    method: str = "auto",
    reference_dataset: Optional[str] = None,
    kernel: str = "local-full-g",
    implementation: str = "native",
    max_structures: Optional[int] = None,
    regularization: float = 1e-6,
    streaming: bool = False,
    output_keys: Optional[List[str]] = None,
    out: str = "uncertainty_model",
    load_weights_only: bool = False,
    cfg_path: Optional[str] = None,
    timeout_sec: int = 1800,
) -> Dict[str, Any]:
    """Inject uncertainty outputs into a single PyTorch-native NeuralNetworkPotential."""
    from curator_mcp.runners.uncertainty import inject_uncertainty_model as runner

    return runner(
        model_path=model_path,
        method=method,
        reference_dataset=reference_dataset,
        kernel=kernel,
        implementation=implementation,
        max_structures=max_structures,
        regularization=regularization,
        streaming=streaming,
        output_keys=output_keys,
        out=out,
        load_weights_only=load_weights_only,
        cfg_path=cfg_path,
        timeout_sec=timeout_sec,
    )


def deploy_model(
    model_path: str,
    format: str = "torchscript",
    element_types: Optional[List[str]] = None,
    out: str = "deployed_model",
    load_weights_only: bool = False,
    cfg_path: Optional[str] = None,
    timeout_sec: int = 1800,
) -> Dict[str, Any]:
    """Deploy an existing model into torchscript, pair-curator, mliap, or mliap-kk format."""
    from curator_mcp.runners.deploy import deploy_model as runner

    return runner(
        model_path=model_path,
        format=format,
        element_types=element_types,
        out=out,
        load_weights_only=load_weights_only,
        cfg_path=cfg_path,
        timeout_sec=timeout_sec,
    )


def list_simulation_engines() -> Dict[str, Any]:
    """List task-aware simulation backends and their current MCP support level."""
    from curator_mcp.runners.simulation import list_simulation_engines as runner

    return runner()


def run_simulation(
    task: Optional[Dict[str, Any]] = None,
    system: Optional[Dict[str, Any]] = None,
    model: Optional[Dict[str, Any]] = None,
    protocol: Optional[Dict[str, Any]] = None,
    backend_policy: Optional[Dict[str, Any]] = None,
    decision_policy: Optional[Dict[str, Any]] = None,
    out: str = "simulation",
    timeout_sec: int = 1800,
) -> Dict[str, Any]:
    """Run task-aware local simulation cases and return validator-ready evidence.

    Use protocol presets such as ``short_md_smoke_v1`` for smoke checks and
    ``md_direct_use_probe_v1`` for direct-use MD validation. ASE and TorchSim
    are routed through backend_policy; LAMMPS remains explicitly listed as not
    implemented until scheduler-aware support exists. NPT/stress-dependent
    runs require explicit model stress capability provenance.
    """
    from curator_mcp.runners.simulation import run_simulation as runner

    return runner(
        task=task,
        system=system,
        model=model,
        protocol=protocol,
        backend_policy=backend_policy,
        decision_policy=decision_policy,
        out=out,
        timeout_sec=timeout_sec,
    )


def validate_simulation_result(
    result: Optional[Dict[str, Any]] = None,
    summary_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
    decision_policy: Optional[Dict[str, Any]] = None,
    task_type: str = "md",
    criteria_profile: str = "md_direct_use_validation_v1",
) -> Dict[str, Any]:
    """Validate structured simulation evidence against direct-use criteria."""
    from curator_mcp.runners.simulation import validate_simulation_result as runner

    return runner(
        result=result,
        summary_path=summary_path,
        manifest_path=manifest_path,
        decision_policy=decision_policy,
        task_type=task_type,
        criteria_profile=criteria_profile,
    )


def list_backend_runtimes(
    runtime_root: Optional[str] = None,
    uv_executable: str = "uv",
) -> Dict[str, Any]:
    """List host or local backend runtimes configured for foundation MLIPs."""
    from curator_mcp.runners.backend_runtime import list_backend_runtimes as runner

    return runner(runtime_root=runtime_root, uv_executable=uv_executable)


def run_backend_runtime(
    task: str,
    model_id: Optional[str] = None,
    adapter_spec: Optional[str] = None,
    backend: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    atoms_path: Optional[str] = None,
    device: str = "cpu",
    download: bool = False,
    runtime: Optional[str] = None,
    runtime_root: Optional[str] = None,
    model_cache: Optional[str] = None,
    uv_cache_dir: Optional[str] = None,
    out: str = "backend_runtime",
    uv_executable: str = "uv",
    timeout_sec: int = 300,
) -> Dict[str, Any]:
    """Run a backend task such as health, resolve, fetch, probe, or predict."""
    from curator_mcp.runners.backend_runtime import run_backend_runtime as runner

    return runner(
        task=task,
        model_id=model_id,
        adapter_spec=adapter_spec,
        backend=backend,
        params=params,
        atoms_path=atoms_path,
        device=device,
        download=download,
        runtime=runtime,
        runtime_root=runtime_root,
        model_cache=model_cache,
        uv_cache_dir=uv_cache_dir,
        out=out,
        uv_executable=uv_executable,
        timeout_sec=timeout_sec,
    )


def predict_model(
    model_id: str,
    atoms_path: str,
    backend: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    download: bool = False,
    runtime: Optional[str] = None,
    runtime_root: Optional[str] = None,
    model_cache: Optional[str] = None,
    uv_cache_dir: Optional[str] = None,
    out: str = "model_predict",
    uv_executable: str = "uv",
    timeout_sec: int = 300,
) -> Dict[str, Any]:
    """Predict energy, forces, and stress for ASE-readable structures using a backend runtime.

    Host Python is used when it already provides the backend package; otherwise
    a local runtime venv can be prepared with ``sync_backend_runtime``.
    ``model_id`` can be a registry alias or a full external adapter spec.
    Extra options such as ``atoms_index`` and ``max_structures`` can be passed
    through ``params``.
    """
    from curator_mcp.runners.backend_runtime import run_backend_runtime as runner

    return runner(
        task="predict",
        model_id=model_id,
        backend=backend,
        params=params,
        atoms_path=atoms_path,
        device=device,
        download=download,
        runtime=runtime,
        runtime_root=runtime_root,
        model_cache=model_cache,
        uv_cache_dir=uv_cache_dir,
        out=out,
        uv_executable=uv_executable,
        timeout_sec=timeout_sec,
    )


def sync_backend_runtime(
    runtime: str,
    runtime_root: Optional[str] = None,
    uv_executable: str = "uv",
    uv_cache_dir: Optional[str] = None,
    mode: str = "shared-torch",
    python_executable: Optional[str] = None,
    base_python: Optional[str] = None,
    timeout_sec: int = 3600,
    out: str = "backend_runtime_sync",
) -> Dict[str, Any]:
    """Explicitly run uv sync for a local backend runtime.

    If the backend package is already available in host Python, no install is
    needed. Otherwise the default shared mode creates a runtime venv that
    inherits system site packages and constrains Torch-family base packages to
    the versions already installed in the base Python. Use
    ``mode="isolated"`` only for a fully independent environment.
    """
    from curator_mcp.runners.backend_runtime import sync_backend_runtime as runner

    return runner(
        runtime=runtime,
        runtime_root=runtime_root,
        uv_executable=uv_executable,
        uv_cache_dir=uv_cache_dir,
        mode=mode,
        python_executable=python_executable,
        base_python=base_python,
        timeout_sec=timeout_sec,
        out=out,
    )


def start_train_model(
    base_model: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    finetune: Optional[Dict[str, Any]] = None,
    distill: Optional[Dict[str, Any]] = None,
    training: Optional[Dict[str, Any]] = None,
    postprocess: Optional[Dict[str, Any]] = None,
    outputs: Optional[str] = None,
    config_patch: Optional[Dict[str, Any]] = None,
    out: str = "train_model",
    device: Optional[str] = None,
    seed: Optional[int] = None,
    run_async: bool = True,
    timeout_sec: int = 3600,
) -> Dict[str, Any]:
    """Start a Curator training run from a structured MCP request.

    The runner reuses Curator's default train config through ``read_user_config``
    and applies structured patches for data, pretrained model, finetuning,
    distillation, trainer limits, and postprocessing. Async mode is the default
    so long training jobs do not block the MCP stdio server.
    """
    from curator_mcp.runners.train import start_train_model as runner

    return runner(
        base_model=base_model,
        data=data,
        finetune=finetune,
        distill=distill,
        training=training,
        postprocess=postprocess,
        outputs=outputs,
        config_patch=config_patch,
        out=out,
        device=device,
        seed=seed,
        run_async=run_async,
        timeout_sec=timeout_sec,
    )


def get_train_job(out: str) -> Dict[str, Any]:
    """Return status and artifact paths for a training job directory."""
    from curator_mcp.runners.train import get_train_job as runner

    return runner(out=out)


def collect_train_result(out: str) -> Dict[str, Any]:
    """Collect the manifest, metrics, checkpoints, and distillation artifacts from a train run."""
    from curator_mcp.runners.train import collect_train_result as runner

    return runner(out=out)


def cancel_train_job(out: str) -> Dict[str, Any]:
    """Request cancellation of a local async training job."""
    from curator_mcp.runners.train import cancel_train_job as runner

    return runner(out=out)


def plan_training_strategy(
    objective: str = "",
    user_constraints: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    candidate_models: Optional[List[Dict[str, Any]]] = None,
    out: str = "train_strategy_plan",
) -> Dict[str, Any]:
    """Plan a structured training workflow before launching training."""
    from curator_mcp.runners.train import plan_training_strategy as runner

    return runner(
        objective=objective,
        user_constraints=user_constraints,
        data=data,
        candidate_models=candidate_models,
        out=out,
    )


def validate_train_workflow_spec(workflow_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a train workflow spec, dependencies, and stage references."""
    from curator_mcp.runners.train import validate_train_workflow_spec as runner

    return runner(workflow_spec=workflow_spec)


def start_train_workflow(
    workflow_spec: Dict[str, Any],
    out: str = "train_workflow",
    run_async: bool = True,
    timeout_sec: int = 7200,
) -> Dict[str, Any]:
    """Start a local ordered train workflow made of structured train stages."""
    from curator_mcp.runners.train import start_train_workflow as runner

    return runner(
        workflow_spec=workflow_spec,
        out=out,
        run_async=run_async,
        timeout_sec=timeout_sec,
    )


def get_train_workflow(out: str) -> Dict[str, Any]:
    """Return status and artifact paths for a train workflow directory."""
    from curator_mcp.runners.train import get_train_workflow as runner

    return runner(out=out)


def collect_train_workflow_result(out: str) -> Dict[str, Any]:
    """Collect a train workflow manifest and final model artifact."""
    from curator_mcp.runners.train import collect_train_workflow_result as runner

    return runner(out=out)


def build_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise RuntimeError(
            "The MCP Python package is not installed. Install CURATOR with "
            "`pip install -e '.[mcp]'` or install `mcp` in this environment."
        ) from exc

    mcp = FastMCP("curator")

    mcp.tool()(curator_evaluate_help)
    mcp.tool()(curator_evaluate)
    mcp.tool()(fetch_model)
    mcp.tool()(list_foundation_models)
    mcp.tool()(inject_uncertainty_model)
    mcp.tool()(deploy_model)
    mcp.tool()(list_simulation_engines)
    mcp.tool()(run_simulation)
    mcp.tool()(validate_simulation_result)
    mcp.tool()(list_backend_runtimes)
    mcp.tool()(sync_backend_runtime)
    mcp.tool()(run_backend_runtime)
    mcp.tool()(predict_model)
    mcp.tool()(start_train_model)
    mcp.tool()(get_train_job)
    mcp.tool()(collect_train_result)
    mcp.tool()(cancel_train_job)
    mcp.tool()(plan_training_strategy)
    mcp.tool()(validate_train_workflow_spec)
    mcp.tool()(start_train_workflow)
    mcp.tool()(get_train_workflow)
    mcp.tool()(collect_train_workflow_result)

    return mcp


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
