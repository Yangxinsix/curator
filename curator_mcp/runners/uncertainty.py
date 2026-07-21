from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from curator.data import properties

from .artifacts import ensure_dir, error_result, path_status, utc_now, write_json, write_text


_ALLOWED_METHODS = {"auto", "mahalanobis"}
_ALLOWED_IMPLEMENTATIONS = {"native", "scriptable"}


def _resolve_method(method: str, reference_dataset: Optional[str]) -> str:
    normalized = str(method or "auto").strip().lower()
    if normalized not in _ALLOWED_METHODS:
        raise ValueError(f"method must be one of {sorted(_ALLOWED_METHODS)}, got {method!r}.")
    if normalized != "auto":
        return normalized
    if reference_dataset:
        return "mahalanobis"
    raise ValueError("method=auto requires reference_dataset for Mahalanobis uncertainty injection.")


def _normalize_implementation(implementation: str) -> str:
    normalized = str(implementation or "native").strip().lower()
    if normalized not in _ALLOWED_IMPLEMENTATIONS:
        raise ValueError(
            f"implementation must be one of {sorted(_ALLOWED_IMPLEMENTATIONS)}, got {implementation!r}."
        )
    return normalized


def _uncertainty_keys(
    method: str,
    *,
    implementation: str,
    kernel: str,
    output_keys: Optional[List[str]] = None,
) -> List[str]:
    if method == "mahalanobis":
        keys = [properties.maha_dist]
        normalized_kernel = str(kernel).replace("-", "_")
        if (
            implementation == "scriptable"
            and normalized_kernel.startswith("local_")
            and (output_keys is None or properties.maha_dist_per_atom in output_keys)
        ):
            keys.append(properties.maha_dist_per_atom)
        return keys
    return []


def _build_spec(
    method: str,
    *,
    reference_dataset: Optional[str],
    kernel: str,
    max_structures: Optional[int],
    regularization: float,
    streaming: bool,
    output_keys: Optional[List[str]],
) -> Dict[str, Any]:
    if reference_dataset in (None, "", "none", "null"):
        raise ValueError("Mahalanobis uncertainty injection requires reference_dataset.")
    return {
        "method": method,
        "dataset": reference_dataset,
        "output_keys": output_keys,
        "maha": {
            "kernel": kernel,
            "max_structures": max_structures,
            "regularization": regularization,
            "streaming": streaming,
        },
    }


def _target_capabilities(method: str, implementation: str, kernel: str) -> Dict[str, Any]:
    scriptable = method == "mahalanobis" and implementation == "scriptable"
    normalized_kernel = str(kernel).replace("-", "_")
    return {
        "model_kind": "NeuralNetworkPotential",
        "native_pytorch": True,
        "uncertainty_forward_outputs": True,
        "torchscript_exportable": bool(scriptable and normalized_kernel in {"gnn", "local_gnn"}),
        "deployable_formats": (
            ["torchscript", "pair-curator", "mliap", "mliap-kk"]
            if scriptable
            else ["mliap", "mliap-kk"]
        ),
        "requires_python_runtime_for_uncertainty": implementation == "native",
    }


def _run_subprocess(
    command: List[str],
    *,
    run_dir: Path,
    timeout_sec: int,
    stdout_name: str,
    stderr_name: str,
) -> Tuple[subprocess.CompletedProcess[str], Dict[str, str]]:
    completed = subprocess.run(
        command,
        cwd=str(Path.cwd()),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    return completed, {
        "stdout": write_text(run_dir / stdout_name, completed.stdout),
        "stderr": write_text(run_dir / stderr_name, completed.stderr),
    }


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
    """Inject uncertainty into a single PyTorch-native NeuralNetworkPotential."""

    run_dir = ensure_dir(out)
    artifacts: Dict[str, Any] = {}
    try:
        resolved_method = _resolve_method(method, reference_dataset)
        implementation = _normalize_implementation(implementation)
        spec = _build_spec(
            resolved_method,
            reference_dataset=reference_dataset,
            kernel=kernel,
            max_structures=max_structures,
            regularization=regularization,
            streaming=streaming,
            output_keys=output_keys,
        )
        target_path = run_dir / "uncertainty_model.pth"
        request = {
            "model_path": str(model_path),
            "target_path": str(target_path),
            "load_weights_only": load_weights_only,
            "cfg_path": cfg_path,
            "uncertainty_spec": spec,
            "implementation": implementation,
        }
        request_path = run_dir / "inject_uncertainty_request.json"
        artifacts["request"] = write_json(request_path, request)
        snippet = (
            "import json, sys, torch; "
            "from pathlib import Path; "
            "from curator.model import NeuralNetworkPotential; "
            "from curator.simulate.uncertainty.inject import inject_uncertainty; "
            "from curator.utils import load_models, normalize_config_sequences, read_user_config; "
            "request=json.loads(Path(sys.argv[1]).read_text()); "
            "cfg=None; "
            "cfg_path=request.get('cfg_path'); "
            "\nif cfg_path is not None:\n"
            "    cfg=read_user_config(cfg_path, config_path='configs', config_name='train')\n"
            "    normalize_config_sequences(cfg)\n"
            "models=load_models(request['model_path'], device=None, load_compiled=False, load_weights_only=request.get('load_weights_only', False), cfg=cfg); "
            "\nif len(models) != 1:\n"
            "    raise ValueError('inject_uncertainty_model expects exactly one model_path')\n"
            "model=models[0]; "
            "\nif not isinstance(model, NeuralNetworkPotential):\n"
            "    raise TypeError(f'inject_uncertainty_model expects NeuralNetworkPotential, got {type(model)}')\n"
            "inject_uncertainty(model, request.get('uncertainty_spec'), implementation=request.get('implementation', 'native')); "
            "torch.save(model, request['target_path'])"
        )
        command = [sys.executable, "-c", snippet, str(request_path)]
        completed, logs = _run_subprocess(
            command,
            run_dir=run_dir,
            timeout_sec=timeout_sec,
            stdout_name="inject_uncertainty_stdout.txt",
            stderr_name="inject_uncertainty_stderr.txt",
        )
        artifacts.update(logs)
        artifacts["model"] = str(target_path)

        uncertainty_keys = _uncertainty_keys(
            resolved_method,
            implementation=implementation,
            kernel=kernel,
            output_keys=output_keys,
        )
        manifest = {
            "created_at": utc_now(),
            "tool": "inject_uncertainty_model",
            "ok": completed.returncode == 0 and target_path.exists(),
            "status": "completed" if completed.returncode == 0 and target_path.exists() else "failed",
            "source_model_path": str(model_path),
            "model_path": str(target_path),
            "model_kind": "NeuralNetworkPotential",
            "method": resolved_method,
            "implementation": implementation,
            "reference_dataset": reference_dataset,
            "uncertainty_spec": spec,
            "uncertainty_keys": uncertainty_keys,
            "capabilities": _target_capabilities(resolved_method, implementation, kernel),
            "returncode": completed.returncode,
            "artifacts": artifacts,
            "target_status": path_status(target_path),
        }
        artifacts["manifest"] = write_json(run_dir / "uncertainty_manifest.json", manifest)

        if completed.returncode != 0:
            return error_result(
                "InjectUncertaintyFailed",
                f"Uncertainty injection subprocess failed with return code {completed.returncode}.",
                log_path=artifacts.get("stderr"),
                artifacts=artifacts,
                method=resolved_method,
                implementation=implementation,
                returncode=completed.returncode,
            )
        if not target_path.exists():
            return error_result(
                "MissingArtifact",
                f"Injection completed but expected model artifact was not written: {target_path}",
                log_path=artifacts["stdout"],
                artifacts=artifacts,
                method=resolved_method,
                implementation=implementation,
                returncode=completed.returncode,
            )

        return {
            "ok": True,
            "status": "completed",
            "method": resolved_method,
            "implementation": implementation,
            "model_path": str(target_path),
            "model_kind": "NeuralNetworkPotential",
            "uncertainty_keys": uncertainty_keys,
            "capabilities": _target_capabilities(resolved_method, implementation, kernel),
            "artifacts": artifacts,
            "returncode": completed.returncode,
        }
    except subprocess.TimeoutExpired as exc:
        artifacts["stdout"] = write_text(run_dir / "inject_uncertainty_stdout.txt", exc.stdout or "")
        artifacts["stderr"] = write_text(run_dir / "inject_uncertainty_stderr.txt", exc.stderr or "")
        return error_result(
            "Timeout",
            f"Uncertainty injection exceeded timeout_sec={timeout_sec}.",
            log_path=artifacts.get("stderr"),
            artifacts=artifacts,
            recoverable=True,
        )
    except Exception as exc:
        return error_result(type(exc).__name__, str(exc), artifacts=artifacts)


def build_uncertainty_adapter(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Deprecated compatibility stub; use inject_uncertainty_model then deploy_model."""

    return error_result(
        "DeprecatedTool",
        "build_uncertainty_adapter mixed uncertainty injection and deployment. "
        "Use inject_uncertainty_model first, then deploy_model if a deployed format is needed.",
        recoverable=True,
    )
