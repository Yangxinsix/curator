from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .artifacts import ensure_dir, error_result, path_status, read_json, utc_now, write_json, write_text


_ALLOWED_FORMATS = {
    "torchscript",
    "pair-curator",
    "pair_curator",
    "lammps-pair-curator",
    "lammps_pair_curator",
    "mliap",
    "mliap-kk",
    "mliap_kk",
    "lammps-mliap",
    "lammps_mliap",
}


def _normalize_format(format: str) -> str:
    normalized = str(format or "torchscript").strip().lower()
    if normalized not in _ALLOWED_FORMATS:
        raise ValueError(f"format must be one of {sorted(_ALLOWED_FORMATS)}, got {format!r}.")
    if normalized in {"pair_curator", "lammps-pair-curator", "lammps_pair_curator"}:
        return "pair-curator"
    if normalized in {"mliap-kk", "mliap_kk"}:
        return "mliap-kk"
    if normalized in {"lammps-mliap", "lammps_mliap"}:
        return "mliap"
    return normalized


def _target_file(run_dir: Path, format: str) -> Path:
    if format == "torchscript":
        return run_dir / "torchscript_model.pt"
    if format == "pair-curator":
        return run_dir / "pair_curator_model.pt"
    return run_dir / "mliap_model.pt"


def _format_capabilities(format: str) -> Dict[str, Any]:
    return {
        "deploy_format": format,
        "torchscript": format in {"torchscript", "pair-curator"},
        "lammps_mliap_wrapper": format in {"mliap", "mliap-kk"},
        "kokkos_mliap": format == "mliap-kk",
    }


def _source_uncertainty_manifest(model_path: str) -> Optional[Dict[str, Any]]:
    manifest_path = Path(model_path).expanduser().resolve().parent / "uncertainty_manifest.json"
    if not manifest_path.exists():
        return None
    return read_json(manifest_path)


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


def deploy_model(
    model_path: str,
    format: str = "torchscript",
    element_types: Optional[List[str]] = None,
    out: str = "deployed_model",
    load_weights_only: bool = False,
    cfg_path: Optional[str] = None,
    timeout_sec: int = 1800,
) -> Dict[str, Any]:
    """Deploy an existing model into a deployable model format."""

    run_dir = ensure_dir(out)
    artifacts: Dict[str, Any] = {}
    try:
        deploy_format = _normalize_format(format)
        target_path = _target_file(run_dir, deploy_format)
        source_manifest = _source_uncertainty_manifest(model_path)
        if (
            deploy_format in {"torchscript", "pair-curator"}
            and source_manifest is not None
            and not source_manifest.get("capabilities", {}).get("torchscript_exportable", True)
        ):
            artifacts["source_manifest"] = str(
                Path(model_path).expanduser().resolve().parent / "uncertainty_manifest.json"
            )
            return error_result(
                "IncompatibleDeployFormat",
                "The source uncertainty model is not TorchScript-exportable. "
                "Use inject_uncertainty_model(..., implementation='scriptable', kernel='gnn/local-gnn') "
                "before deploying to torchscript or pair-curator.",
                artifacts=artifacts,
                format=deploy_format,
            )
        request = {
            "model_path": str(model_path),
            "target_path": str(target_path),
            "load_weights_only": load_weights_only,
            "cfg_path": cfg_path,
            "return_model": False,
            "lammps_mliap": deploy_format in {"mliap", "mliap-kk"},
            "element_types": element_types,
        }
        request_path = run_dir / "deploy_model_request.json"
        artifacts["request"] = write_json(request_path, request)
        snippet = (
            "import json, sys; "
            "from pathlib import Path; "
            "from curator.commands.deploy import deploy; "
            "request=json.loads(Path(sys.argv[1]).read_text()); "
            "deploy(**request)"
        )
        command = [sys.executable, "-c", snippet, str(request_path)]
        completed, logs = _run_subprocess(
            command,
            run_dir=run_dir,
            timeout_sec=timeout_sec,
            stdout_name="deploy_model_stdout.txt",
            stderr_name="deploy_model_stderr.txt",
        )
        artifacts.update(logs)
        artifacts["model"] = str(target_path)
        manifest = {
            "created_at": utc_now(),
            "tool": "deploy_model",
            "ok": completed.returncode == 0 and target_path.exists(),
            "status": "completed" if completed.returncode == 0 and target_path.exists() else "failed",
            "source_model_path": str(model_path),
            "model_path": str(target_path),
            "format": deploy_format,
            "element_types": element_types,
            "source_uncertainty_manifest": source_manifest,
            "returncode": completed.returncode,
            "artifacts": artifacts,
            "capabilities": _format_capabilities(deploy_format),
            "target_status": path_status(target_path),
        }
        artifacts["manifest"] = write_json(run_dir / "deploy_manifest.json", manifest)

        if completed.returncode != 0:
            return error_result(
                "DeployFailed",
                f"Deploy subprocess failed with return code {completed.returncode}.",
                log_path=artifacts.get("stderr"),
                artifacts=artifacts,
                format=deploy_format,
                returncode=completed.returncode,
            )
        if not target_path.exists():
            return error_result(
                "MissingArtifact",
                f"Deploy completed but expected model artifact was not written: {target_path}",
                log_path=artifacts["stdout"],
                artifacts=artifacts,
                format=deploy_format,
                returncode=completed.returncode,
            )

        return {
            "ok": True,
            "status": "completed",
            "format": deploy_format,
            "model_path": str(target_path),
            "capabilities": _format_capabilities(deploy_format),
            "artifacts": artifacts,
            "returncode": completed.returncode,
        }
    except subprocess.TimeoutExpired as exc:
        artifacts["stdout"] = write_text(run_dir / "deploy_model_stdout.txt", exc.stdout or "")
        artifacts["stderr"] = write_text(run_dir / "deploy_model_stderr.txt", exc.stderr or "")
        return error_result(
            "Timeout",
            f"Deploy exceeded timeout_sec={timeout_sec}.",
            log_path=artifacts.get("stderr"),
            artifacts=artifacts,
            recoverable=True,
        )
    except Exception as exc:
        return error_result(type(exc).__name__, str(exc), artifacts=artifacts)
