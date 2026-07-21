from __future__ import annotations

import importlib.util
import importlib.metadata
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from curator.model.adapter import parse_external_model_spec

from .artifacts import ensure_dir, error_result, path_status, read_json, utc_now, write_json, write_text


_RUNTIME_BY_SCHEME = {
    "mace": "mace",
    "matgl": "matgl",
    "nequip": "nequip",
    "nequip_hf": "nequip",
    "nequip_net": "nequip",
    "allegro": "nequip",
    "allegro_net": "nequip",
    "esen": "fairchem",
    "eqv2": "fairchem",
    "orb": "orb",
    "sevennet": "sevennet",
    "mattersim": "mattersim",
}

_RUNTIME_BACKENDS = {
    "mace": ["mace"],
    "matgl": ["matgl"],
    "nequip": ["nequip", "nequip_hf", "nequip_net", "allegro", "allegro_net"],
    "fairchem": ["esen", "eqv2"],
    "orb": ["orb"],
    "sevennet": ["sevennet"],
    "mattersim": ["mattersim"],
}

_RUNTIME_MODULES = {
    "mace": ["mace"],
    "matgl": ["matgl"],
    "nequip": ["nequip"],
    "fairchem": ["fairchem"],
    "orb": ["orb_models"],
    "sevennet": ["sevenn"],
    "mattersim": ["mattersim"],
}

_RUNTIME_PACKAGES = {
    "mace": ["mace-torch"],
    "matgl": ["matgl>=4.0"],
    "nequip": ["nequip", "nequip-allegro", "huggingface-hub"],
    "fairchem": ["fairchem-core", "huggingface-hub"],
    "orb": ["orb-models==0.7.0", "matscipy"],
    "sevennet": ["sevenn"],
    "mattersim": ["mattersim"],
}

_NO_DEPS_SHARED_RUNTIMES = {"matgl"}

_ONLY_BINARY_SHARED_RUNTIMES = {"orb"}

_RUNTIME_PACKAGE_OVERRIDES = {
    "orb": ["dm-tree==0.1.10"],
}

_BASE_PACKAGE_CONSTRAINTS = [
    "torch",
    "torchaudio",
    "torchvision",
    "torch-geometric",
    "triton",
]

_BASE_PACKAGE_EXCLUDES = [
    "torch",
    "torchaudio",
    "torchvision",
    "torch-geometric",
    "triton",
    "cuda-bindings",
    "nvidia-cublas",
    "nvidia-cuda-cupti",
    "nvidia-cuda-nvrtc",
    "nvidia-cuda-runtime",
    "nvidia-cudnn-cu11",
    "nvidia-cudnn-cu12",
    "nvidia-cudnn-cu13",
    "nvidia-cufft",
    "nvidia-cufile",
    "nvidia-curand",
    "nvidia-cusolver",
    "nvidia-cusparse",
    "nvidia-cusparselt-cu12",
    "nvidia-cusparselt-cu13",
    "nvidia-nccl-cu11",
    "nvidia-nccl-cu12",
    "nvidia-nccl-cu13",
    "nvidia-nvjitlink",
    "nvidia-nvshmem-cu12",
    "nvidia-nvshmem-cu13",
]


@dataclass
class BackendRuntimeSpec:
    runtime: str
    project_dir: str
    backends: list[str]
    command: list[str]
    execution_mode: str
    python: Optional[str]
    required_modules: list[str]
    host_modules: dict[str, bool]
    host_usable: bool
    host_diagnostics: dict[str, Any]
    exists: bool
    pyproject_exists: bool
    venv_exists: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime_root(runtime_root: Optional[str] = None) -> Path:
    return Path(runtime_root).expanduser().resolve() if runtime_root else _repo_root() / "runtimes"


def _venv_python(project_dir: Path) -> Path:
    return project_dir / ".venv" / "bin" / "python"


def _module_status(modules: list[str]) -> dict[str, bool]:
    return {module: importlib.util.find_spec(module) is not None for module in modules}


def _version_at_least(installed: str, minimum: str) -> bool:
    try:
        from packaging.version import Version

        return Version(installed) >= Version(minimum)
    except Exception:
        def parts(value: str) -> tuple[int, ...]:
            nums: list[int] = []
            for part in value.replace("-", ".").split("."):
                if not part.isdigit():
                    break
                nums.append(int(part))
            return tuple(nums)

        return parts(installed) >= parts(minimum)


def _host_runtime_diagnostics(runtime: str) -> dict[str, Any]:
    modules = list(_RUNTIME_MODULES.get(runtime, []))
    module_status = _module_status(modules)
    checks: list[dict[str, Any]] = []
    usable = bool(modules) and all(module_status.values())
    versions: dict[str, Optional[str]] = {}

    for module in modules:
        try:
            versions[module] = importlib.metadata.version(module.replace("_", "-"))
        except Exception:
            versions[module] = None

    if runtime == "matgl" and usable:
        version = versions.get("matgl")
        if version is None or not _version_at_least(version, "4.0.0"):
            usable = False
            checks.append(
                {
                    "name": "matgl_version",
                    "ok": False,
                    "message": f"MatGL >=4.0 is required for current pretrained model loading; found {version or 'unknown'}.",
                }
            )
        else:
            checks.append({"name": "matgl_version", "ok": True, "version": version})
        try:
            from matgl.ext.ase import PESCalculator  # noqa: F401

            checks.append({"name": "matgl_ext_ase", "ok": True})
        except Exception as exc:
            usable = False
            checks.append(
                {
                    "name": "matgl_ext_ase",
                    "ok": False,
                    "message": f"{exc.__class__.__name__}: {exc}",
                }
            )

    return {
        "modules": module_status,
        "versions": versions,
        "checks": checks,
        "usable": usable,
    }


def _python_module_status(python: str, modules: list[str]) -> dict[str, bool]:
    script = (
        "import importlib.util, json, sys; "
        "mods=sys.argv[1:]; "
        "print(json.dumps({m: importlib.util.find_spec(m) is not None for m in mods}))"
    )
    try:
        completed = subprocess.run(
            [python, "-c", script, *modules],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception:
        return {module: False for module in modules}
    if completed.returncode != 0:
        return {module: False for module in modules}
    try:
        import json

        payload = json.loads(completed.stdout)
    except Exception:
        return {module: False for module in modules}
    return {module: bool(payload.get(module)) for module in modules}


def _python_package_versions(python: str, packages: list[str]) -> dict[str, Optional[str]]:
    script = (
        "import importlib.metadata as metadata, json, sys; "
        "versions={}; "
        "\nfor package in sys.argv[1:]:\n"
        "    try:\n"
        "        versions[package] = metadata.version(package)\n"
        "    except Exception:\n"
        "        versions[package] = None\n"
        "print(json.dumps(versions))"
    )
    try:
        completed = subprocess.run(
            [python, "-c", script, *packages],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception:
        return {package: None for package in packages}
    if completed.returncode != 0:
        return {package: None for package in packages}
    try:
        import json

        payload = json.loads(completed.stdout)
    except Exception:
        return {package: None for package in packages}
    return {package: payload.get(package) for package in packages}


def _python_can_import(python: str, statement: str) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            [python, "-c", statement],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception as exc:
        return False, f"{exc.__class__.__name__}: {exc}"
    if completed.returncode == 0:
        return True, ""
    return False, (completed.stderr or completed.stdout).strip()


def _process_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _python_runtime_available(python: str, runtime: str) -> bool:
    modules = list(_RUNTIME_MODULES.get(runtime, []))
    if not modules or not all(_python_module_status(python, modules).values()):
        return False
    if runtime == "matgl":
        version = _python_package_versions(python, ["matgl"]).get("matgl")
        if version is None or not _version_at_least(version, "4.0.0"):
            return False
        ok, _ = _python_can_import(python, "from matgl.ext.ase import PESCalculator")
        return ok
    return True


def _host_runtime_available(runtime: str) -> bool:
    return bool(_host_runtime_diagnostics(runtime).get("usable"))


def _runtime_python(project_dir: Path, runtime: str) -> tuple[str, Optional[str]]:
    if _host_runtime_available(runtime):
        return "host", sys.executable
    venv_python = _venv_python(project_dir)
    if venv_python.exists() and _python_runtime_available(str(venv_python), runtime):
        return "venv", str(venv_python)
    return "missing", None


def _runtime_for_backend_or_spec(backend: Optional[str], adapter_spec: Optional[str]) -> Optional[str]:
    if backend:
        normalized = str(backend).strip().lower()
        if normalized in _RUNTIME_BACKENDS:
            return normalized
        return _RUNTIME_BY_SCHEME.get(normalized)
    if adapter_spec:
        parsed = parse_external_model_spec(adapter_spec)
        if parsed is not None:
            return _RUNTIME_BY_SCHEME.get(parsed.scheme)
        try:
            from curator.model.foundation import list_foundation_models

            token = str(adapter_spec).strip().lower()
            listing = list_foundation_models(potential_only=False, include_dynamic=False)
            for entry in listing.get("candidates", []):
                aliases = [str(item).lower() for item in entry.get("aliases", [])]
                if token == str(entry.get("id", "")).lower() or token in aliases:
                    parsed_entry = parse_external_model_spec(str(entry.get("adapter_spec", "")))
                    if parsed_entry is not None:
                        return _RUNTIME_BY_SCHEME.get(parsed_entry.scheme)
        except Exception:
            return None
    return None


def runtime_spec(
    runtime: str,
    *,
    runtime_root: Optional[str] = None,
    uv_executable: str = "uv",
) -> BackendRuntimeSpec:
    normalized = str(runtime).strip().lower()
    project_dir = _runtime_root(runtime_root) / normalized
    execution_mode, python_executable = _runtime_python(project_dir, normalized)
    command = (
        [python_executable, "-m", "curator_mcp.backend_runner"]
        if python_executable is not None
        else []
    )
    modules = list(_RUNTIME_MODULES.get(normalized, []))
    host_diagnostics = _host_runtime_diagnostics(normalized)
    return BackendRuntimeSpec(
        runtime=normalized,
        project_dir=str(project_dir),
        backends=list(_RUNTIME_BACKENDS.get(normalized, [])),
        command=command,
        execution_mode=execution_mode,
        python=python_executable,
        required_modules=modules,
        host_modules=_module_status(modules),
        host_usable=bool(host_diagnostics.get("usable")),
        host_diagnostics=host_diagnostics,
        exists=project_dir.exists(),
        pyproject_exists=(project_dir / "pyproject.toml").exists(),
        venv_exists=_venv_python(project_dir).exists(),
    )


def list_backend_runtimes(
    runtime_root: Optional[str] = None,
    uv_executable: str = "uv",
) -> dict[str, Any]:
    runtimes = [runtime_spec(name, runtime_root=runtime_root, uv_executable=uv_executable) for name in sorted(_RUNTIME_BACKENDS)]
    return {
        "ok": True,
        "status": "completed",
        "created_at": utc_now(),
        "runtime_root": str(_runtime_root(runtime_root)),
        "uv_executable": uv_executable,
        "uv_available": shutil.which(uv_executable) is not None,
        "runtimes": [asdict(item) for item in runtimes],
    }


def sync_backend_runtime(
    runtime: str,
    *,
    runtime_root: Optional[str] = None,
    uv_executable: str = "uv",
    uv_cache_dir: Optional[str] = None,
    mode: str = "shared-torch",
    python_executable: Optional[str] = None,
    base_python: Optional[str] = None,
    timeout_sec: int = 3600,
    out: str = "backend_runtime_sync",
) -> dict[str, Any]:
    run_dir = ensure_dir(out)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    result_path = run_dir / "result.json"
    spec = runtime_spec(runtime, runtime_root=runtime_root, uv_executable=uv_executable)
    normalized = spec.runtime
    normalized_mode = str(mode).strip().lower().replace("_", "-")
    project_dir = Path(spec.project_dir)
    venv_python = _venv_python(project_dir)
    python_executable = base_python or python_executable or sys.executable
    packages = list(_RUNTIME_PACKAGES.get(normalized, []))
    artifacts = {
        "result": str(result_path),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    constraints_path = run_dir / "base_package_constraints.txt"
    excludes_path = run_dir / "base_package_excludes.txt"

    if normalized_mode not in {"shared-torch", "shared-base", "isolated"}:
        result = error_result(
            "ValueError",
            "mode must be 'shared-torch', 'shared-base', or 'isolated'.",
            artifacts=artifacts,
            runtime=asdict(spec),
            mode=mode,
        )
        write_json(result_path, result)
        return result

    if spec.execution_mode == "host" and normalized_mode != "isolated":
        result = {
            "ok": True,
            "status": "completed",
            "created_at": utc_now(),
            "runtime": asdict(spec),
            "execution_mode": "host",
            "message": "Backend modules are already available in the host Python environment; no runtime install was needed.",
            "command": [],
            "artifacts": {key: path_status(value) for key, value in artifacts.items()},
        }
        write_json(result_path, result)
        return result

    if shutil.which(uv_executable) is None:
        result = error_result(
            "RuntimeUnavailable",
            f"uv executable {uv_executable!r} was not found.",
            artifacts=artifacts,
            runtime=asdict(spec),
        )
        write_json(result_path, result)
        return result
    if not spec.pyproject_exists:
        result = error_result(
            "RuntimeUnavailable",
            f"Backend runtime project is missing pyproject.toml: {Path(spec.project_dir) / 'pyproject.toml'}",
            artifacts=artifacts,
            runtime=asdict(spec),
        )
        write_json(result_path, result)
        return result
    if not packages:
        result = error_result(
            "RuntimeUnavailable",
            f"Unknown backend runtime {runtime!r}; no install package list is configured.",
            artifacts=artifacts,
            runtime=asdict(spec),
        )
        write_json(result_path, result)
        return result

    if normalized_mode == "isolated":
        command = [uv_executable, "sync", "--project", spec.project_dir]
        try:
            env = os.environ.copy()
            env.setdefault("UV_HTTP_TIMEOUT", "300")
            env.setdefault("UV_LINK_MODE", "hardlink")
            if uv_cache_dir:
                env["UV_CACHE_DIR"] = str(Path(uv_cache_dir).expanduser())
            completed = subprocess.run(
                command,
                cwd=str(_repo_root()),
                env=env,
                capture_output=True,
                text=True,
                timeout=int(timeout_sec),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            write_text(stdout_path, _process_text(exc.stdout))
            write_text(stderr_path, _process_text(exc.stderr))
            result = error_result(
                "TimeoutExpired",
                f"Isolated backend runtime sync timed out after {timeout_sec} seconds.",
                artifacts=artifacts,
                runtime=asdict(spec),
                command=command,
                timeout_sec=timeout_sec,
            )
            write_json(result_path, result)
            return result
        write_text(stdout_path, completed.stdout)
        write_text(stderr_path, completed.stderr)
        refreshed_spec = runtime_spec(normalized, runtime_root=runtime_root, uv_executable=uv_executable)
        result = {
            "ok": completed.returncode == 0,
            "status": "completed" if completed.returncode == 0 else "failed",
            "created_at": utc_now(),
            "runtime": asdict(refreshed_spec),
            "execution_mode": refreshed_spec.execution_mode,
            "mode": normalized_mode,
            "command": command,
            "returncode": completed.returncode,
            "artifacts": {key: path_status(value) for key, value in artifacts.items()},
        }
        if completed.returncode != 0:
            result["error_info"] = {
                "type": "UvSyncFailed",
                "message": f"uv sync failed with return code {completed.returncode}.",
                "recoverable": True,
                "log_path": str(stderr_path),
            }
        write_json(result_path, result)
        return result

    base_status = _python_module_status(python_executable, ["torch"])
    if not base_status.get("torch"):
        result = error_result(
            "BaseRuntimeUnavailable",
            f"Base Python {python_executable!r} does not provide torch. Install torch/CUDA once in the base environment, or pass base_python pointing at an environment that already has torch.",
            artifacts=artifacts,
            runtime=asdict(spec),
            base_python=python_executable,
            base_modules=base_status,
        )
        write_json(result_path, result)
        return result

    venv_command = [
        uv_executable,
        "venv",
        str(project_dir / ".venv"),
        "--system-site-packages",
        "--python",
        python_executable,
        "--allow-existing",
        "--no-python-downloads",
    ]
    install_command = [
        uv_executable,
        "pip",
        "install",
        "--python",
        str(venv_python),
        *packages,
    ]
    if normalized in _ONLY_BINARY_SHARED_RUNTIMES:
        install_command.extend(["--only-binary", ":all:"])
    if normalized in _NO_DEPS_SHARED_RUNTIMES:
        install_command.insert(5, "--no-deps")
    package_overrides = _RUNTIME_PACKAGE_OVERRIDES.get(normalized, [])
    if package_overrides:
        overrides_path = run_dir / f"{normalized}_package_overrides.txt"
        write_text(overrides_path, "\n".join(package_overrides) + "\n")
        install_command.extend(["--overrides", str(overrides_path)])
        artifacts["package_overrides"] = str(overrides_path)
    write_text(excludes_path, "\n".join(_BASE_PACKAGE_EXCLUDES) + "\n")
    install_command.extend(["--excludes", str(excludes_path)])
    base_versions = _python_package_versions(python_executable, _BASE_PACKAGE_CONSTRAINTS)
    constraints = [
        f"{package}=={version}"
        for package, version in base_versions.items()
        if version
    ]
    write_text(constraints_path, "\n".join(constraints) + ("\n" if constraints else ""))
    if constraints:
        install_command.extend(["--constraints", str(constraints_path)])
    artifacts["base_package_constraints"] = str(constraints_path)
    artifacts["base_package_excludes"] = str(excludes_path)
    commands = [venv_command, install_command]

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    completed_commands: list[dict[str, Any]] = []
    try:
        env = os.environ.copy()
        env.setdefault("UV_HTTP_TIMEOUT", "300")
        env.setdefault("UV_LINK_MODE", "hardlink")
        env["PATH"] = f"{venv_python.parent}{os.pathsep}{env.get('PATH', '')}"
        if uv_cache_dir:
            env["UV_CACHE_DIR"] = str(Path(uv_cache_dir).expanduser())
        for command in commands:
            completed = subprocess.run(
                command,
                cwd=str(_repo_root()),
                env=env,
                capture_output=True,
                text=True,
                timeout=int(timeout_sec),
                check=False,
            )
            stdout_chunks.append(completed.stdout)
            stderr_chunks.append(completed.stderr)
            completed_commands.append({"command": command, "returncode": completed.returncode})
            if completed.returncode != 0:
                break
    except subprocess.TimeoutExpired as exc:
        stdout_chunks.append(_process_text(exc.stdout))
        stderr_chunks.append(_process_text(exc.stderr))
        write_text(stdout_path, "".join(stdout_chunks))
        write_text(stderr_path, "".join(stderr_chunks))
        result = error_result(
            "TimeoutExpired",
            f"Backend runtime install timed out after {timeout_sec} seconds.",
            artifacts=artifacts,
            runtime=asdict(spec),
            commands=commands,
            timeout_sec=timeout_sec,
        )
        write_json(result_path, result)
        return result

    write_text(stdout_path, "".join(stdout_chunks))
    write_text(stderr_path, "".join(stderr_chunks))
    returncode = completed_commands[-1]["returncode"] if completed_commands else 1
    refreshed_spec = runtime_spec(normalized, runtime_root=runtime_root, uv_executable=uv_executable)
    result = {
        "ok": returncode == 0,
        "status": "completed" if returncode == 0 else "failed",
        "created_at": utc_now(),
        "runtime": asdict(refreshed_spec),
        "execution_mode": refreshed_spec.execution_mode,
        "base_python": python_executable,
        "shared_base_packages": base_versions,
        "commands": completed_commands,
        "returncode": returncode,
        "artifacts": {key: path_status(value) for key, value in artifacts.items()},
    }
    if returncode != 0:
        result["error_info"] = {
            "type": "RuntimeInstallFailed",
            "message": f"Backend runtime install failed with return code {returncode}.",
            "recoverable": True,
            "log_path": str(stderr_path),
        }
    write_json(result_path, result)
    result["artifacts"] = {key: path_status(value) for key, value in artifacts.items()}
    write_json(result_path, result)
    return result


def run_backend_runtime(
    *,
    task: str,
    model_id: Optional[str] = None,
    adapter_spec: Optional[str] = None,
    backend: Optional[str] = None,
    params: Optional[Mapping[str, Any]] = None,
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
) -> dict[str, Any]:
    run_dir = ensure_dir(out)
    artifacts: dict[str, Any] = {}
    request_path = run_dir / "request.json"
    result_path = run_dir / "result.json"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"

    resolved_runtime = runtime or _runtime_for_backend_or_spec(backend, adapter_spec or model_id)
    if resolved_runtime is None:
        result = error_result(
            "ValueError",
            "Could not infer backend runtime. Pass runtime or backend, or use an adapter_spec with a known scheme.",
            artifacts={"request": str(request_path), "result": str(result_path)},
        )
        write_json(result_path, result)
        return result

    spec = runtime_spec(resolved_runtime, runtime_root=runtime_root, uv_executable=uv_executable)
    request = {
        "task": str(task),
        "model_id": model_id,
        "adapter_spec": adapter_spec,
        "backend": backend,
        "params": dict(params or {}),
        "atoms_path": atoms_path,
        "device": device,
        "download": bool(download),
        "runtime": resolved_runtime,
        "model_cache": model_cache,
        "output_dir": str(run_dir),
        "timeout_sec": int(timeout_sec),
        "created_at": utc_now(),
    }
    artifacts["request"] = write_json(request_path, request)

    if spec.python is None:
        result = error_result(
            "RuntimeUnavailable",
            f"Backend runtime {resolved_runtime!r} is not available in host Python and no runtime venv exists. Run sync_backend_runtime({resolved_runtime!r}) first.",
            artifacts={**artifacts, "result": str(result_path)},
            runtime=asdict(spec),
        )
        write_json(result_path, result)
        return result
    if not spec.pyproject_exists:
        result = error_result(
            "RuntimeUnavailable",
            f"Backend runtime project is missing pyproject.toml: {Path(spec.project_dir) / 'pyproject.toml'}",
            artifacts={**artifacts, "result": str(result_path)},
            runtime=asdict(spec),
        )
        write_json(result_path, result)
        return result

    command = [*spec.command, "--request", str(request_path), "--out", str(run_dir)]
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(_repo_root())
        if not env.get("PYTHONPATH")
        else f"{_repo_root()}{os.pathsep}{env['PYTHONPATH']}"
    )
    try:
        completed = subprocess.run(
            command,
            cwd=spec.project_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=int(timeout_sec),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        write_text(stdout_path, _process_text(exc.stdout))
        write_text(stderr_path, _process_text(exc.stderr))
        result = error_result(
            "TimeoutExpired",
            f"Backend runtime timed out after {timeout_sec} seconds.",
            artifacts={
                **artifacts,
                "result": str(result_path),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            },
            runtime=asdict(spec),
            command=command,
            timeout_sec=timeout_sec,
        )
        write_json(result_path, result)
        return result

    write_text(stdout_path, completed.stdout)
    write_text(stderr_path, completed.stderr)
    artifacts.update({"result": str(result_path), "stdout": str(stdout_path), "stderr": str(stderr_path)})

    if result_path.exists():
        result = read_json(result_path)
        result.setdefault("runtime", asdict(spec))
        result.setdefault("command", command)
        result.setdefault("artifacts", {}).update({key: path_status(value) for key, value in artifacts.items()})
        return result

    result = error_result(
        "RuntimeProcessFailed" if completed.returncode else "MissingResult",
        "Backend runtime did not write result.json.",
        artifacts=artifacts,
        runtime=asdict(spec),
        command=command,
        returncode=completed.returncode,
    )
    write_json(result_path, result)
    return result


__all__ = [
    "BackendRuntimeSpec",
    "list_backend_runtimes",
    "run_backend_runtime",
    "runtime_spec",
    "sync_backend_runtime",
]
