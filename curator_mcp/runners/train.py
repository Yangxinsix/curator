from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from omegaconf import DictConfig, OmegaConf

from curator.utils import find_best_model, read_user_config

from .artifacts import ensure_dir, error_result, path_status, read_json, utc_now, write_json


_DISTILL_OUTPUT_GROUPS = {
    "energy_force": "energy_force_distill",
    "energy_forces": "energy_force_distill",
    "energy_force_distill": "energy_force_distill",
    "energy_force_hessian": "energy_force_hessian_distill",
    "energy_force_hessian_distill": "energy_force_hessian_distill",
    "hessian": "energy_force_hessian_distill",
    "energy_force_teacher_hessian": "energy_force_teacher_hessian_distill",
    "energy_force_teacher_hessian_distill": "energy_force_teacher_hessian_distill",
    "teacher_hessian": "energy_force_teacher_hessian_distill",
    "energy_force_projected_hessian": "energy_force_projected_hessian_distill",
    "energy_force_projected_hessian_distill": "energy_force_projected_hessian_distill",
    "projected_hessian": "energy_force_projected_hessian_distill",
    "energy_force_teacher_projected_hessian": "energy_force_teacher_projected_hessian_distill",
    "energy_force_teacher_projected_hessian_distill": "energy_force_teacher_projected_hessian_distill",
    "teacher_projected_hessian": "energy_force_teacher_projected_hessian_distill",
    "energy_force_teacher_dynamic_projected_hessian": "energy_force_teacher_dynamic_projected_hessian_distill",
    "energy_force_teacher_dynamic_projected_hessian_distill": "energy_force_teacher_dynamic_projected_hessian_distill",
    "teacher_dynamic_projected_hessian": "energy_force_teacher_dynamic_projected_hessian_distill",
    "dynamic_projected_hessian": "energy_force_teacher_dynamic_projected_hessian_distill",
}

_STANDARD_OUTPUT_GROUPS = {
    "energy_force",
    "energy_force_pa",
    "energy_force_per_species",
    "energy_force_residual",
    "energy_force_virial",
    "energy_force_virial_pa",
    "energy_force_virial_per_species",
}


def _as_mapping(value: Optional[Mapping[str, Any]], name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=False)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return dict(value)


def _normalize_distill_outputs(outputs: Optional[str]) -> Optional[str]:
    if outputs is None:
        return None
    normalized = str(outputs).strip().lower()
    if normalized in {"", "none", "null", "false"}:
        return None
    if normalized not in _DISTILL_OUTPUT_GROUPS:
        raise ValueError(
            f"distill.outputs must be one of {sorted(_DISTILL_OUTPUT_GROUPS)}, got {outputs!r}."
        )
    return _DISTILL_OUTPUT_GROUPS[normalized]


def _normalize_outputs(outputs: Optional[str]) -> Optional[str]:
    if outputs is None:
        return None
    normalized = str(outputs).strip().lower()
    if normalized in _STANDARD_OUTPUT_GROUPS:
        return normalized
    if normalized in _DISTILL_OUTPUT_GROUPS:
        return _DISTILL_OUTPUT_GROUPS[normalized]
    raise ValueError(
        f"outputs must be one of {sorted(_STANDARD_OUTPUT_GROUPS | set(_DISTILL_OUTPUT_GROUPS))}, got {outputs!r}."
    )


def _resolve_model_path(base_model: Mapping[str, Any]) -> Any:
    if not base_model:
        return None
    source = (
        base_model.get("model_path")
        or base_model.get("adapter_spec")
        or base_model.get("checkpoint")
        or base_model.get("path")
        or base_model.get("model_id")
    )
    if source is None:
        return None
    mode = base_model.get("mode")
    transform = base_model.get("transform")
    if mode is None and transform is None:
        return source
    model_path: Dict[str, Any] = {"path": source}
    if mode is not None:
        model_path["mode"] = mode
    if transform is not None:
        model_path["transform"] = transform
    return model_path


def _deep_update(target: Dict[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _request_overrides(request: Mapping[str, Any]) -> list[str]:
    overrides: list[str] = []
    finetune = _as_mapping(request.get("finetune"), "finetune")
    finetune_mode = finetune.get("mode", finetune.get("strategy"))
    if finetune_mode is not None:
        normalized_finetune = str(finetune_mode).strip().lower()
        if normalized_finetune in {"", "none", "null", "false"}:
            normalized_finetune = "full"
        if normalized_finetune not in {"full", "head_only", "lora"}:
            raise ValueError(
                "finetune.mode must be one of ['full', 'head_only', 'lora'], "
                f"got {finetune_mode!r}."
            )
        overrides.append(f"finetune={normalized_finetune}")

    distill = _as_mapping(request.get("distill"), "distill")
    if bool(distill.get("enabled", False)):
        outputs_group = _normalize_distill_outputs(distill.get("outputs", "energy_force"))
        if outputs_group is not None:
            overrides.append(f"task/outputs={outputs_group}")
        overrides.append("task/distill=offline")
        return overrides

    outputs_group = _normalize_outputs(request.get("outputs"))
    if outputs_group is not None:
        overrides.append(f"task/outputs={outputs_group}")
    return overrides


def build_train_config(request: Mapping[str, Any]) -> DictConfig:
    """Build a full train DictConfig from a structured MCP request.

    This reuses Curator's Hydra defaults through ``read_user_config`` without
    invoking Hydra's CLI/job runtime.
    """

    cfg = read_user_config(
        None,
        config_path=str(request.get("config_path", "configs")),
        config_name=str(request.get("config_name", "train")),
        overrides=_request_overrides(request),
    )

    patch: Dict[str, Any] = {"cfg": None}
    out = request.get("out") or request.get("run_path")
    if out is not None:
        patch["run_path"] = str(out)
    if request.get("device") is not None:
        patch["device"] = request["device"]
    if request.get("seed") is not None:
        patch["seed"] = request["seed"]

    base_model = _as_mapping(request.get("base_model"), "base_model")
    model_path = _resolve_model_path(base_model)
    if model_path is not None:
        patch["model_path"] = model_path

    data = _as_mapping(request.get("data"), "data")
    training = _as_mapping(request.get("training"), "training")
    data_patch = dict(data)
    for source_key, target_key in (
        ("batch_size", "batch_size"),
        ("val_batch_size", "val_batch_size"),
        ("test_batch_size", "test_batch_size"),
        ("num_workers", "num_workers"),
        ("pin_memory", "pin_memory"),
    ):
        if source_key in training and target_key not in data_patch:
            data_patch[target_key] = training[source_key]
    if data_patch:
        patch["data"] = data_patch

    trainer_patch = dict(_as_mapping(training.get("trainer"), "training.trainer"))
    for source_key, target_key in (
        ("max_epochs", "max_epochs"),
        ("min_epochs", "min_epochs"),
        ("limit_train_batches", "limit_train_batches"),
        ("limit_val_batches", "limit_val_batches"),
        ("num_sanity_val_steps", "num_sanity_val_steps"),
        ("accelerator", "accelerator"),
        ("devices", "devices"),
        ("logger", "logger"),
        ("enable_model_summary", "enable_model_summary"),
        ("log_every_n_steps", "log_every_n_steps"),
        ("precision", "precision"),
        ("gradient_clip_val", "gradient_clip_val"),
    ):
        if source_key in training and target_key not in trainer_patch:
            trainer_patch[target_key] = training[source_key]
    if trainer_patch:
        patch["trainer"] = trainer_patch

    finetune = _as_mapping(request.get("finetune"), "finetune")
    if finetune:
        mode = finetune.get("mode", finetune.get("strategy"))
        if mode is not None:
            patch["finetune"] = str(mode)
        wrapper_patch: Dict[str, Any] = {}
        if str(mode or "").lower() == "lora":
            wrapper_patch["adapter"] = "lora"
        for source_key, target_key in (
            ("backend", "backend"),
            ("adapter", "adapter"),
            ("lora_rank", "lora_rank"),
            ("rank", "lora_rank"),
            ("lora_alpha", "lora_alpha"),
            ("alpha", "lora_alpha"),
            ("lora_freeze_base", "lora_freeze_base"),
            ("freeze_base", "lora_freeze_base"),
            ("lora_target_groups", "lora_target_groups"),
            ("target_groups", "lora_target_groups"),
        ):
            if source_key in finetune:
                wrapper_patch[target_key] = finetune[source_key]
        if wrapper_patch:
            patch["wrapper"] = wrapper_patch

    distill = _as_mapping(request.get("distill"), "distill")
    if bool(distill.get("enabled", False)):
        distill_patch: Dict[str, Any] = {}
        for source_key, target_key in (
            ("teacher_model", "teacher_model_path"),
            ("teacher_model_path", "teacher_model_path"),
            ("teacher_labels_path", "teacher_labels_path"),
            ("teacher_cfg", "teacher_cfg"),
            ("overwrite", "overwrite"),
            ("resume", "resume"),
            ("label_scope", "label_scope"),
        ):
            if source_key in distill:
                distill_patch[target_key] = distill[source_key]
        if distill_patch:
            patch.setdefault("task", {})["distill"] = distill_patch
        if "hessian_num_samples" in distill:
            patch.setdefault("task", {})["hessian_num_samples"] = distill["hessian_num_samples"]
        if "hessian_mask_key" in distill:
            patch.setdefault("task", {})["hessian_mask_key"] = distill["hessian_mask_key"]
        projected = _as_mapping(distill.get("projected_hessian"), "distill.projected_hessian")
        if projected:
            patch.setdefault("task", {})["projected_hessian"] = projected

    postprocess = _as_mapping(request.get("postprocess"), "postprocess")
    if "deploy" in postprocess:
        patch["deploy_model"] = bool(postprocess["deploy"])
    elif "deploy_model" in request:
        patch["deploy_model"] = bool(request["deploy_model"])

    raw_patch = _as_mapping(request.get("config_patch"), "config_patch")
    _deep_update(patch, raw_patch)

    return OmegaConf.merge(cfg, OmegaConf.create(patch))


def _safe_best_model(run_dir: Path) -> tuple[Optional[str], Optional[float]]:
    try:
        best = find_best_model(run_dir / "model_path")
    except Exception:
        best = None
    if best is None:
        return None, None
    path, metric = best
    return str(path), metric


def _find_metrics_csv(run_dir: Path) -> Optional[str]:
    candidates = sorted(run_dir.glob("lightning_logs/**/metrics.csv"))
    return str(candidates[-1]) if candidates else None


def _read_last_metrics(metrics_csv: Optional[str]) -> Dict[str, Any]:
    if metrics_csv is None:
        return {}
    try:
        with open(metrics_csv, newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if any(value for value in row.values())]
    except Exception:
        return {}
    if not rows:
        return {}
    last = {}
    for key, value in rows[-1].items():
        if value in (None, ""):
            continue
        try:
            last[key] = float(value)
        except ValueError:
            last[key] = value
    return last


def _distill_artifacts(run_dir: Path) -> Dict[str, Any]:
    distill_dir = run_dir / "distill_dataset"
    sqlite_paths = sorted(str(path) for path in distill_dir.glob("*.sqlite"))
    return {
        "enabled": bool(sqlite_paths),
        "directory": str(distill_dir) if distill_dir.exists() else None,
        "sqlite": sqlite_paths,
        "label_scope": (
            "dataset"
            if any(Path(path).name == "dataset.sqlite" for path in sqlite_paths)
            else "split"
            if sqlite_paths
            else None
        ),
    }


def _build_model_artifact(
    best_checkpoint: Optional[str],
    compiled_model: Optional[str] = None,
) -> Dict[str, Any]:
    primary = best_checkpoint or compiled_model
    return {
        "best_checkpoint": best_checkpoint,
        "compiled_model": compiled_model,
        "primary_model_path": primary,
        "load_as": {
            "base_model": (
                {"model_path": best_checkpoint, "mode": "weights"}
                if best_checkpoint is not None
                else {"model_path": compiled_model, "mode": "model"}
                if compiled_model is not None
                else None
            ),
            "teacher_model": (
                {"teacher_model_path": best_checkpoint}
                if best_checkpoint is not None
                else {"teacher_model_path": compiled_model}
                if compiled_model is not None
                else None
            ),
            "predict_model": {"model_path": primary} if primary is not None else None,
        },
    }


def _ensure_model_artifact(result: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), Mapping) else {}
    best_checkpoint = result.get("best_checkpoint") or artifacts.get("best_checkpoint")
    compiled_model = artifacts.get("compiled_model")
    result["model_artifact"] = _build_model_artifact(best_checkpoint, compiled_model)
    return result


def collect_train_result(out: str) -> Dict[str, Any]:
    run_dir = Path(out).expanduser().resolve()
    manifest_path = run_dir / "training_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if isinstance(manifest, dict) and "model_artifact" not in manifest:
            return _ensure_model_artifact(manifest)
        return manifest

    job_path = run_dir / "training_job.json"
    job = read_json(job_path) if job_path.exists() else {}
    best_checkpoint, best_metric = _safe_best_model(run_dir)
    metrics_csv = _find_metrics_csv(run_dir)
    artifacts = {
        "run_path": str(run_dir),
        "config": str(run_dir / "config.yaml") if (run_dir / "config.yaml").exists() else None,
        "training_log": str(run_dir / "training.log") if (run_dir / "training.log").exists() else None,
        "metrics_csv": metrics_csv,
        "best_checkpoint": best_checkpoint,
        "compiled_model": str(run_dir / "compiled_model.pt") if (run_dir / "compiled_model.pt").exists() else None,
        "manifest": str(manifest_path),
        "job": str(job_path) if job_path.exists() else None,
    }
    result = {
        "created_at": utc_now(),
        "tool": "train_model",
        "ok": best_checkpoint is not None or artifacts["compiled_model"] is not None,
        "status": job.get("status", "completed" if best_checkpoint else "unknown"),
        "run_path": str(run_dir),
        "best_checkpoint": best_checkpoint,
        "best_metric": best_metric,
        "last_metrics": _read_last_metrics(metrics_csv),
        "distill": _distill_artifacts(run_dir),
        "artifacts": artifacts,
        "artifact_status": {name: path_status(path) for name, path in artifacts.items() if path is not None},
    }
    return _ensure_model_artifact(result)


def _write_job(run_dir: Path, payload: Mapping[str, Any]) -> str:
    job_path = run_dir / "training_job.json"
    return write_json(job_path, dict(payload))


def _write_manifest(run_dir: Path, payload: Mapping[str, Any]) -> str:
    return write_json(run_dir / "training_manifest.json", dict(payload))


def _run_worker(request_path: str) -> int:
    request = read_json(request_path)
    run_dir = ensure_dir(request.get("out") or request.get("run_path") or "train_model")
    job_path = run_dir / "training_job.json"
    job = read_json(job_path) if job_path.exists() else {}
    job.update(
        {
            "status": "running",
            "started_at": job.get("started_at") or utc_now(),
            "pid": os.getpid(),
            "request": str(Path(request_path).expanduser().resolve()),
        }
    )
    _write_job(run_dir, job)

    try:
        cfg = build_train_config(request)
        OmegaConf.save(cfg, run_dir / "mcp_train_config.yaml", resolve=False)

        from curator.commands.train import run_train_config

        run_train_config(cfg)
        result = collect_train_result(str(run_dir))
        result.update({"ok": True, "status": "completed", "completed_at": utc_now()})
        result["artifacts"]["request"] = str(Path(request_path).expanduser().resolve())
        result["artifacts"]["mcp_config"] = str(run_dir / "mcp_train_config.yaml")
        _write_manifest(run_dir, result)
        job.update({"status": "completed", "completed_at": result["completed_at"], "manifest": result["artifacts"]["manifest"]})
        _write_job(run_dir, job)
        return 0
    except Exception as exc:
        error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        error_path = run_dir / "train_model_exception.txt"
        error_path.write_text(error_text, encoding="utf-8")
        result = error_result(
            type(exc).__name__,
            str(exc),
            recoverable=False,
            log_path=str(error_path),
            artifacts={
                "run_path": str(run_dir),
                "request": str(Path(request_path).expanduser().resolve()),
                "exception": str(error_path),
                "training_log": str(run_dir / "training.log") if (run_dir / "training.log").exists() else None,
                "job": str(job_path),
            },
            run_path=str(run_dir),
        )
        result["tool"] = "train_model"
        result["created_at"] = utc_now()
        _write_manifest(run_dir, result)
        job.update({"status": "failed", "completed_at": result["created_at"], "manifest": str(run_dir / "training_manifest.json")})
        _write_job(run_dir, job)
        return 1


def _is_process_alive(pid: Optional[int]) -> bool:
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def get_train_job(out: str) -> Dict[str, Any]:
    run_dir = Path(out).expanduser().resolve()
    job_path = run_dir / "training_job.json"
    if not job_path.exists():
        return error_result(
            "JobNotFound",
            f"No training job metadata found at {job_path}.",
            recoverable=True,
            artifacts={"job": str(job_path), "run_path": str(run_dir)},
        )
    job = read_json(job_path)
    manifest_path = run_dir / "training_manifest.json"
    status = job.get("status", "unknown")
    if manifest_path.exists() and status == "running":
        manifest = read_json(manifest_path)
        status = str(manifest.get("status", status))
        job["status"] = status
        _write_job(run_dir, job)
    elif status == "running" and not _is_process_alive(job.get("pid")):
        status = "unknown_finished"
        job["status"] = status
        _write_job(run_dir, job)
    return {
        "ok": status in {"running", "completed"},
        "status": status,
        "job": job,
        "run_path": str(run_dir),
        "artifacts": {
            "job": str(job_path),
            "manifest": str(manifest_path) if manifest_path.exists() else None,
            "stdout": job.get("stdout"),
            "stderr": job.get("stderr"),
            "training_log": str(run_dir / "training.log") if (run_dir / "training.log").exists() else None,
        },
    }


def cancel_train_job(out: str) -> Dict[str, Any]:
    run_dir = Path(out).expanduser().resolve()
    job_path = run_dir / "training_job.json"
    if not job_path.exists():
        return error_result("JobNotFound", f"No training job metadata found at {job_path}.")
    job = read_json(job_path)
    pid = job.get("pid")
    if not _is_process_alive(pid):
        job["status"] = "not_running"
        _write_job(run_dir, job)
        return {"ok": False, "status": "not_running", "job": job}
    os.kill(int(pid), signal.SIGTERM)
    job["status"] = "cancel_requested"
    job["cancel_requested_at"] = utc_now()
    _write_job(run_dir, job)
    return {"ok": True, "status": "cancel_requested", "job": job}


def start_train_model(
    base_model: Optional[Mapping[str, Any]] = None,
    data: Optional[Mapping[str, Any]] = None,
    finetune: Optional[Mapping[str, Any]] = None,
    distill: Optional[Mapping[str, Any]] = None,
    training: Optional[Mapping[str, Any]] = None,
    postprocess: Optional[Mapping[str, Any]] = None,
    outputs: Optional[str] = None,
    config_patch: Optional[Mapping[str, Any]] = None,
    out: str = "train_model",
    device: Optional[str] = None,
    seed: Optional[int] = None,
    run_async: bool = True,
    timeout_sec: int = 3600,
) -> Dict[str, Any]:
    run_dir = ensure_dir(out)
    request = {
        "base_model": dict(base_model or {}),
        "data": dict(data or {}),
        "finetune": dict(finetune or {}),
        "distill": dict(distill or {}),
        "training": dict(training or {}),
        "postprocess": dict(postprocess or {}),
        "outputs": outputs,
        "config_patch": dict(config_patch or {}),
        "out": str(run_dir),
        "device": device,
        "seed": seed,
    }
    request_path = run_dir / "train_model_request.json"
    stdout_path = run_dir / "train_model_stdout.txt"
    stderr_path = run_dir / "train_model_stderr.txt"
    write_json(request_path, request)

    command = [sys.executable, "-m", "curator_mcp.runners.train", "--request", str(request_path)]
    job = {
        "created_at": utc_now(),
        "status": "queued",
        "pid": None,
        "command": command,
        "cwd": str(Path.cwd()),
        "request": str(request_path),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "run_path": str(run_dir),
    }
    _write_job(run_dir, job)

    if run_async:
        stdout_handle = open(stdout_path, "w", encoding="utf-8")
        stderr_handle = open(stderr_path, "w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command,
                cwd=str(Path.cwd()),
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                start_new_session=True,
            )
        finally:
            stdout_handle.close()
            stderr_handle.close()
        job.update({"status": "running", "pid": process.pid, "started_at": utc_now()})
        _write_job(run_dir, job)
        return {
            "ok": True,
            "status": "running",
            "job_id": str(run_dir),
            "run_path": str(run_dir),
            "artifacts": {
                "request": str(request_path),
                "job": str(run_dir / "training_job.json"),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "manifest": str(run_dir / "training_manifest.json"),
            },
            "pid": process.pid,
        }

    try:
        with open(stdout_path, "w", encoding="utf-8") as stdout_handle, open(
            stderr_path, "w", encoding="utf-8"
        ) as stderr_handle:
            completed = subprocess.run(
                command,
                cwd=str(Path.cwd()),
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
    except subprocess.TimeoutExpired:
        job.update({"status": "timeout", "completed_at": utc_now()})
        _write_job(run_dir, job)
        return error_result(
            "TrainingTimeout",
            f"Training did not finish within {timeout_sec} seconds.",
            recoverable=True,
            artifacts={
                "request": str(request_path),
                "job": str(run_dir / "training_job.json"),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "run_path": str(run_dir),
            },
            run_path=str(run_dir),
        )
    result = collect_train_result(str(run_dir))
    result["returncode"] = completed.returncode
    result.setdefault("artifacts", {})
    result["artifacts"].update(
        {
            "request": str(request_path),
            "job": str(run_dir / "training_job.json"),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
    )
    if completed.returncode != 0 and result.get("ok", False):
        result["ok"] = False
        result["status"] = "failed"
    return result


def _stage_ids(stages: list[Mapping[str, Any]]) -> list[str]:
    return [str(stage.get("id") or f"stage_{index}") for index, stage in enumerate(stages)]


def _validate_stage_ref(ref: str, available_ids: set[str]) -> bool:
    if "." not in ref:
        return False
    stage_id, _field = ref.split(".", 1)
    return stage_id in available_ids


def validate_train_workflow_spec(workflow_spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a train workflow spec before execution.

    The first implementation intentionally supports ordered, local train
    stages. ``depends_on`` may reference earlier stages, and artifact refs use
    strings such as ``finetune_teacher.best_checkpoint``.
    """

    if not isinstance(workflow_spec, Mapping):
        return error_result("InvalidWorkflowSpec", "workflow_spec must be a mapping.")

    stages = workflow_spec.get("stages")
    if not isinstance(stages, list) or not stages:
        return error_result(
            "InvalidWorkflowSpec",
            "workflow_spec.stages must be a non-empty list.",
            recoverable=True,
        )

    errors: list[Dict[str, Any]] = []
    warnings: list[Dict[str, Any]] = []
    seen: set[str] = set()
    ids = _stage_ids(stages)
    duplicates = sorted({stage_id for stage_id in ids if ids.count(stage_id) > 1})
    if duplicates:
        errors.append({"type": "DuplicateStageId", "stage_ids": duplicates})

    for index, raw_stage in enumerate(stages):
        if not isinstance(raw_stage, Mapping):
            errors.append({"type": "InvalidStage", "index": index, "message": "stage must be a mapping"})
            continue

        stage_id = ids[index]
        stage_type = str(raw_stage.get("type", "train"))
        if stage_type != "train":
            errors.append(
                {
                    "type": "UnsupportedStageType",
                    "stage_id": stage_id,
                    "message": "Only type='train' is supported in this workflow runner.",
                }
            )

        depends_on = raw_stage.get("depends_on", [])
        if depends_on is None:
            depends_on = []
        if not isinstance(depends_on, list):
            errors.append({"type": "InvalidDependsOn", "stage_id": stage_id})
        else:
            missing = [dep for dep in depends_on if dep not in seen]
            if missing:
                errors.append(
                    {
                        "type": "InvalidDependency",
                        "stage_id": stage_id,
                        "missing_or_not_previous": missing,
                    }
                )

        available = set(seen)
        for ref_key in ("base_model_from",):
            ref = raw_stage.get(ref_key)
            if ref is not None and not _validate_stage_ref(str(ref), available):
                errors.append({"type": "InvalidStageReference", "stage_id": stage_id, "field": ref_key, "ref": ref})

        distill = raw_stage.get("distill")
        if isinstance(distill, Mapping) and distill.get("teacher_model_from") is not None:
            ref = str(distill["teacher_model_from"])
            if not _validate_stage_ref(ref, available):
                errors.append(
                    {
                        "type": "InvalidStageReference",
                        "stage_id": stage_id,
                        "field": "distill.teacher_model_from",
                        "ref": ref,
                    }
                )
        if isinstance(distill, Mapping) and bool(distill.get("enabled", False)):
            has_teacher_source = any(
                distill.get(key) is not None
                for key in ("teacher_model_path", "teacher_model", "teacher_model_from", "teacher_labels_path")
            )
            if not has_teacher_source:
                errors.append(
                    {
                        "type": "MissingDistillTeacher",
                        "stage_id": stage_id,
                        "message": (
                            "distill.enabled=true requires teacher_model_path, "
                            "teacher_model_from, or teacher_labels_path."
                        ),
                    }
                )
            try:
                _normalize_distill_outputs(distill.get("outputs", "energy_force"))
            except ValueError as exc:
                errors.append({"type": "InvalidDistillOutputs", "stage_id": stage_id, "message": str(exc)})

        finetune = raw_stage.get("finetune")
        if isinstance(finetune, Mapping):
            mode = finetune.get("mode", finetune.get("strategy"))
            if mode is not None and str(mode).lower() not in {"full", "head_only", "lora"}:
                errors.append({"type": "InvalidFinetuneMode", "stage_id": stage_id, "mode": mode})

        defaults = workflow_spec.get("defaults", {})
        defaults_data = defaults.get("data") if isinstance(defaults, Mapping) else None
        if not raw_stage.get("data") and not defaults_data:
            warnings.append(
                {
                    "type": "MissingData",
                    "stage_id": stage_id,
                    "message": "No stage data or workflow defaults.data provided.",
                }
            )

        seen.add(stage_id)

    return {
        "ok": not errors,
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "warnings": warnings,
        "stage_ids": ids,
        "supported_stage_types": ["train"],
        "supported_reference_examples": [
            "previous_stage.best_checkpoint",
            "previous_stage.model_artifact.primary_model_path",
        ],
    }


def _lookup_stage_ref(stage_results: Mapping[str, Mapping[str, Any]], ref: str) -> Any:
    stage_id, field_path = ref.split(".", 1)
    if stage_id not in stage_results:
        raise KeyError(f"Stage reference {ref!r} uses unknown stage {stage_id!r}.")
    value: Any = stage_results[stage_id]
    for part in field_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"Stage reference {ref!r} cannot resolve field {part!r}.")
        value = value[part]
    return value


def _resolve_stage_refs(value: Any, stage_results: Mapping[str, Mapping[str, Any]]) -> Any:
    if isinstance(value, Mapping):
        return {key: _resolve_stage_refs(item, stage_results) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_stage_refs(item, stage_results) for item in value]
    if isinstance(value, str) and "." in value:
        stage_id = value.split(".", 1)[0]
        if stage_id in stage_results:
            return _lookup_stage_ref(stage_results, value)
    return value


def _merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _stage_request(
    stage: Mapping[str, Any],
    workflow_defaults: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage_out: Path,
) -> Dict[str, Any]:
    resolved_stage = _resolve_stage_refs(stage, stage_results)
    resolved_defaults = _resolve_stage_refs(workflow_defaults, stage_results)
    request: Dict[str, Any] = {}
    for key in (
        "base_model",
        "data",
        "finetune",
        "distill",
        "training",
        "postprocess",
        "outputs",
        "config_patch",
        "device",
        "seed",
    ):
        if isinstance(resolved_defaults.get(key), Mapping) or isinstance(resolved_stage.get(key), Mapping):
            request[key] = _merge_dicts(
                _as_mapping(resolved_defaults.get(key), f"defaults.{key}"),
                _as_mapping(resolved_stage.get(key), f"stage.{key}"),
            )
        elif key in resolved_stage:
            request[key] = resolved_stage[key]
        elif key in resolved_defaults:
            request[key] = resolved_defaults[key]

    base_model_from = resolved_stage.get("base_model_from")
    if base_model_from is not None:
        base_model = _as_mapping(request.get("base_model"), "base_model")
        base_model.setdefault("model_path", base_model_from)
        base_model.setdefault("mode", "weights")
        request["base_model"] = base_model

    distill = _as_mapping(request.get("distill"), "distill")
    teacher_model_from = distill.pop("teacher_model_from", None)
    if teacher_model_from is not None:
        distill["teacher_model_path"] = teacher_model_from
        distill.setdefault("enabled", True)
        request["distill"] = distill

    request["out"] = str(stage_out)
    return request


def _summarize_stage_result(stage_id: str, result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "id": stage_id,
        "ok": bool(result.get("ok", False)),
        "status": result.get("status"),
        "run_path": result.get("run_path"),
        "best_checkpoint": result.get("best_checkpoint"),
        "best_metric": result.get("best_metric"),
        "model_artifact": result.get("model_artifact"),
        "artifacts": result.get("artifacts", {}),
        "error": result.get("error"),
    }


def _execute_train_workflow(workflow_spec: Mapping[str, Any], out: str) -> Dict[str, Any]:
    workflow_dir = ensure_dir(out)
    validation = validate_train_workflow_spec(workflow_spec)
    if not validation.get("ok", False):
        result = error_result(
            "InvalidWorkflowSpec",
            "Train workflow spec failed validation.",
            recoverable=True,
            artifacts={"run_path": str(workflow_dir)},
        )
        result["validation"] = validation
        write_json(workflow_dir / "train_workflow_manifest.json", result)
        return result

    write_json(workflow_dir / "train_workflow_spec.json", dict(workflow_spec))
    job = {
        "created_at": utc_now(),
        "status": "running",
        "pid": os.getpid(),
        "run_path": str(workflow_dir),
        "manifest": str(workflow_dir / "train_workflow_manifest.json"),
    }
    write_json(workflow_dir / "train_workflow_job.json", job)

    execution = _as_mapping(workflow_spec.get("execution"), "execution")
    stop_on_stage_failure = bool(execution.get("stop_on_stage_failure", True))
    default_timeout = int(execution.get("timeout_sec", 3600))
    workflow_defaults = _as_mapping(workflow_spec.get("defaults"), "defaults")
    stage_results: Dict[str, Dict[str, Any]] = {}
    stage_summaries: list[Dict[str, Any]] = []

    stages = workflow_spec["stages"]
    ids = _stage_ids(stages)
    for index, stage in enumerate(stages):
        stage_id = ids[index]
        stage_out = Path(stage.get("out") or workflow_dir / f"{index:02d}_{stage_id}")
        stage_request = _stage_request(stage, workflow_defaults, stage_results, stage_out)
        stage_result = start_train_model(
            base_model=stage_request.get("base_model"),
            data=stage_request.get("data"),
            finetune=stage_request.get("finetune"),
            distill=stage_request.get("distill"),
            training=stage_request.get("training"),
            postprocess=stage_request.get("postprocess"),
            outputs=stage_request.get("outputs"),
            config_patch=stage_request.get("config_patch"),
            out=stage_request["out"],
            device=stage_request.get("device"),
            seed=stage_request.get("seed"),
            run_async=False,
            timeout_sec=int(stage.get("timeout_sec", default_timeout)),
        )
        stage_result = _ensure_model_artifact(dict(stage_result))
        stage_results[stage_id] = stage_result
        summary = _summarize_stage_result(stage_id, stage_result)
        stage_summaries.append(summary)
        write_json(workflow_dir / f"{index:02d}_{stage_id}_result.json", stage_result)

        if not stage_result.get("ok", False) and stop_on_stage_failure:
            break

    completed_all = len(stage_summaries) == len(stages) and all(stage["ok"] for stage in stage_summaries)
    final_stage = stage_summaries[-1] if stage_summaries else None
    result = {
        "created_at": utc_now(),
        "tool": "train_workflow",
        "ok": completed_all,
        "status": "completed" if completed_all else "failed",
        "workflow_id": workflow_spec.get("workflow_id"),
        "run_path": str(workflow_dir),
        "validation": validation,
        "stages": stage_summaries,
        "final_model_artifact": final_stage.get("model_artifact") if final_stage else None,
        "artifacts": {
            "run_path": str(workflow_dir),
            "spec": str(workflow_dir / "train_workflow_spec.json"),
            "job": str(workflow_dir / "train_workflow_job.json"),
            "manifest": str(workflow_dir / "train_workflow_manifest.json"),
        },
    }
    write_json(workflow_dir / "train_workflow_manifest.json", result)
    job.update({"status": result["status"], "completed_at": result["created_at"]})
    write_json(workflow_dir / "train_workflow_job.json", job)
    return result


def _run_workflow_worker(request_path: str) -> int:
    request = read_json(request_path)
    result = _execute_train_workflow(request["workflow_spec"], request["out"])
    return 0 if result.get("ok", False) else 1


def collect_train_workflow_result(out: str) -> Dict[str, Any]:
    workflow_dir = Path(out).expanduser().resolve()
    manifest_path = workflow_dir / "train_workflow_manifest.json"
    if manifest_path.exists():
        return read_json(manifest_path)
    return error_result(
        "WorkflowResultNotFound",
        f"No train workflow manifest found at {manifest_path}.",
        recoverable=True,
        artifacts={"manifest": str(manifest_path), "run_path": str(workflow_dir)},
    )


def get_train_workflow(out: str) -> Dict[str, Any]:
    workflow_dir = Path(out).expanduser().resolve()
    job_path = workflow_dir / "train_workflow_job.json"
    manifest_path = workflow_dir / "train_workflow_manifest.json"
    if not job_path.exists():
        return error_result(
            "WorkflowJobNotFound",
            f"No train workflow job metadata found at {job_path}.",
            recoverable=True,
            artifacts={"job": str(job_path), "run_path": str(workflow_dir)},
        )
    job = read_json(job_path)
    status = job.get("status", "unknown")
    if manifest_path.exists() and status == "running":
        manifest = read_json(manifest_path)
        status = manifest.get("status", status)
        job["status"] = status
        write_json(job_path, job)
    elif status == "running" and not _is_process_alive(job.get("pid")):
        status = "unknown_finished"
        job["status"] = status
        write_json(job_path, job)
    return {
        "ok": status in {"running", "completed"},
        "status": status,
        "job": job,
        "run_path": str(workflow_dir),
        "artifacts": {
            "job": str(job_path),
            "manifest": str(manifest_path) if manifest_path.exists() else None,
        },
    }


def start_train_workflow(
    workflow_spec: Mapping[str, Any],
    out: str = "train_workflow",
    run_async: bool = True,
    timeout_sec: int = 7200,
) -> Dict[str, Any]:
    workflow_dir = ensure_dir(out)
    validation = validate_train_workflow_spec(workflow_spec)
    if not validation.get("ok", False):
        return {
            "ok": False,
            "status": "invalid",
            "validation": validation,
            "run_path": str(workflow_dir),
            "artifacts": {"run_path": str(workflow_dir)},
        }

    request = {"workflow_spec": dict(workflow_spec), "out": str(workflow_dir)}
    request_path = workflow_dir / "train_workflow_request.json"
    stdout_path = workflow_dir / "train_workflow_stdout.txt"
    stderr_path = workflow_dir / "train_workflow_stderr.txt"
    write_json(request_path, request)

    command = [
        sys.executable,
        "-m",
        "curator_mcp.runners.train",
        "--workflow-request",
        str(request_path),
    ]
    job = {
        "created_at": utc_now(),
        "status": "queued",
        "pid": None,
        "command": command,
        "cwd": str(Path.cwd()),
        "request": str(request_path),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "run_path": str(workflow_dir),
    }
    write_json(workflow_dir / "train_workflow_job.json", job)

    if run_async:
        stdout_handle = open(stdout_path, "w", encoding="utf-8")
        stderr_handle = open(stderr_path, "w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command,
                cwd=str(Path.cwd()),
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                start_new_session=True,
            )
        finally:
            stdout_handle.close()
            stderr_handle.close()
        job.update({"status": "running", "pid": process.pid, "started_at": utc_now()})
        write_json(workflow_dir / "train_workflow_job.json", job)
        return {
            "ok": True,
            "status": "running",
            "job_id": str(workflow_dir),
            "run_path": str(workflow_dir),
            "validation": validation,
            "pid": process.pid,
            "artifacts": {
                "request": str(request_path),
                "job": str(workflow_dir / "train_workflow_job.json"),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "manifest": str(workflow_dir / "train_workflow_manifest.json"),
            },
        }

    try:
        with open(stdout_path, "w", encoding="utf-8") as stdout_handle, open(
            stderr_path, "w", encoding="utf-8"
        ) as stderr_handle:
            completed = subprocess.run(
                command,
                cwd=str(Path.cwd()),
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
    except subprocess.TimeoutExpired:
        job.update({"status": "timeout", "completed_at": utc_now()})
        write_json(workflow_dir / "train_workflow_job.json", job)
        return error_result(
            "TrainWorkflowTimeout",
            f"Train workflow did not finish within {timeout_sec} seconds.",
            recoverable=True,
            artifacts={
                "request": str(request_path),
                "job": str(workflow_dir / "train_workflow_job.json"),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "run_path": str(workflow_dir),
            },
            run_path=str(workflow_dir),
        )

    result = collect_train_workflow_result(str(workflow_dir))
    result["returncode"] = completed.returncode
    return result


def plan_training_strategy(
    objective: str = "",
    user_constraints: Optional[Mapping[str, Any]] = None,
    data: Optional[Mapping[str, Any]] = None,
    candidate_models: Optional[list[Mapping[str, Any]]] = None,
    out: str = "train_strategy_plan",
) -> Dict[str, Any]:
    """Create a rule-based train workflow proposal for an LLM/orchestrator to inspect."""

    constraints = _as_mapping(user_constraints, "user_constraints")
    data_spec = _as_mapping(data, "data")
    models = list(candidate_models or [])
    requested_order = str(constraints.get("required_order") or "").strip().lower()
    objective_text = objective.lower()
    reason_codes: list[str] = []

    if requested_order in {"finetune_then_distill", "finetune->distill", "finetune_distill"}:
        strategy = "finetune_then_distill"
        reason_codes.append("user_required_order")
    elif requested_order in {"finetune", "finetune_only"}:
        strategy = "finetune_only"
        reason_codes.append("user_required_finetune")
    elif requested_order in {"distill", "distill_only"}:
        strategy = "distill_only"
        reason_codes.append("user_required_distill")
    elif any(word in objective_text for word in ("distill", "compress", "student", "hessian")):
        strategy = "distill_only"
        reason_codes.append("objective_indicates_distillation")
    else:
        strategy = "finetune_only"
        reason_codes.append("default_direct_adaptation")

    if constraints.get("time_budget") in {"short", "small", "quick"}:
        finetune_mode = "lora"
        reason_codes.append("short_budget_prefers_lora")
    elif constraints.get("finetune_mode"):
        finetune_mode = str(constraints["finetune_mode"])
        reason_codes.append("user_selected_finetune_mode")
    else:
        finetune_mode = "head_only" if data_spec.get("num_structures", 0) and data_spec.get("num_structures", 0) < 100 else "lora"

    selected_model = models[0] if models else {}
    model_id = selected_model.get("model_id") or selected_model.get("id") or selected_model.get("name")
    base_model = {"model_id": model_id, "mode": "model"} if model_id else {}
    distill_outputs = constraints.get("distill_outputs") or (
        "energy_force_hessian" if constraints.get("allow_hessian") else "energy_force"
    )

    stages: list[Dict[str, Any]] = []
    if strategy == "finetune_then_distill":
        stages.append(
            {
                "id": "finetune_teacher",
                "type": "train",
                "base_model": base_model,
                "finetune": {"mode": finetune_mode},
                "distill": {"enabled": False},
            }
        )
        stages.append(
            {
                "id": "distill_student",
                "type": "train",
                "depends_on": ["finetune_teacher"],
                "base_model": dict(base_model),
                "finetune": {"mode": constraints.get("student_finetune_mode", "full")},
                "distill": {
                    "enabled": True,
                    "teacher_model_from": "finetune_teacher.best_checkpoint",
                    "outputs": distill_outputs,
                    "label_scope": constraints.get("label_scope", "dataset"),
                },
            }
        )
    elif strategy == "distill_only":
        stages.append(
            {
                "id": "distill_student",
                "type": "train",
                "base_model": base_model,
                "finetune": {"mode": constraints.get("student_finetune_mode", "full")},
                "distill": {
                    "enabled": True,
                    "teacher_model_path": constraints.get("teacher_model_path"),
                    "outputs": distill_outputs,
                    "label_scope": constraints.get("label_scope", "dataset"),
                },
            }
        )
    else:
        stages.append(
            {
                "id": "finetune_model",
                "type": "train",
                "base_model": base_model,
                "finetune": {"mode": finetune_mode},
                "distill": {"enabled": False},
            }
        )

    workflow_spec = {
        "workflow_id": constraints.get("workflow_id") or "planned_train_workflow",
        "execution": {
            "mode": constraints.get("execution_mode", "local"),
            "run_async": True,
            "stop_on_stage_failure": True,
        },
        "defaults": {
            "data": data_spec,
            "training": _as_mapping(constraints.get("training"), "user_constraints.training"),
            "postprocess": {"deploy": bool(constraints.get("deploy", False))},
            "device": constraints.get("device"),
        },
        "stages": stages,
    }
    plan = {
        "created_at": utc_now(),
        "tool": "plan_training_strategy",
        "ok": True,
        "status": "planned",
        "strategy": strategy,
        "reason_codes": reason_codes,
        "workflow_spec": workflow_spec,
        "validation": validate_train_workflow_spec(workflow_spec),
        "risks": [],
        "artifacts": {"plan": str(Path(out).expanduser().resolve() / "train_strategy_plan.json")},
    }
    plan_dir = ensure_dir(out)
    write_json(plan_dir / "train_strategy_plan.json", plan)
    return plan


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request")
    parser.add_argument("--workflow-request")
    args = parser.parse_args(argv)
    if args.workflow_request:
        return _run_workflow_worker(args.workflow_request)
    if args.request:
        return _run_worker(args.request)
    parser.error("one of --request or --workflow-request is required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
