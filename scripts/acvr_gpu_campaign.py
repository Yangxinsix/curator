#!/usr/bin/env python3
"""Persist a dependency graph of local, single-GPU jobs across scheduler restarts.

``launch`` detaches both the scheduler and its job runners from the caller.
``resume --state-dir DIR`` recovers live jobs without duplicating them, and retries
failed/interrupted jobs (using resume_command when a checkpoint exists). After a
machine reboot, resume explicitly; this script does not install a boot service.
Child runners persist their exit status independently of the scheduler.

Manifest jobs accept id/argv (or name/command), cwd, env, depends_on and optional
success_artifacts. Set gpu=0 for a CPU job; GPU jobs default to gpu=1 and can pin
gpu_uuid. resume_command replaces argv and substitutes {checkpoint}; legacy
resume_argv appends arguments. GPU UUIDs and scheduler settings are persisted.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid


TERMINAL = {"completed", "failed", "blocked", "cancelled"}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def event(directory, kind, **fields):
    payload = json.dumps({"time": time.time(), "event": kind, **fields}) + "\n"
    descriptor = os.open(directory / "events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, payload.encode())
    finally:
        os.close(descriptor)


def process_start(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return fields[19] if fields[0] != "Z" else None
    except (OSError, IndexError):
        return None


def owned_scheduler(process, directory):
    pid, start = process.get("pid"), process.get("starttime")
    if not pid or not start or process_start(pid) != start:
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return b"run" in command and str(directory).encode() in command
    except OSError:
        return False


def owned_runner(pid, start, token):
    if not pid or not start or process_start(pid) != start:
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return b"_job-runner" in command and token.encode() in command and os.getpgid(pid) == pid
    except (OSError, ProcessLookupError):
        return False


def owned_child(pid, start, token, group):
    if not pid or not start or process_start(pid) != start:
        return False
    try:
        environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        return f"ACVR_CAMPAIGN_TOKEN={token}".encode() in environment and os.getpgid(pid) == group
    except (OSError, ProcessLookupError):
        return False


def gpu_snapshot():
    def query(fields):
        completed = subprocess.run(
            ["nvidia-smi", fields, "--format=csv,noheader,nounits"],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
        )
        return list(csv.reader(completed.stdout.splitlines(), skipinitialspace=True))

    gpus = []
    for row in query("--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu"):
        if row:
            gpus.append({"index": int(row[0]), "uuid": row[1].strip(),
                         "memory_total_mb": int(row[2]), "memory_used_mb": int(row[3]),
                         "utilization_percent": int(row[4]), "compute_pids": []})
    by_uuid = {gpu["uuid"]: gpu for gpu in gpus}
    for row in query("--query-compute-apps=gpu_uuid,pid"):
        if row and row[0].strip() in by_uuid:
            by_uuid[row[0].strip()]["compute_pids"].append(int(row[1]))
    return gpus


def read_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("version", 1) != 1:
        raise ValueError("unsupported manifest version")
    jobs = manifest.get("jobs", [])
    if not jobs:
        raise ValueError("manifest requires at least one job")
    identifiers = set()
    for job in jobs:
        if "name" in job:
            if "id" in job and job["id"] != job["name"]:
                raise ValueError("job name and id disagree")
            job["id"] = job.pop("name")
        if "command" in job:
            if "argv" in job and job["argv"] != job["command"]:
                raise ValueError("job command and argv disagree")
            job["argv"] = job.pop("command")
        identifier = job.get("id", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", identifier) or identifier in identifiers:
            raise ValueError(f"invalid or duplicate job id: {identifier!r}")
        identifiers.add(identifier)
        if not isinstance(job.get("argv"), list) or not job["argv"] or not all(isinstance(v, str) for v in job["argv"]):
            raise ValueError(f"{identifier}: argv must be a nonempty list of strings")
        cwd = Path(job.get("cwd", path.parent))
        job["cwd"] = str((path.parent / cwd).resolve())
        if not Path(job["cwd"]).is_dir():
            raise ValueError(f"{identifier}: cwd does not exist: {job['cwd']}")
        for key in ("resume_checkpoint", "run_path"):
            if job.get(key):
                job[key] = str((Path(job["cwd"]) / job[key]).resolve())
        job["depends_on"] = list(job.get("depends_on", []))
        job["success_artifacts"] = [str((Path(job["cwd"]) / item).resolve())
                                    for item in job.get("success_artifacts", [])]
        if "resume_argv" in job and (not isinstance(job["resume_argv"], list)
                                    or not all(isinstance(v, str) for v in job["resume_argv"])):
            raise ValueError(f"{identifier}: resume_argv must be a list of strings")
        if "resume_command" in job and (not isinstance(job["resume_command"], list)
                or not job["resume_command"] or not all(isinstance(v, str) for v in job["resume_command"])):
            raise ValueError(f"{identifier}: resume_command must be a nonempty list of strings")
        if "resume_command" in job and "resume_argv" in job:
            raise ValueError(f"{identifier}: choose resume_command or resume_argv")
        if ("resume_command" in job or "resume_argv" in job) and not job.get("resume_checkpoint"):
            raise ValueError(f"{identifier}: checkpoint resume command requires resume_checkpoint")
        if job.get("gpu", 1) not in (0, 1):
            raise ValueError(f"{identifier}: gpu must be 0 (CPU) or 1 (single GPU)")
        if job.get("gpu_uuid") and (not isinstance(job["gpu_uuid"], str)
                                     or not job["gpu_uuid"].startswith("GPU-") or job.get("gpu", 1) == 0):
            raise ValueError(f"{identifier}: gpu_uuid requires a GPU UUID and gpu=1")
        if int(job.get("min_free_mb", 0)) < 0:
            raise ValueError(f"{identifier}: min_free_mb cannot be negative")
        if int(job.get("max_retries", 1)) < 0:
            raise ValueError(f"{identifier}: max_retries cannot be negative")
    for payload in [manifest, *jobs]:
        if not isinstance(payload.get("env", {}), dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in payload.get("env", {}).items()
        ):
            raise ValueError("env must map strings to strings")
    remaining = {job["id"]: set(job["depends_on"]) for job in jobs}
    if any(dependency not in identifiers for dependencies in remaining.values() for dependency in dependencies):
        raise ValueError("unknown job dependency")
    while remaining:
        ready = {name for name, dependencies in remaining.items() if not dependencies}
        if not ready:
            raise ValueError("job dependencies contain a cycle")
        remaining = {name: dependencies - ready for name, dependencies in remaining.items() if name not in ready}
    return manifest


def run_job(attempt_dir, token):
    attempt_dir = Path(attempt_dir)
    specification = json.loads((attempt_dir / "request.json").read_text())
    if specification["token"] != token:
        raise ValueError("runner token does not match its request")
    pid = os.getpid()
    atomic_json(attempt_dir / "process.json", {"pid": pid, "starttime": process_start(pid), "token": token})
    result = {"token": token, "started_at": time.time(), "runner_pid": pid}
    stopped = []
    signal.signal(signal.SIGTERM, lambda signum, frame: stopped.append(signum))
    signal.signal(signal.SIGINT, lambda signum, frame: stopped.append(signum))
    try:
        environment = dict(os.environ, **specification["env"])
        with (attempt_dir / "stdout.log").open("a") as output:
            # Pass the GPU lease to the actual job as well. If this runner is
            # killed, a surviving training process still holds its flock.
            lease = specification.get("gpu_lock_fd")
            child = subprocess.Popen(specification["argv"], cwd=specification["cwd"], env=environment,
                                     stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                     pass_fds=() if lease is None else (lease,))
            result["child_pid"] = child.pid
            atomic_json(attempt_dir / "child.json", {"pid": child.pid, "starttime": process_start(child.pid),
                        "group": pid, "token": token})
            if stopped:
                os.killpg(pid, signal.SIGTERM)
            result["exit_code"] = child.wait()
    except Exception as error:
        result.update(exit_code=127, error=str(error))
    result["stopped"] = bool(stopped)
    result["finished_at"] = time.time()
    atomic_json(attempt_dir / "result.json", result)
    return 0


def attempt_process(attempt):
    path = Path(attempt["directory"]) / "process.json"
    if path.exists():
        process = json.loads(path.read_text())
        if process.get("token") == attempt["token"]:
            attempt.update(pid=process["pid"], starttime=process["starttime"])
    if owned_runner(attempt.get("pid"), attempt.get("starttime"), attempt["token"]):
        return True
    child_path = Path(attempt["directory"]) / "child.json"
    if child_path.exists():
        child = json.loads(child_path.read_text())
        if child.get("token") == attempt["token"] and owned_child(
            child["pid"], child["starttime"], attempt["token"], child["group"]
        ):
            attempt["pid"] = child["group"]
            return True
    # Cover a crash between Popen and its durable PID record, including a child
    # whose wrapper was killed. Identity requires this attempt's random token.
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        start = process_start(pid)
        if owned_runner(pid, start, attempt["token"]):
            attempt.update(pid=pid, starttime=start)
            return True
        try:
            group = os.getpgid(pid)
        except ProcessLookupError:
            continue
        if owned_child(pid, start, attempt["token"], group):
            attempt["pid"] = group
            return True
    return False


def terminate_attempt(attempt, sig=signal.SIGTERM):
    if attempt_process(attempt):
        try:
            os.killpg(attempt["pid"], sig)
            return True
        except ProcessLookupError:
            pass
    return False


def scheduler_options(args, manifest, directory):
    """Freeze GPU UUIDs once; a later physical index reorder cannot retarget jobs."""
    path = directory / "scheduler-options.json"
    if path.exists():
        return json.loads(path.read_text())
    needs_gpu = any(job.get("gpu", 1) for job in manifest["jobs"])
    snapshot = gpu_snapshot() if needs_gpu else []
    available = {gpu["uuid"]: gpu for gpu in snapshot}
    by_index = {str(gpu["index"]): gpu["uuid"] for gpu in snapshot}
    requested = args.gpus.split(",") if args.gpus else list(available)
    allowed = []
    for value in requested:
        value = value.strip()
        gpu_uuid = by_index.get(value, value)
        if gpu_uuid not in available:
            raise ValueError(f"requested GPU is not present: {value}")
        if gpu_uuid not in allowed:
            allowed.append(gpu_uuid)
    for job in manifest["jobs"]:
        if job.get("gpu_uuid") and job["gpu_uuid"] not in allowed:
            raise ValueError(f"{job['id']}: gpu_uuid is outside this campaign's allowed GPUs")
    options = {"max_parallel": args.max_parallel, "max_retries": args.max_retries,
               "poll_seconds": args.poll_seconds, "min_free_mb": args.min_free_mb,
               "max_utilization": args.max_utilization, "gpu_uuids": allowed,
               "gpu_lock_dir": str(Path(args.gpu_lock_dir).resolve())}
    atomic_json(directory / "initial-gpu-snapshot.json", {"time": time.time(), "gpus": snapshot})
    atomic_json(path, options)
    return options


def acquire_gpu(gpu, job, options):
    """Take a cross-campaign lease, then recheck external GPU use before Popen."""
    lock_dir = Path(options["gpu_lock_dir"])
    lock_dir.mkdir(parents=True, exist_ok=True)
    lease = (lock_dir / (gpu["uuid"] + ".lock")).open("a")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = next((item for item in gpu_snapshot() if item["uuid"] == gpu["uuid"]), None)
        minimum = int(job.get("min_free_mb", options["min_free_mb"]))
        if (current is None or current["compute_pids"]
                or current["memory_total_mb"] - current["memory_used_mb"] < minimum
                or current["utilization_percent"] > options["max_utilization"]):
            lease.close()
            return None, current
        return lease, current
    except BlockingIOError:
        lease.close()
        return None, None
    except BaseException:
        lease.close()
        raise


def run_campaign(args):
    directory = Path(args.state_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "scheduler.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError("another scheduler holds this campaign lock")
    manifest = read_manifest(args.manifest)
    options = scheduler_options(args, manifest, directory)
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    state_path = directory / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["manifest_hash"] != digest:
            raise ValueError("manifest differs from the existing campaign; use its original manifest or a new state directory")
    else:
        state = {"version": 1, "campaign_id": uuid.uuid4().hex, "manifest_hash": digest,
                 "created_at": time.time(), "jobs": {
                     job["id"]: {"status": "pending", "attempts": []} for job in manifest["jobs"]}}
        atomic_json(directory / "manifest.json", manifest)
    if args.retry_failed:
        stop_path = directory / "stop.request.json"
        stop_path.unlink(missing_ok=True)
        for status in state["jobs"].values():
            if status["status"] != "completed":
                status["retry_base"] = len(status["attempts"])
            if status["status"] in {"failed", "blocked", "cancelled"}:
                status.update(status="pending", error=None)
    state["scheduler"] = {"pid": os.getpid(), "starttime": process_start(os.getpid())}
    state["status"] = "running"
    for key in ("finished_at", "exit_code", "scheduler_error"):
        state.pop(key, None)
    event(directory, "scheduler_started", **state["scheduler"])
    stop_path = directory / "stop.request.json"
    detached = []
    signal.signal(signal.SIGTERM, lambda signum, frame: detached.append(signum))
    signal.signal(signal.SIGINT, lambda signum, frame: detached.append(signum))
    children = []
    allowed = set(options["gpu_uuids"])
    atomic_json(state_path, state)
    try:
        while not detached:
            stopping = stop_path.exists() and json.loads(stop_path.read_text()).get("campaign_id") == state["campaign_id"]
            for job in manifest["jobs"]:
                status = state["jobs"][job["id"]]
                if status["status"] != "running":
                    continue
                attempt = status["attempts"][-1]
                result_path = Path(attempt["directory"]) / "result.json"
                result = json.loads(result_path.read_text()) if result_path.exists() else None
                alive = attempt_process(attempt)
                if stopping and alive:
                    terminate_attempt(attempt, signal.SIGKILL if time.time() - status.setdefault("stop_sent_at", time.time()) > 10 else signal.SIGTERM)
                if result is None and not alive and time.time() - attempt["started_at"] > 5:
                    result = {"exit_code": None, "finished_at": time.time(),
                              "error": "runner disappeared without an exit record", "stopped": stopping}
                if result is None:
                    continue
                if result.get("token", attempt["token"]) != attempt["token"]:
                    raise ValueError("attempt exit record token mismatch")
                missing = [path for path in job["success_artifacts"] if not Path(path).exists()]
                attempt.update(result=result, finished_at=time.time())
                if stopping or result.get("stopped"):
                    status["status"] = "cancelled"
                elif result["exit_code"] == 0 and not missing:
                    status["status"] = "completed"
                else:
                    status["error"] = result.get("error") or (f"missing success artifacts: {missing}" if missing else f"exit code {result['exit_code']}")
                    retries = int(job.get("max_retries", options["max_retries"]))
                    used = len(status["attempts"]) - status.get("retry_base", 0)
                    status["status"] = "pending" if used <= retries else "failed"
                event(directory, "job_finished", job=job["id"], status=status["status"], result=result,
                      error=status.get("error"))
            children = [child for child in children if child.poll() is None]
            try:
                state["gpus"] = gpu_snapshot() if allowed else []
                state.pop("gpu_error", None)
            except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
                state["gpus"] = []
                state["gpu_error"] = str(error)
            running = [job for job in state["jobs"].values() if job["status"] == "running"]
            reserved = {job["attempts"][-1]["gpu_uuid"] for job in running}
            free = [gpu for gpu in state["gpus"] if gpu["uuid"] not in reserved and not gpu["compute_pids"]
                    and gpu["utilization_percent"] <= options["max_utilization"] and gpu["uuid"] in allowed]
            for job in manifest["jobs"]:
                status = state["jobs"][job["id"]]
                if status["status"] != "pending":
                    continue
                dependencies = [state["jobs"][name]["status"] for name in job["depends_on"]]
                if stopping:
                    status["status"] = "cancelled"
                    continue
                if any(value in {"failed", "blocked", "cancelled"} for value in dependencies):
                    status.update(status="blocked", error="a dependency did not complete successfully")
                    event(directory, "job_blocked", job=job["id"], dependencies=job["depends_on"])
                    continue
                if not all(value == "completed" for value in dependencies):
                    status["waiting_for"] = job["depends_on"]
                    continue
                status.pop("waiting_for", None)
                if len(running) >= options["max_parallel"]:
                    continue
                gpu, lease = None, None
                if job.get("gpu", 1):
                    for candidate in list(free):
                        if job.get("gpu_uuid") and candidate["uuid"] != job["gpu_uuid"]:
                            continue
                        try:
                            lease, current = acquire_gpu(candidate, job, options)
                        except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
                            state["gpu_error"] = str(error)
                            continue
                        if lease is not None:
                            gpu = current
                            free.remove(candidate)
                            break
                    if gpu is None:
                        continue
                attempt = {"number": len(status["attempts"]) + 1, "token": uuid.uuid4().hex,
                           "gpu_uuid": gpu["uuid"] if gpu else None, "gpu_index": gpu["index"] if gpu else None,
                           "gpu_snapshot": gpu, "started_at": time.time()}
                attempt_dir = directory / "jobs" / job["id"] / f"attempt-{attempt['number']}"
                attempt["directory"] = str(attempt_dir)
                argv = list(job["argv"])
                checkpoint = job.get("resume_checkpoint")
                if attempt["number"] > 1 and checkpoint and Path(checkpoint).is_file():
                    if "resume_command" in job:
                        argv = [value.replace("{checkpoint}", checkpoint) for value in job["resume_command"]]
                    else:
                        resume_argv = job.get("resume_argv", ["model_path={path: " + json.dumps(checkpoint) + ", mode: resume}"])
                        argv.extend(value.replace("{checkpoint}", checkpoint) for value in resume_argv)
                    attempt["resumed_from"] = checkpoint
                environment = dict(manifest.get("env", {}), **job.get("env", {}))
                environment.setdefault("OMP_NUM_THREADS", "1")
                environment.setdefault("MKL_NUM_THREADS", "1")
                environment.update(CUDA_VISIBLE_DEVICES=gpu["uuid"] if gpu else "",
                                   PYTHONUNBUFFERED="1", ACVR_CAMPAIGN_TOKEN=attempt["token"])
                atomic_json(attempt_dir / "request.json", {"token": attempt["token"], "argv": argv,
                            "cwd": job["cwd"], "env": environment,
                            "gpu_lock_fd": lease.fileno() if lease is not None else None})
                status.update(status="running", error=None)
                status.pop("stop_sent_at", None)
                status["attempts"].append(attempt)
                # Persist intent before spawning; the runner writes its identity
                # separately, so a scheduler crash cannot orphan its ownership.
                atomic_json(state_path, state)
                try:
                    with (attempt_dir / "runner.log").open("a") as runner_output:
                        process = subprocess.Popen(
                            [sys.executable, str(Path(__file__).resolve()), "_job-runner", "--attempt-dir", str(attempt_dir),
                             "--token", attempt["token"]], start_new_session=True, stdin=subprocess.DEVNULL,
                            stdout=runner_output, stderr=subprocess.STDOUT,
                            pass_fds=() if lease is None else (lease.fileno(),),
                        )
                except OSError as error:
                    atomic_json(attempt_dir / "result.json", {"token": attempt["token"], "exit_code": 127,
                                "error": str(error), "started_at": attempt["started_at"], "finished_at": time.time()})
                    continue
                finally:
                    if lease is not None:
                        lease.close()
                children.append(process)
                attempt.update(pid=process.pid, starttime=process_start(process.pid))
                running.append(status)
                event(directory, "job_started", job=job["id"], attempt=attempt["number"], pid=process.pid,
                      gpu_uuid=attempt["gpu_uuid"], argv=argv)
            state["updated_at"] = time.time()
            statuses = [job["status"] for job in state["jobs"].values()]
            if all(status in TERMINAL for status in statuses):
                state["status"] = "completed" if all(status == "completed" for status in statuses) else "stopped" if stopping else "failed"
            atomic_json(state_path, state)
            event(directory, "heartbeat", status=state["status"], jobs={name: job["status"] for name, job in state["jobs"].items()})
            if state["status"] != "running":
                state["finished_at"] = time.time()
                state["exit_code"] = 0 if state["status"] == "completed" else 1
                atomic_json(state_path, state)
                return 0 if state["status"] == "completed" else 1
            deadline = time.monotonic() + options["poll_seconds"]
            while time.monotonic() < deadline and not detached and not (not stopping and stop_path.exists()):
                time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
        state["status"] = "detached"
        state["updated_at"] = time.time()
        atomic_json(state_path, state)
        event(directory, "scheduler_detached", signal=detached[0])
        return 0
    except Exception as error:
        state.update(status="scheduler_failed", scheduler_error=str(error), updated_at=time.time())
        atomic_json(state_path, state)
        event(directory, "scheduler_failed", error=str(error))
        raise
    finally:
        lock.close()


def launch_campaign(args):
    directory = Path(args.state_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # Serialize launch callers as well as running schedulers: the second caller
    # must not race the first scheduler's startup before it acquires its lock.
    with (directory / "launcher.lock").open("a") as launcher_lock:
        fcntl.flock(launcher_lock, fcntl.LOCK_EX)
        for name in ("state.json", "scheduler-launch.json"):
            path = directory / name
            if path.exists():
                stored = json.loads(path.read_text())
                identity = stored.get("scheduler", stored)
                if owned_scheduler(identity, directory):
                    print(json.dumps({"status": "already_running", "scheduler": identity,
                                      "state_dir": str(directory)}))
                    return 0
        with (directory / "scheduler.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(json.dumps({"status": "already_running", "state_dir": str(directory)}))
                return 0
        path = args.manifest or str(directory / "manifest.json")
        manifest = read_manifest(path)
        state_path = directory / "state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
            if state["manifest_hash"] != digest:
                raise ValueError("manifest differs from the existing campaign")
        scheduler_options(args, manifest, directory)
        atomic_json(directory / "manifest.json", manifest)
        argv = [sys.executable, str(Path(__file__).resolve()), "run", "--manifest",
                str(directory / "manifest.json"), "--state-dir", str(directory)]
        if args.command == "resume":
            argv.append("--retry-failed")
        with (directory / "scheduler.log").open("a") as output:
            process = subprocess.Popen(argv, start_new_session=True, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT)
        identity = {"pid": process.pid, "starttime": process_start(process.pid), "launched_at": time.time()}
        atomic_json(directory / "scheduler-launch.json", identity)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if state_path.exists():
                state = json.loads(state_path.read_text())
                if state.get("scheduler", {}).get("pid") == process.pid:
                    print(json.dumps({"status": "launched", "scheduler": identity,
                                      "campaign_status": state["status"], "state_dir": str(directory)}))
                    return 0
            if process.poll() is not None:
                raise ValueError(f"scheduler exited during startup ({process.returncode}); see {directory / 'scheduler.log'}")
            time.sleep(0.05)
        raise ValueError(f"scheduler startup not confirmed; inspect {directory / 'scheduler.log'} before retrying")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("launch", "launch a detached persistent scheduler"),
                            ("resume", "recover live jobs and retry failed jobs using saved settings"),
                            ("run", "run the scheduler in the foreground (launch detaches it)")):
        run = commands.add_parser(name, help=help_text)
        run.add_argument("--manifest", required=name != "resume")
        run.add_argument("--state-dir", required=True)
        run.add_argument("--max-parallel", type=int, default=1)
        run.add_argument("--max-retries", type=int, default=1)
        run.add_argument("--poll-seconds", type=float, default=10)
        run.add_argument("--min-free-mb", type=int, default=2048)
        run.add_argument("--max-utilization", type=int, default=20)
        run.add_argument("--gpus", help="comma-separated indices or UUIDs; resolved UUIDs persist across resume")
        run.add_argument("--gpu-lock-dir", default=os.environ.get(
            "ACVR_GPU_LOCK_DIR", f"/tmp/acvr-gpu-locks-{os.getuid()}"))
        run.add_argument("--retry-failed", action="store_true", help=argparse.SUPPRESS)
    for name in ("status", "stop"):
        command = commands.add_parser(name)
        command.add_argument("--state-dir", required=True)
    runner = commands.add_parser("_job-runner", help=argparse.SUPPRESS)
    runner.add_argument("--attempt-dir", required=True)
    runner.add_argument("--token", required=True)
    args = parser.parse_args()
    try:
        if args.command == "_job-runner":
            return run_job(args.attempt_dir, args.token)
        if args.command in {"run", "launch", "resume"}:
            if args.max_parallel < 1 or args.max_retries < 0 or args.poll_seconds <= 0:
                raise ValueError("max_parallel/poll_seconds must be positive and max_retries nonnegative")
            if args.min_free_mb < 0 or not 0 <= args.max_utilization <= 100:
                raise ValueError("min_free_mb must be nonnegative and max_utilization between 0 and 100")
            return run_campaign(args) if args.command == "run" else launch_campaign(args)
        directory = Path(args.state_dir).resolve()
        state = json.loads((directory / "state.json").read_text())
        if args.command == "status":
            state["scheduler_alive"] = owned_scheduler(state.get("scheduler", {}), directory)
            for job in state["jobs"].values():
                if job["attempts"]:
                    job["process_alive"] = attempt_process(job["attempts"][-1])
            try:
                options = json.loads((directory / "scheduler-options.json").read_text())
                state["gpus"] = gpu_snapshot() if options["gpu_uuids"] else []
            except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
                state["gpu_error"] = str(error)
            print(json.dumps(state, indent=2))
            return 0
        atomic_json(directory / "stop.request.json", {"campaign_id": state["campaign_id"], "time": time.time()})
        stopped = [name for name, job in state["jobs"].items() if job["status"] == "running"
                   and terminate_attempt(job["attempts"][-1])]
        event(directory, "stop_requested", jobs=stopped)
        print(json.dumps({"stop_requested": True, "signalled_jobs": stopped}))
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
