#!/usr/bin/env python3
"""Persist a dependency graph of local, single-GPU jobs across scheduler restarts.

Run the scheduler in a detached session to survive terminal disconnects. After a
machine reboot, run the same command again; this script does not install a boot
service. Child runners persist their exit status independently of the scheduler.
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
            child = subprocess.Popen(specification["argv"], cwd=specification["cwd"], env=environment,
                                     stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
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


def run_campaign(args):
    directory = Path(args.state_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "scheduler.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError("another scheduler holds this campaign lock")
    manifest = read_manifest(args.manifest)
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
    state["scheduler"] = {"pid": os.getpid(), "starttime": process_start(os.getpid())}
    state["status"] = "running"
    event(directory, "scheduler_started", **state["scheduler"])
    stop_path = directory / "stop.request.json"
    detached = []
    signal.signal(signal.SIGTERM, lambda signum, frame: detached.append(signum))
    signal.signal(signal.SIGINT, lambda signum, frame: detached.append(signum))
    children = []
    allowed = set(args.gpus.split(",")) if args.gpus else None
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
                    result = {"exit_code": None, "error": "runner disappeared without an exit record", "stopped": stopping}
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
                    retries = int(job.get("max_retries", args.max_retries))
                    status["status"] = "pending" if len(status["attempts"]) <= retries else "failed"
                event(directory, "job_finished", job=job["id"], status=status["status"], result=result,
                      error=status.get("error"))
            children = [child for child in children if child.poll() is None]
            try:
                state["gpus"] = gpu_snapshot()
                state.pop("gpu_error", None)
            except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
                state["gpus"] = []
                state["gpu_error"] = str(error)
            running = [job for job in state["jobs"].values() if job["status"] == "running"]
            reserved = {job["attempts"][-1]["gpu_uuid"] for job in running}
            free = [gpu for gpu in state["gpus"] if gpu["uuid"] not in reserved and not gpu["compute_pids"]
                    and gpu["memory_total_mb"] - gpu["memory_used_mb"] >= args.min_free_mb
                    and gpu["utilization_percent"] <= args.max_utilization
                    and (allowed is None or str(gpu["index"]) in allowed or gpu["uuid"] in allowed)]
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
                if not free or len(running) >= args.max_parallel:
                    continue
                gpu = free.pop(0)
                attempt = {"number": len(status["attempts"]) + 1, "token": uuid.uuid4().hex,
                           "gpu_uuid": gpu["uuid"], "gpu_index": gpu["index"], "started_at": time.time()}
                attempt_dir = directory / "jobs" / job["id"] / f"attempt-{attempt['number']}"
                attempt["directory"] = str(attempt_dir)
                argv = list(job["argv"])
                checkpoint = job.get("resume_checkpoint")
                if attempt["number"] > 1 and checkpoint and Path(checkpoint).is_file():
                    resume_argv = job.get("resume_argv", ["model_path={path: " + json.dumps(checkpoint) + ", mode: resume}"])
                    argv.extend(value.replace("{checkpoint}", checkpoint) for value in resume_argv)
                    attempt["resumed_from"] = checkpoint
                environment = dict(manifest.get("env", {}), **job.get("env", {}))
                environment.update(CUDA_VISIBLE_DEVICES=gpu["uuid"], OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                   PYTHONUNBUFFERED="1", ACVR_CAMPAIGN_TOKEN=attempt["token"])
                atomic_json(attempt_dir / "request.json", {"token": attempt["token"], "argv": argv,
                            "cwd": job["cwd"], "env": environment})
                status.update(status="running", error=None)
                status["attempts"].append(attempt)
                # Persist intent before spawning; the runner writes its identity
                # separately, so a scheduler crash cannot orphan its ownership.
                atomic_json(state_path, state)
                process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "_job-runner", "--attempt-dir", str(attempt_dir),
                     "--token", attempt["token"]], start_new_session=True, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                children.append(process)
                attempt.update(pid=process.pid, starttime=process_start(process.pid))
                running.append(status)
                event(directory, "job_started", job=job["id"], attempt=attempt["number"], pid=process.pid,
                      gpu_uuid=gpu["uuid"], argv=argv)
            state["updated_at"] = time.time()
            statuses = [job["status"] for job in state["jobs"].values()]
            if all(status in TERMINAL for status in statuses):
                state["status"] = "completed" if all(status == "completed" for status in statuses) else "stopped" if stopping else "failed"
            atomic_json(state_path, state)
            event(directory, "heartbeat", status=state["status"], jobs={name: job["status"] for name, job in state["jobs"].items()})
            if state["status"] != "running":
                return 0 if state["status"] == "completed" else 1
            deadline = time.monotonic() + args.poll_seconds
            while time.monotonic() < deadline and not detached and not (not stopping and stop_path.exists()):
                time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
        state["status"] = "detached"
        state["updated_at"] = time.time()
        atomic_json(state_path, state)
        event(directory, "scheduler_detached", signal=detached[0])
        return 0
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run or recover a campaign from its persistent state")
    run.add_argument("--manifest", required=True)
    run.add_argument("--state-dir", required=True)
    run.add_argument("--max-parallel", type=int, default=1)
    run.add_argument("--max-retries", type=int, default=1)
    run.add_argument("--poll-seconds", type=float, default=10)
    run.add_argument("--min-free-mb", type=int, default=2048)
    run.add_argument("--max-utilization", type=int, default=20)
    run.add_argument("--gpus", help="optional comma-separated physical indices or GPU UUIDs")
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
        if args.command == "run":
            if args.max_parallel < 1 or args.max_retries < 0 or args.poll_seconds <= 0:
                raise ValueError("max_parallel/poll_seconds must be positive and max_retries nonnegative")
            return run_campaign(args)
        directory = Path(args.state_dir).resolve()
        state = json.loads((directory / "state.json").read_text())
        if args.command == "status":
            try:
                state["gpus"] = gpu_snapshot()
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
