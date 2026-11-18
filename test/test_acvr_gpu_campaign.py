"""Campaign persistence and GPU allocation using mock nvidia-smi and tiny jobs."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/acvr_gpu_campaign.py"
JOB = """
import json, os, sys, time
from pathlib import Path
identifier, artifact, delay = sys.argv[1:4]
with open('starts.jsonl', 'a') as handle:
    handle.write(json.dumps({'id': identifier, 'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'time': time.time()})+'\\n')
if '--fail' in sys.argv:
    sys.exit(3)
if '--retry' in sys.argv and '--resumed' not in sys.argv:
    Path('resume.ckpt').write_text('checkpoint')
    sys.exit(2)
time.sleep(float(delay))
Path(artifact).write_text(json.dumps({'id': identifier, 'gpu': os.environ['CUDA_VISIBLE_DEVICES'],
    'omp': os.environ['OMP_NUM_THREADS'], 'mkl': os.environ['MKL_NUM_THREADS'],
    'shared': os.environ.get('SHARED'), 'local': os.environ.get('LOCAL'), 'argv': sys.argv,
    'finished': time.time()}))
print('finished', identifier, flush=True)
"""


@pytest.fixture
def campaign(tmp_path):
    gpu_file = tmp_path / "gpus.json"
    gpu_file.write_text(json.dumps({"busy": [1]}))
    binary = tmp_path / "nvidia-smi"
    binary.write_text(f"#!{sys.executable}\n" + """
import json, os, sys
state = json.load(open(os.environ['MOCK_GPU_FILE']))
if '--query-compute-apps=gpu_uuid,pid' in sys.argv:
    for index in state['busy']:
        print(f'GPU-{index}, 99999')
else:
    for index in range(3):
        print(f'{index}, GPU-{index}, 24000, 100, 0')
""")
    binary.chmod(0o755)
    environment = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
                       MOCK_GPU_FILE=str(gpu_file), ACVR_GPU_LOCK_DIR=str(tmp_path / "gpu-locks"))
    return tmp_path, environment, gpu_file


def manifest(directory, jobs):
    path = directory / "manifest.json"
    path.write_text(json.dumps({"version": 1, "env": {"SHARED": "global"}, "jobs": jobs}))
    return path


def job(directory, identifier, dependencies=(), delay=0.1, extra=()):
    return {"id": identifier, "argv": [sys.executable, "-c", JOB, identifier, f"{identifier}.json", str(delay), *extra],
            "cwd": str(directory), "depends_on": list(dependencies), "success_artifacts": [f"{identifier}.json"]}


def start(directory, environment, path, *extra):
    return subprocess.Popen([sys.executable, str(SCRIPT), "run", "--manifest", str(path),
        "--state-dir", str(directory / "state"), "--poll-seconds", "0.1", "--max-parallel", "2", *extra],
        env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.03)
    raise AssertionError("condition did not become true")


def finish(process, expected=0):
    output, _ = process.communicate(timeout=20)
    assert process.returncode == expected, output


def test_dag_assigns_free_gpu_uuids_preserves_environment_and_artifacts(campaign):
    directory, environment, _ = campaign
    jobs = [job(directory, "a", delay=0.4), job(directory, "b", delay=0.4), job(directory, "c", ("a", "b"))]
    jobs[0]["env"] = {"LOCAL": "job", "OMP_NUM_THREADS": "3", "MKL_NUM_THREADS": "2"}
    process = start(directory, environment, manifest(directory, jobs))
    finish(process)
    state = json.loads((directory / "state/state.json").read_text())
    assert state["status"] == "completed"
    outputs = [json.loads((directory / f"{name}.json").read_text()) for name in ("a", "b", "c")]
    assert {output["gpu"] for output in outputs[:2]} == {"GPU-0", "GPU-2"}
    assert all(output["gpu"] != "GPU-1" for output in outputs)
    assert outputs[0]["omp"] == "3" and outputs[0]["mkl"] == "2"
    assert all(output["omp"] == output["mkl"] == "1" for output in outputs[1:])
    assert outputs[0]["shared"] == "global" and outputs[0]["local"] == "job"
    starts = [json.loads(line) for line in (directory / "starts.jsonl").read_text().splitlines()]
    assert starts[2]["time"] >= max(output["finished"] for output in outputs[:2])
    assert "finished a" in (directory / "state/jobs/a/attempt-1/stdout.log").read_text()
    assert all(json.loads(line) for line in (directory / "state/events.jsonl").read_text().splitlines())
    assert not list((directory / "state").glob("*.tmp"))
    # Re-running a finished campaign must not execute its jobs again.
    finish(start(directory, environment, directory / "manifest.json"))
    assert len((directory / "starts.jsonl").read_text().splitlines()) == 3


def test_restart_recovers_live_runner_and_concurrent_scheduler_lock(campaign):
    directory, environment, _ = campaign
    path = manifest(directory, [job(directory, "long", delay=2)])
    original = start(directory, environment, path)
    wait_for(lambda: (directory / "starts.jsonl").exists())
    duplicate = start(directory, environment, path)
    finish(duplicate, expected=2)
    original.kill()
    original.communicate(timeout=5)
    # Simulate a scheduler crash between Popen and recording its PID. The
    # surviving runner must also be found through its unique /proc token.
    state_path = directory / "state/state.json"
    state = json.loads(state_path.read_text())
    attempt = state["jobs"]["long"]["attempts"][0]
    attempt.pop("pid", None)
    attempt.pop("starttime", None)
    for filename in ("process.json", "child.json"):
        (Path(attempt["directory"]) / filename).unlink(missing_ok=True)
    state_path.write_text(json.dumps(state))
    recovered = start(directory, environment, path)
    finish(recovered)
    state = json.loads((directory / "state/state.json").read_text())
    assert len(state["jobs"]["long"]["attempts"]) == 1
    assert len((directory / "starts.jsonl").read_text().splitlines()) == 1


def test_retry_uses_checkpoint_and_failed_dependencies_are_blocked(campaign):
    directory, environment, _ = campaign
    retry = job(directory, "retry", extra=("--retry",))
    retry.update(resume_checkpoint="resume.ckpt", resume_argv=["--resumed", "{checkpoint}"])
    failed = job(directory, "failed", extra=("--fail",))
    failed["max_retries"] = 0
    jobs = [retry, failed, job(directory, "blocked", ("failed",))]
    finish(start(directory, environment, manifest(directory, jobs)), expected=1)
    state = json.loads((directory / "state/state.json").read_text())
    assert state["jobs"]["retry"]["status"] == "completed"
    assert len(state["jobs"]["retry"]["attempts"]) == 2
    assert state["jobs"]["failed"]["status"] == "failed"
    assert state["jobs"]["blocked"]["status"] == "blocked"
    assert not (directory / "blocked.json").exists()
    assert "--resumed" in json.loads((directory / "retry.json").read_text())["argv"]


def test_pending_waits_for_gpu_and_stop_leaves_unrelated_process_alive(campaign):
    directory, environment, gpu_file = campaign
    gpu_file.write_text(json.dumps({"busy": [0, 1, 2]}))
    process = start(directory, environment, manifest(directory, [job(directory, "long", delay=20)]))
    outsider = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"], start_new_session=True)
    try:
        wait_for(lambda: (directory / "state/state.json").exists())
        time.sleep(0.3)
        assert not (directory / "starts.jsonl").exists()
        gpu_file.write_text(json.dumps({"busy": [1]}))
        wait_for(lambda: (directory / "starts.jsonl").exists())
        stopped = subprocess.run([sys.executable, str(SCRIPT), "stop", "--state-dir", str(directory / "state")],
                                 env=environment, capture_output=True, text=True, timeout=5)
        assert stopped.returncode == 0, stopped.stderr
        finish(process, expected=1)
        assert outsider.poll() is None
        state = json.loads((directory / "state/state.json").read_text())
        assert state["status"] == "stopped"
        assert state["jobs"]["long"]["status"] == "cancelled"
    finally:
        outsider.terminate()
        outsider.wait(timeout=5)
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_atomic_state_and_pid_identity_do_not_accept_reused_or_foreign_pid(campaign):
    directory, _, _ = campaign
    specification = importlib.util.spec_from_file_location("acvr_gpu_campaign", SCRIPT)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.atomic_json(directory / "state.json", {"version": 1})
    module.atomic_json(directory / "state.json", {"version": 2})
    assert json.loads((directory / "state.json").read_text()) == {"version": 2}
    assert not module.owned_runner(os.getpid(), module.process_start(os.getpid()), "foreign-token")
    assert not module.owned_runner(os.getpid(), "wrong-starttime", "foreign-token")
    bad = manifest(directory, [job(directory, "a", ("b",)), job(directory, "b", ("a",))])
    with pytest.raises(ValueError, match="cycle"):
        module.read_manifest(bad)


def test_success_exit_requires_declared_artifacts(campaign):
    directory, environment, _ = campaign
    missing = job(directory, "missing")
    missing.update(argv=[sys.executable, "-c", "print('no artifact')"], max_retries=0)
    finish(start(directory, environment, manifest(directory, [missing])), expected=1)
    state = json.loads((directory / "state/state.json").read_text())
    assert state["jobs"]["missing"]["status"] == "failed"
    assert "missing success artifacts" in state["jobs"]["missing"]["error"]


def test_dead_wrapper_does_not_duplicate_a_surviving_owned_child(campaign):
    directory, environment, _ = campaign
    specification = job(directory, "orphan", delay=1.5)
    specification["max_retries"] = 0
    path = manifest(directory, [specification])
    original = start(directory, environment, path)
    wait_for(lambda: (directory / "starts.jsonl").exists())
    original.kill()
    original.communicate(timeout=5)
    process = json.loads((directory / "state/jobs/orphan/attempt-1/process.json").read_text())
    os.kill(process["pid"], signal.SIGKILL)
    recovered = start(directory, environment, path)
    time.sleep(0.3)
    assert recovered.poll() is None
    assert len((directory / "starts.jsonl").read_text().splitlines()) == 1
    finish(recovered, expected=1)
    state = json.loads((directory / "state/state.json").read_text())
    assert len(state["jobs"]["orphan"]["attempts"]) == 1
    assert state["jobs"]["orphan"]["status"] == "failed"
    assert (directory / "orphan.json").exists()


def detached(directory, environment, command="launch", path=None):
    argv = [sys.executable, str(SCRIPT), command, "--state-dir", str(directory / "state")]
    if path is not None:
        argv.extend(["--manifest", str(path), "--poll-seconds", "0.05", "--max-retries", "0"])
    result = subprocess.run(argv, env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def state_when(directory, predicate):
    path = directory / "state/state.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    return value if predicate(value) else None


def test_detached_cpu_scheduler_survives_launch_exit_and_recovers_without_duplication(campaign):
    directory, environment, _ = campaign
    jobs = [job(directory, "train", delay=1.8), job(directory, "eval", ("train",))]
    for specification in jobs:
        specification["gpu"] = 0
    path = manifest(directory, jobs)
    launched = detached(directory, environment, path=path)
    pid = launched["scheduler"]["pid"]
    assert os.getsid(pid) == pid
    wait_for(lambda: (directory / "starts.jsonl").exists())
    duplicate = detached(directory, environment, path=path)
    assert duplicate["status"] == "already_running"
    assert duplicate["scheduler"]["pid"] == pid
    attempt = json.loads((directory / "state/jobs/train/attempt-1/process.json").read_text())
    assert os.getsid(attempt["pid"]) == attempt["pid"]
    os.kill(pid, signal.SIGKILL)
    # The launcher has already exited and the scheduler is now dead. The job
    # remains alive in its own OS session and resume must adopt it.
    recovered = detached(directory, environment, command="resume")
    assert recovered["scheduler"]["pid"] != pid
    state = wait_for(lambda: state_when(directory, lambda item: item["status"] == "completed"))
    assert state["exit_code"] == 0 and state["finished_at"] >= state["created_at"]
    assert all(len(value["attempts"]) == 1 for value in state["jobs"].values())
    starts = [json.loads(line) for line in (directory / "starts.jsonl").read_text().splitlines()]
    assert [value["id"] for value in starts] == ["train", "eval"]
    assert all(value["gpu"] == "" for value in starts)
    result = json.loads((directory / "state/jobs/train/attempt-1/result.json").read_text())
    assert result["exit_code"] == 0 and result["finished_at"] >= result["started_at"]


def test_detached_resume_replaces_command_with_last_checkpoint_and_unblocks_eval(campaign):
    directory, environment, _ = campaign
    initial = "from pathlib import Path; Path('last.ckpt').write_text('saved'); raise SystemExit(7)"
    resumed = "import json,sys; from pathlib import Path; Path('train.json').write_text(json.dumps(sys.argv[1:]))"
    train = {"name": "train", "command": [sys.executable, "-c", initial], "gpu": 0,
             "cwd": str(directory), "resume_checkpoint": "last.ckpt",
             "resume_command": [sys.executable, "-c", resumed, "{checkpoint}"],
             "max_retries": 0, "success_artifacts": ["train.json"]}
    evaluation = job(directory, "eval", ("train",))
    evaluation["gpu"] = 0
    detached(directory, environment, path=manifest(directory, [train, evaluation]))
    state = wait_for(lambda: state_when(directory, lambda item: item["status"] == "failed"))
    assert state["jobs"]["train"]["attempts"][0]["result"]["exit_code"] == 7
    assert state["jobs"]["eval"]["status"] == "blocked"
    wait_for(lambda: not process_alive(state["scheduler"]["pid"]))
    detached(directory, environment, command="resume")
    state = wait_for(lambda: state_when(directory, lambda item: item["status"] == "completed"))
    assert len(state["jobs"]["train"]["attempts"]) == 2
    assert len(state["jobs"]["eval"]["attempts"]) == 1
    assert json.loads((directory / "train.json").read_text()) == [str(directory / "last.ckpt")]
    assert state["jobs"]["train"]["attempts"][1]["resumed_from"] == str(directory / "last.ckpt")
    wait_for(lambda: not process_alive(state["scheduler"]["pid"]))
    detached(directory, environment, command="resume")
    wait_for(lambda: state_when(directory, lambda item: item["status"] == "completed"))
    assert len((directory / "starts.jsonl").read_text().splitlines()) == 1


def process_alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def test_gpu_lease_survives_scheduler_and_wrapper_death(campaign):
    directory, environment, _ = campaign
    first = job(directory, "first", delay=1.2)
    first["gpu_uuid"] = "GPU-0"
    original = start(directory, environment, manifest(directory, [first]), "--gpus", "0")
    wait_for(lambda: (directory / "starts.jsonl").exists())
    original.kill()
    original.communicate(timeout=5)
    wrapper = json.loads((directory / "state/jobs/first/attempt-1/process.json").read_text())
    os.kill(wrapper["pid"], signal.SIGKILL)
    second_dir = directory / "second"
    second_dir.mkdir()
    second = job(second_dir, "second")
    second["gpu_uuid"] = "GPU-0"
    process = start(second_dir, environment, manifest(second_dir, [second]), "--gpus", "0")
    finish(process)
    first_finished = json.loads((directory / "first.json").read_text())["finished"]
    second_started = json.loads((second_dir / "starts.jsonl").read_text())["time"]
    assert second_started >= first_finished
    options = json.loads((second_dir / "state/scheduler-options.json").read_text())
    assert options["gpu_uuids"] == ["GPU-0"]


def test_acquire_gpu_rechecks_resources_after_taking_lock(campaign, monkeypatch):
    directory, _, _ = campaign
    specification = importlib.util.spec_from_file_location("acvr_gpu_campaign_recheck", SCRIPT)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    gpu = {"uuid": "GPU-0", "index": 0, "memory_total_mb": 24000,
           "memory_used_mb": 0, "utilization_percent": 0, "compute_pids": []}
    options = {"gpu_lock_dir": str(directory / "locks"), "min_free_mb": 1024, "max_utilization": 20}
    monkeypatch.setattr(module, "gpu_snapshot", lambda: [dict(gpu, compute_pids=[73528])])
    lease, current = module.acquire_gpu(gpu, {}, options)
    assert lease is None and current["compute_pids"] == [73528]
    monkeypatch.setattr(module, "gpu_snapshot", lambda: [gpu])
    lease, _ = module.acquire_gpu(gpu, {}, options)
    assert lease is not None
    lease.close()
