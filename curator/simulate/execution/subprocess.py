from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from omegaconf import OmegaConf

from curator.utils import write_text


def execute_simulation_config(
    config: Mapping[str, Any],
    run_dir: Path,
    timeout_sec: int,
) -> Tuple[subprocess.CompletedProcess[str], Dict[str, Any]]:
    config_path = run_dir / "input_config.yaml"
    OmegaConf.save(OmegaConf.create(dict(config)), config_path, resolve=False)
    artifacts: Dict[str, Any] = {"run_dir": str(run_dir), "input_config": str(config_path)}

    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "from curator.commands.simulate import run_simulation_config_file; "
            "run_simulation_config_file(sys.argv[1])"
        ),
        str(config_path),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(Path.cwd()),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    artifacts["subprocess_walltime_sec"] = max(0.0, time.perf_counter() - started)
    artifacts["stdout"] = str(write_text(run_dir / "simulate_stdout.txt", completed.stdout))
    artifacts["stderr"] = str(write_text(run_dir / "simulate_stderr.txt", completed.stderr))
    artifacts["log"] = str(run_dir / "simulation.log")
    artifacts["trajectory"] = str(run_dir / "trajectory.traj")
    artifacts["warning_structures"] = str(run_dir / "warning_struct.traj")
    artifacts["summary"] = str(run_dir / "simulation_summary.json")
    artifacts["command"] = command
    return completed, artifacts
