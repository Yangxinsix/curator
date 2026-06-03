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

    return mcp


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
