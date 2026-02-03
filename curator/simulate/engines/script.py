from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.engine import BaseEngine
from ..core.context import SimContext


class ScriptEngine(BaseEngine):
    """
    Run an external Python script to perform simulation.
    The script is expected to write the output trajectory to out_traj.
    """

    def __init__(
        self,
        script: str,
        *,
        python: str = "python",
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        workdir: Optional[str] = None,
        model_path: Optional[str] = None,
        init_traj: Optional[str] = None,
        out_traj: Optional[str] = None,
        run_path: Optional[str] = None,
        raise_on_nonzero: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        self.script = str(script)
        self.python = python
        self.args = list(args) if args is not None else []
        self.env = dict(env) if env is not None else {}
        self.workdir = workdir
        self.model_path = model_path
        self.init_traj = init_traj
        self.out_traj = out_traj
        self.run_path = run_path
        self.raise_on_nonzero = raise_on_nonzero
        self.ctx: Optional[SimContext] = None
        self.atoms = None

    def setup(self, ctx: SimContext) -> None:
        self.ctx = ctx
        self.atoms = ctx.atoms

    def _attach_to_backend(self, fn, interval: int) -> None:
        # No per-step callbacks for external scripts.
        return None

    def run(self, **run_kwargs) -> None:
        if self.ctx is None:
            raise RuntimeError("Call setup(ctx) before run().")

        cmd = [self.python, self.script]
        if self.model_path:
            cmd += ["--model_path", str(self.model_path)]
        if self.init_traj:
            cmd += ["--init_traj", str(self.init_traj)]
        if self.out_traj:
            cmd += ["--out_traj", str(self.out_traj)]
        if self.run_path:
            cmd += ["--run_path", str(self.run_path)]
        cmd += self.args

        env = os.environ.copy()
        env.update(self.env)
        cwd = self.workdir or self.run_path or None

        result = subprocess.run(cmd, cwd=cwd, env=env)
        if result.returncode != 0 and self.raise_on_nonzero:
            raise RuntimeError(f"ScriptEngine failed with exit code {result.returncode}: {' '.join(cmd)}")
