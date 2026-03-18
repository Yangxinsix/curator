from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Optional

from ase import Atoms
from ase.data import atomic_numbers
from omegaconf import DictConfig, ListConfig, OmegaConf

from ...core.context import SimContext
from ...core.engine import BaseEngine
from .io import (
    convert_lammps_dump_to_trajectory,
    detect_pair_style,
    infer_specorder,
    load_input,
    render_input,
    touch_empty_file,
    write_lammps_data,
)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    return dict(value)


def _as_command(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return [str(item) for item in value]
    return shlex.split(str(value))


class LammpsEngine(BaseEngine):
    """
    Thin bridge between CURATOR and LAMMPS.

    Responsibilities:
    - write a LAMMPS data file from one ASE Atoms object
    - inspect the LAMMPS input and resolve the right model artifact automatically
    - materialize an input script from user-provided template/text
    - run the LAMMPS executable
    - convert dump outputs into CURATOR trajectory artifacts
    """

    def __init__(
        self,
        *,
        model_path: Any,
        outputs: Any,
        command: Any = "lmp",
        input: Optional[str] = None,
        input_text: Optional[str] = None,
        input_variables: Any = None,
        specorder: Optional[list[str]] = None,
    ):
        super().__init__()
        self.log = logging.getLogger(__name__)
        self.model_path = model_path
        self.outputs_cfg = outputs
        self.command = _as_command(command)
        self.input = input
        self.input_text = input_text
        self.input_variables = _as_dict(input_variables)
        self.specorder = list(specorder) if specorder is not None else None

        self.ctx: Optional[SimContext] = None
        self.atoms: Optional[Atoms] = None
        self._resolved_outputs: dict[str, Optional[Path]] = {}
        self._resolved_specorder: list[str] = []
        self._model_file: Optional[Path] = None
        self.run_dir: Optional[Path] = None
        self.input_file: Optional[Path] = None
        self.data_file: Optional[Path] = None
        self.dump_file: Optional[Path] = None
        self.uncertain_dump_file: Optional[Path] = None
        self.stdout_file: Optional[Path] = None
        self.log_file: Optional[Path] = None
        self.deployed_model_path: Optional[Path] = None
        self.input_source: Optional[str] = None

    def _attach_to_backend(self, fn, interval: int) -> None:
        return

    def _resolve_path(self, value: str | Path | None) -> Optional[Path]:
        if value is None:
            return None
        path = Path(value)
        if not path.is_absolute():
            base = self.run_dir or Path.cwd()
            path = base / path
        return path.resolve()

    def _resolve_outputs(self) -> dict[str, Optional[Path]]:
        outputs = _as_dict(self.outputs_cfg)
        return {
            "pool_set": self._resolve_path(outputs.get("pool_set")),
            "uncertain_set": self._resolve_path(outputs.get("uncertain_set")),
            "restart_source": self._resolve_path(outputs.get("restart_source")),
            "raw_dir": self._resolve_path(outputs.get("raw_dir")),
        }

    def _resolve_model_file(self, lammps_input: str) -> Optional[Path]:
        if self.model_path is None:
            return None

        pair_style = detect_pair_style(lammps_input)
        paths = self.model_path if isinstance(self.model_path, (list, tuple, ListConfig)) else [self.model_path]
        resolved_paths = [self._resolve_path(path) for path in paths]
        single_path = resolved_paths[0] if len(resolved_paths) == 1 else None

        model_kind = "unknown"
        if single_path is not None:
            if zipfile.is_zipfile(single_path):
                try:
                    import torch

                    torch.jit.load(single_path, map_location="cpu")
                except Exception:
                    pass
                else:
                    model_kind = "torchscript"
            else:
                try:
                    import torch

                    loaded = torch.load(single_path, map_location="cpu", weights_only=False)
                except Exception:
                    loaded = None
                if (
                    loaded is not None
                    and loaded.__class__.__name__ == "LAMMPS_MLIAP"
                    and loaded.__class__.__module__.endswith("lammps_mliap_interface")
                ):
                    model_kind = "mliap"

        if pair_style == "curator":
            if model_kind == "mliap":
                raise ValueError(
                    "LAMMPS input requests pair_style curator, but model_path points to an exported MLIAP model."
                )
            if model_kind == "torchscript":
                return single_path
            if single_path is None:
                raise ValueError("pair_style curator currently requires a single model_path.")
            from curator.cli import deploy as deploy_model

            deploy_model(
                single_path,
                target_path=str(self.deployed_model_path),
            )
            return self.deployed_model_path

        if pair_style is None:
            self.log.warning("No pair_style found in LAMMPS input. Using model_path directly.")
            if single_path is None:
                raise ValueError("Direct LAMMPS model usage requires a single model_path.")
            return single_path
        if pair_style.startswith("mliap"):
            if model_kind == "mliap":
                return single_path
            if model_kind == "torchscript":
                raise ValueError(
                    "LAMMPS input requests pair_style mliap*, but model_path points to a deployed TorchScript/pair model. "
                    "Provide an original training checkpoint for auto-export, or pass an already-exported LAMMPS MLIAP model."
                )
            if not self._resolved_specorder:
                raise ValueError("specorder is required for LAMMPS MLIAP inputs.")
            from curator.cli import deploy as deploy_model

            deploy_model(
                self.model_path,
                target_path=str(self.deployed_model_path),
                lammps_mliap=True,
                element_types=self._resolved_specorder,
            )
            return self.deployed_model_path
        self.log.warning("Unsupported pair_style '%s' for model auto-resolution. Using model_path directly.", pair_style)
        if single_path is None:
            raise ValueError("Direct LAMMPS model usage requires a single model_path.")
        return single_path

    def _build_template_variables(self, steps: int) -> dict[str, Any]:
        variables = dict(self.input_variables)
        variables.update(
            {
                "steps": int(steps),
                "boundary": " ".join("p" if flag else "f" for flag in self.atoms.get_pbc()),
                "run_dir": self.run_dir,
                "input_file": self.input_file,
                "data_file": self.data_file,
                "dump_file": self.dump_file,
                "uncertain_dump_file": self.uncertain_dump_file,
                "stdout_file": self.stdout_file,
                "log_file": self.log_file,
                "model_file": self._model_file,
                "pool_set": self._resolved_outputs.get("pool_set"),
                "uncertain_set": self._resolved_outputs.get("uncertain_set"),
                "restart_source": self._resolved_outputs.get("restart_source"),
                "raw_dir": self._resolved_outputs.get("raw_dir"),
                "specorder": self._resolved_specorder,
                "elements": self._resolved_specorder,
                "atomic_numbers": [atomic_numbers[symbol] for symbol in self._resolved_specorder],
            }
        )
        return variables

    def _materialize_input(self, *, steps: int) -> None:
        rendered = render_input(self.input_source, self._build_template_variables(steps))
        self.input_file.parent.mkdir(parents=True, exist_ok=True)
        self.input_file.write_text(rendered)

    def _build_command(self) -> list[str]:
        return [
            *self.command,
            "-in",
            str(self.input_file),
            "-log",
            str(self.log_file),
        ]

    def setup(self, ctx: SimContext) -> None:
        self.ctx = ctx
        self.atoms = ctx.atoms
        if isinstance(self.atoms, list):
            raise NotImplementedError("LammpsEngine currently supports only a single ASE Atoms object.")
        if not isinstance(self.atoms, Atoms):
            raise TypeError("LammpsEngine.setup expects a single ASE Atoms object.")

        self._resolved_outputs = self._resolve_outputs()
        self.run_dir = self._resolved_outputs.get("raw_dir") or Path.cwd()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.input_file = (self.run_dir / "in.lammps").resolve()
        self.data_file = (self.run_dir / "system.data").resolve()
        self.dump_file = (self.run_dir / "dump.lammpstrj").resolve()
        self.uncertain_dump_file = (self.run_dir / "uncertain_dump.lammpstrj").resolve()
        self.stdout_file = (self.run_dir / "lammps.stdout").resolve()
        self.log_file = (self.run_dir / "lammps.log").resolve()
        self.deployed_model_path = (self.run_dir / "lammps_model.pt").resolve()

        self._resolved_specorder = infer_specorder(self.atoms, self.specorder)
        self.input_source = load_input(
            input=self.input,
            input_text=self.input_text,
        )
        self._model_file = self._resolve_model_file(self.input_source)

        write_lammps_data(
            self.data_file,
            self.atoms,
            specorder=self._resolved_specorder,
            masses=True,
            atom_style="atomic",
        )

        self._materialize_input(steps=0)
        ctx.state["lammps"] = {
            "command": self._build_command(),
            "input_file": str(self.input_file),
            "data_file": str(self.data_file),
            "dump_file": str(self.dump_file),
            "log_file": str(self.log_file),
            "stdout_file": str(self.stdout_file),
            "model_file": None if self._model_file is None else str(self._model_file),
            "specorder": list(self._resolved_specorder),
        }

    def run(self, steps: int, **_) -> None:
        if self.ctx is None:
            raise RuntimeError("Call setup(ctx) before run().")

        self._materialize_input(steps=steps)

        command = self._build_command()
        self.log.info("Running LAMMPS command: %s", " ".join(command))
        self.stdout_file.parent.mkdir(parents=True, exist_ok=True)
        with self.stdout_file.open("w") as stdout_handle:
            proc = subprocess.run(
                command,
                cwd=str(self.run_dir),
                stdout=stdout_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

        if proc.returncode != 0:
            raise RuntimeError(
                f"LAMMPS exited with code {proc.returncode}. See {self.stdout_file} and {self.log_file}."
            )

        pool_set = self._resolved_outputs.get("pool_set")
        if pool_set is not None:
            nframes = convert_lammps_dump_to_trajectory(
                self.dump_file,
                pool_set,
                specorder=self._resolved_specorder,
            )
            if nframes == 0:
                raise RuntimeError(f"LAMMPS did not produce trajectory frames in {self.dump_file}.")

        uncertain_set = self._resolved_outputs.get("uncertain_set")
        if uncertain_set is not None:
            if self.uncertain_dump_file.exists() and self.uncertain_dump_file.stat().st_size > 0:
                convert_lammps_dump_to_trajectory(
                    self.uncertain_dump_file,
                    uncertain_set,
                    specorder=self._resolved_specorder,
                )
            else:
                touch_empty_file(uncertain_set)

        restart_source = self._resolved_outputs.get("restart_source")
        if restart_source is not None and pool_set is not None and restart_source != pool_set:
            restart_source.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pool_set, restart_source)
