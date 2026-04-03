from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ase import Atoms
from ase.io import read, write


def infer_specorder(atoms: Atoms, specorder: Optional[Sequence[str]] = None) -> list[str]:
    if specorder:
        return [str(symbol) for symbol in specorder]

    ordered: list[str] = []
    seen = set()
    for symbol in atoms.get_chemical_symbols():
        if symbol not in seen:
            ordered.append(symbol)
            seen.add(symbol)
    return ordered


def write_lammps_data(
    path: Path | str,
    atoms: Atoms,
    *,
    specorder: Sequence[str],
    masses: bool = True,
    atom_style: str = "atomic",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write(
        str(path),
        atoms,
        format="lammps-data",
        specorder=list(specorder),
        masses=bool(masses),
        atom_style=atom_style,
    )
    return path


def touch_empty_file(path: Path | str | None) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def load_input(
    *,
    input: Optional[str],
    input_text: Optional[str],
) -> str:
    if input_text:
        return str(input_text)
    if input:
        return Path(input).read_text()
    return resources.files("curator.simulate.engines.lammps").joinpath("templates/default.in").read_text()


def render_input(lammps_input: str, variables: Mapping[str, Any]) -> str:
    normalized = {str(key): _normalize_template_value(value) for key, value in variables.items()}
    return lammps_input.format_map(normalized)


def detect_pair_style(lammps_input: str) -> Optional[str]:
    for raw_line in lammps_input.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "pair_style":
            return fields[1]
    return None


def convert_lammps_dump_to_trajectory(
    dump_file: Path | str,
    output_path: Path | str,
    *,
    specorder: Optional[Sequence[str]] = None,
) -> int:
    dump_file = Path(dump_file)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not dump_file.exists() or dump_file.stat().st_size == 0:
        touch_empty_file(output_path)
        return 0

    read_kwargs = {"format": "lammps-dump-text"}
    if specorder:
        read_kwargs["specorder"] = list(specorder)

    images = read(str(dump_file), index=":", **read_kwargs)
    if isinstance(images, Atoms):
        images = [images]
    else:
        images = list(images)

    if not images:
        touch_empty_file(output_path)
        return 0

    write(str(output_path), images)
    return len(images)


def _normalize_template_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return str(value)
