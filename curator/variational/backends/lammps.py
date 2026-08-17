"""Read-only contracts for LAMMPS ELECTRODE reference calculations.

This module does not launch LAMMPS and does not duplicate its electrostatic
solver.  It turns official ``thermo_style custom`` output into typed oracle
records that can be compared with future ACVR production backends.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class ElectrodeThermoRecord:
    boundary: str
    step: int
    temperature: float
    pair_energy: float
    total_energy: float
    bottom_charge: float
    top_charge: float
    bottom_potential: Optional[float] = None
    top_potential: Optional[float] = None

    @property
    def potential_difference(self) -> Optional[float]:
        if self.bottom_potential is None or self.top_potential is None:
            return None
        return self.top_potential - self.bottom_potential


_REQUIRED_COLUMNS = (
    "Step",
    "Temp",
    "E_pair",
    "TotEng",
    "c_qbot",
    "c_qtop",
)


def _record(boundary: str, columns: Iterable[str], values: Iterable[str]):
    row: Dict[str, float] = dict(zip(columns, map(float, values)))
    if not all(column in row for column in _REQUIRED_COLUMNS):
        return None
    bottom_potential = row.get("f_conq[1]")
    top_potential = row.get("f_conq[2]")
    return ElectrodeThermoRecord(
        boundary=boundary,
        step=int(row["Step"]),
        temperature=row["Temp"],
        pair_energy=row["E_pair"],
        total_energy=row["TotEng"],
        bottom_charge=row["c_qbot"],
        top_charge=row["c_qtop"],
        bottom_potential=bottom_potential,
        top_potential=top_potential,
    )


def parse_electrode_thermo(text: str, boundary: str) -> Tuple[ElectrodeThermoRecord, ...]:
    """Parse all ELECTRODE thermo rows from a LAMMPS screen or log string."""

    if boundary not in ("constant_potential", "constant_charge"):
        raise ValueError("boundary must be constant_potential or constant_charge")
    header = None
    records = []
    for line in text.splitlines():
        tokens = line.split()
        if tokens and tokens[0] == "Step" and all(
            column in tokens for column in _REQUIRED_COLUMNS
        ):
            header = tuple(tokens)
            continue
        if header is None or len(tokens) != len(header):
            continue
        try:
            record = _record(boundary, header, tokens)
        except ValueError:
            continue
        if record is not None:
            records.append(record)
    if not records:
        raise ValueError("no compatible LAMMPS ELECTRODE thermo rows were found")
    return tuple(records)


def read_electrode_thermo(
    path: str, boundary: str
) -> Tuple[ElectrodeThermoRecord, ...]:
    """Read and parse an ELECTRODE log without retaining LAMMPS state."""

    return parse_electrode_thermo(
        Path(path).expanduser().read_text(encoding="utf-8"),
        boundary=boundary,
    )
