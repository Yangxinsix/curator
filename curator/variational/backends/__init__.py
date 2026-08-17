"""Reference and production backends for variational problems."""

from .dense import solve_dense, solve_schur
from .ewald import EwaldChargeOperator
from .lammps import (
    ElectrodeThermoRecord,
    parse_electrode_thermo,
    read_electrode_thermo,
)
from .iterative import (
    ProjectedCGResult,
    solve_projected_cg,
    solve_projected_cg_problem,
)

__all__ = [
    "ElectrodeThermoRecord",
    "EwaldChargeOperator",
    "ProjectedCGResult",
    "parse_electrode_thermo",
    "read_electrode_thermo",
    "solve_dense",
    "solve_projected_cg",
    "solve_projected_cg_problem",
    "solve_schur",
]
