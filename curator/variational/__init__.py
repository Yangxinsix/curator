"""Amortized constrained variational response primitives.

The public convention is row-major: equality rows satisfy ``A @ x = b`` and
linear observables satisfy ``y = B @ x``.
"""

from .analytic import AnalyticQuadraticCase, make_consistent_quadratic_case
from .backends import (
    ElectrodeThermoRecord,
    EwaldChargeOperator,
    parse_electrode_thermo,
    read_electrode_thermo,
    solve_dense,
    solve_schur,
)
from .compiler import (
    BoundaryRow,
    CompiledBoundary,
    RaggedCompiledBoundary,
    compile_boundary,
    compile_boundary_batch,
)
from .functional import QuadraticProblem
from .kkt import assemble_kkt, evaluate_kkt
from .projection import project_to_equalities
from .operators import DenseQuadraticOperator, QuadraticOperator, VariationalOperator
from .response import solve_linear_response
from .result import KKTResiduals, LinearResponse, ProjectionResult, VariationalResult
from .validation import (
    InconsistentConstraintsError,
    RankDeficientConstraintsError,
    SingularKKTError,
    ThermodynamicGaugeError,
    UnstableVariationalProblemError,
    VariationalValidationError,
    minimum_reduced_curvature,
    thermodynamic_gauge_basis,
    validate_equalities,
    validate_problem,
)
from .spec import (
    BoundarySpec,
    ConstraintSpec,
    ControlSpec,
    EnvironmentSpec,
    GaugeSpec,
    GroupSpec,
    LinearTermSpec,
    VariableLayout,
    VariableSpec,
)

__all__ = [
    "AnalyticQuadraticCase",
    "BoundaryRow",
    "BoundarySpec",
    "CompiledBoundary",
    "ConstraintSpec",
    "ControlSpec",
    "EnvironmentSpec",
    "ElectrodeThermoRecord",
    "EwaldChargeOperator",
    "GaugeSpec",
    "GroupSpec",
    "InconsistentConstraintsError",
    "KKTResiduals",
    "LinearResponse",
    "LinearTermSpec",
    "DenseQuadraticOperator",
    "ProjectionResult",
    "QuadraticOperator",
    "QuadraticProblem",
    "RankDeficientConstraintsError",
    "RaggedCompiledBoundary",
    "SingularKKTError",
    "ThermodynamicGaugeError",
    "UnstableVariationalProblemError",
    "VariationalResult",
    "VariationalOperator",
    "VariationalValidationError",
    "VariableLayout",
    "VariableSpec",
    "assemble_kkt",
    "evaluate_kkt",
    "compile_boundary",
    "compile_boundary_batch",
    "make_consistent_quadratic_case",
    "minimum_reduced_curvature",
    "project_to_equalities",
    "parse_electrode_thermo",
    "read_electrode_thermo",
    "solve_dense",
    "solve_schur",
    "solve_linear_response",
    "thermodynamic_gauge_basis",
    "validate_equalities",
    "validate_problem",
]
