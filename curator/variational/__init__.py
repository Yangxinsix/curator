"""Amortized constrained variational response primitives.

The public convention is row-major: equality rows satisfy ``A @ x = b`` and
linear observables satisfy ``y = B @ x``.
"""

from .analytic import AnalyticQuadraticCase, make_consistent_quadratic_case
from .amortized import AmortizedResult, solve_amortized
from .amortized_training import (
    AmortizedTeacherBatch,
    AmortizedTrainingBatch,
    build_amortized_teacher_batch,
    estimate_kkt_multipliers,
    prepare_amortized_training_batch,
)
from .backends import (
    ElectrodeThermoRecord,
    EwaldChargeOperator,
    ProjectedCGResult,
    parse_electrode_thermo,
    read_electrode_thermo,
    solve_dense,
    solve_projected_cg,
    solve_projected_cg_problem,
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
from .functionals import FunctionalProblem, FunctionalStationarity, solve_functional_reference
from .functional_response import FunctionalLinearResponse, solve_functional_linear_response
from .forces import ForceMode, VariationalDerivative, differentiate_variational_energy
from .implicit import implicit_kkt_state, solve_implicit
from .kkt import assemble_kkt, evaluate_kkt
from .nonlinear import (
    NewtonKKTOptions,
    NewtonKKTResult,
    NonlinearEqualityProblem,
    nonlinear_from_quadratic,
    solve_newton_kkt,
)
from .observables import EnergyObservables, ObservableSpec, CHARGE_BOUNDARY_OBSERVABLES
from .stationary_observables import StationaryEnergyObservables
from .projection import project_to_equalities
from .operators import (
    BoundaryQuadraticOperator,
    DenseQuadraticOperator,
    OperatorQuadraticProblem,
    QuadraticOperator,
    VariationalOperator,
)
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
from .variables.charge_runtime import ACVRChargeRuntime, ChargeRuntimeResult

__all__ = [
    "CHARGE_BOUNDARY_OBSERVABLES",
    "StationaryEnergyObservables",
    "AnalyticQuadraticCase",
    "ACVRChargeRuntime",
    "AmortizedResult",
    "AmortizedTeacherBatch",
    "AmortizedTrainingBatch",
    "BoundaryRow",
    "BoundaryQuadraticOperator",
    "BoundarySpec",
    "CompiledBoundary",
    "ChargeRuntimeResult",
    "ConstraintSpec",
    "ControlSpec",
    "EnvironmentSpec",
    "ElectrodeThermoRecord",
    "EwaldChargeOperator",
    "EnergyObservables",
    "GaugeSpec",
    "ForceMode",
    "FunctionalProblem",
    "FunctionalLinearResponse",
    "FunctionalStationarity",
    "GroupSpec",
    "InconsistentConstraintsError",
    "KKTResiduals",
    "LinearResponse",
    "LinearTermSpec",
    "NewtonKKTOptions",
    "NewtonKKTResult",
    "NonlinearEqualityProblem",
    "DenseQuadraticOperator",
    "OperatorQuadraticProblem",
    "ObservableSpec",
    "ProjectionResult",
    "ProjectedCGResult",
    "QuadraticOperator",
    "QuadraticProblem",
    "RankDeficientConstraintsError",
    "RaggedCompiledBoundary",
    "SingularKKTError",
    "ThermodynamicGaugeError",
    "UnstableVariationalProblemError",
    "VariationalResult",
    "VariationalOperator",
    "VariationalDerivative",
    "VariationalValidationError",
    "VariableLayout",
    "VariableSpec",
    "assemble_kkt",
    "build_amortized_teacher_batch",
    "evaluate_kkt",
    "estimate_kkt_multipliers",
    "compile_boundary",
    "compile_boundary_batch",
    "differentiate_variational_energy",
    "make_consistent_quadratic_case",
    "implicit_kkt_state",
    "minimum_reduced_curvature",
    "nonlinear_from_quadratic",
    "project_to_equalities",
    "prepare_amortized_training_batch",
    "parse_electrode_thermo",
    "read_electrode_thermo",
    "solve_dense",
    "solve_functional_reference",
    "solve_functional_linear_response",
    "solve_implicit",
    "solve_amortized",
    "solve_projected_cg",
    "solve_projected_cg_problem",
    "solve_schur",
    "solve_linear_response",
    "solve_newton_kkt",
    "thermodynamic_gauge_basis",
    "validate_equalities",
    "validate_problem",
]
