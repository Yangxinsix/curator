# ACVR solver modes and certificates

The dense solver is the small-system oracle. It validates equality rank and
positive curvature on the feasible tangent space before solving the KKT
system.

The projected-CG backend is matrix-free: it only calls `operator.matvec(x)`.
Its `converged` flag certifies final stationarity, independently of the initial
residual, and it separately certifies equality feasibility. Those conditions
do not prove that a stationary point is a minimum. A negative-curvature mode
can be orthogonal to every Krylov vector and remain unseen. Therefore
`reliable=True` additionally requires the caller to provide
`reduced_stability_certified=True`, based on model construction or an
independent reduced-curvature check. Without that certificate the state may be
stationary, but it is deliberately reported as unreliable.

Always compile a boundary before a matrix-free solve:

```python
from curator.variational import solve_projected_cg_problem

operator_problem = compiled_boundary.to_operator_problem(
    charge_operator,
    electronegativity,
)
result = solve_projected_cg_problem(
    operator_problem,
    reduced_stability_certified=True,
)
```

`to_operator_problem()` applies all terms:

- fixed-intensive controls modify the linear term;
- Robin environments add a low-rank Hessian operator, linear term and
  constant;
- equality matrices and targets are carried unchanged.

Calling the raw projected-CG function with uncompiled electronegativity is only
correct when no control or environment modifies the functional.

The amortized quadratic state machine is:

```text
proposal -> hard projection -> KKT certificate -> correction -> exact fallback
```

It always reports the projected residual, correction count, fallback trigger
and final KKT residual. This implementation is a dense quadratic reference;
operator-based nonlinear correction and scalable preconditioning remain
production work.
