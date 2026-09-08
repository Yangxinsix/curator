# ACVR nonlinear solves, implicit differentiation, and force modes

## Nonlinear reference solver

`NonlinearEqualityProblem` is the float64 dense-autograd oracle for

$$
\min_x \phi(x),\qquad c(x)=0.
$$

`solve_newton_kkt` forms the exact Lagrangian Hessian

$$
W=\nabla^2_{xx}\left[\phi(x)+\lambda^Tc(x)\right]
$$

and solves the Newton KKT system. The line search decreases a scaled joint
stationarity/feasibility residual, not the objective alone. A converged KKT
root is reported separately from a stable minimum: `reliable=True` requires a
positive reduced Lagrangian Hessian. This implementation intentionally
materializes dense autograd Jacobians/Hessians; matrix-free Newton--MINRES and
hard preservation of a separate linear-constraint block remain production
work. The solver deliberately detaches its initial state and iteration history;
differentiate a converged state with the implicit adjoint interface rather than
backpropagating through this reference Newton loop.

## Implicit KKT backward

`solve_implicit(QuadraticProblem)` has the same forward state, multiplier and
energy semantics as `solve_dense`, but its backward solves one transposed KKT
system. It supports cotangents on both the state and multiplier and propagates
gradients through $H$, $g$, $A$, $b$, and therefore through compiled
controls such as $g=g_0-B^Th$.

For the minimized energy, the tested conjugate identities are

$$
\frac{\partial\Phi^*}{\partial H}=\frac12xx^T,\quad
\frac{\partial\Phi^*}{\partial g}=x,\quad
\frac{\partial\Phi^*}{\partial b}=-\lambda,\quad
\frac{\partial\Phi^*}{\partial h}=-Bx.
$$

This is the dense reference backward. A matrix-free adjoint using the same
production KKT operator is still required for large systems.

## Three force modes

Let

$$
\mathcal L(R,x,\lambda)=\phi(R,x)+\lambda^Tc(R,x),\qquad
r=\nabla_x\mathcal L.
$$

`differentiate_variational_energy` exposes three modes explicitly.

### `envelope`

$$
F_{\rm env}=-\left.\partial_R\mathcal L\right|_{x,\lambda}.
$$

State and multiplier are detached. This includes the multiplier/Pulay term
when $A$ or $b$ depends on geometry. It is the relaxed force at a certified
KKT state; at a nonstationary state it is only an approximation. The result
reports numerical `kkt_certified` separately, but does not call a finite-
tolerance envelope field strictly conservative.

### `full_differentiable`

$$
F_{\rm full}=-\frac{d}{dR}\phi(R,\hat x(R)).
$$

This follows the actual predictor/projection/corrector graph and is exactly
conservative for the reported composite energy. For a feasible state,

$$
F_{\rm full}-F_{\rm env}=-r^T\frac{\partial\hat x}{\partial R}.
$$

It need not approximate the relaxed force when the state has a large KKT
residual.

### `adjoint_corrected`

For a feasible approximate state, solve

$$
\mathcal K^T\begin{bmatrix}u\\\nu\end{bmatrix}
=\begin{bmatrix}r\\0\end{bmatrix}
$$

and differentiate the scalar surrogate

$$
S=\mathcal L-u^Tr-\nu^Tc
$$

with $x,\lambda,u,\nu$ fixed. The correction removes the first-order state
error in the relaxed force; tests observe linear envelope error and quadratic
adjoint error. It is not generally conservative for the current predictor
energy and is reported accordingly. The current adjoint solve is detached and
first-derivative only, so `adjoint_corrected` rejects `create_graph=True`
instead of returning an incomplete second derivative.

For legacy QEq, the full envelope force must include explicit derivatives of
electronegativity, hardness and the Ewald kernel. The existing legacy
`ewald_forces` output contains only the fixed-charge Ewald component; that
compatibility behavior must not be renamed as a complete envelope force.
