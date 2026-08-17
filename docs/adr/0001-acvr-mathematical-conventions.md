# ADR 0001: ACVR mathematical and matrix conventions

Status: accepted, 2026-08-17

## Decision

The public Python API uses row-major linear maps. For a flattened internal
state (x\in\mathbb R^n),

\[
A\in\mathbb R^{m\times n},\qquad Ax=b,
\]

and an observable is

\[
B\in\mathbb R^{p\times n},\qquad y=Bx.
\]

The base quadratic functional and Lagrangian are

\[
F(x)=c+\frac12x^THx+g^Tx,
\qquad
\mathcal L=F+\lambda^T(Ax-b).
\]

Consequently, the equality-constrained KKT system is

\[
\begin{bmatrix}H&A^T\\A&0\end{bmatrix}
\begin{bmatrix}x\\\lambda\end{bmatrix}
=
\begin{bmatrix}-g\\b\end{bmatrix}.
\]

With this multiplier sign, the optimized potential obeys

\[
\frac{d\Phi^*}{db}=-\lambda.
\]

Thus the physical common potential of a fixed-charge QEq block is
(-\lambda), not (\lambda).

The older theory note uses column maps, (A_{\rm old}^Tx=a). Its exact code
mapping is (A=A_{\rm old}^T\) and (b=a). Production code, YAML expansion
reports, and new tests must use the row-major convention in this ADR.

## Boundary compilation

A fixed extensive observable is a hard equality (Bx=\bar y). A fixed
intensive control is a Legendre term

\[
\Phi=F-h^TBx,
\]

so it changes (g\) to (g-B^Th); it is not compiled as a hard equality.

A Robin or finite-reservoir environment is

\[
F_{\rm env}=\frac12(Bx-y_r)^TK(Bx-y_r)-h_0^TBx.
\]

It contributes

\[
H\leftarrow H+B^TKB,
\quad
g\leftarrow g-B^T(Ky_r+h_0),
\quad
c\leftarrow c+\frac12y_r^TKy_r.
\]

The constant term is retained because it affects total energies and boundary
derivatives even though it does not change the minimizing state.

## Projection convention

`metric=M` means that projection minimizes

\[
\frac12(z-x)^TM(z-x),\qquad Az=b,
\]

where (M) must be symmetric positive definite. The solution contains
(M^{-1}A^T(AM^{-1}A^T)^{-1}\). A variable named `metric` must never contain
the inverse metric without saying so explicitly.

## Response convention

For a parameter (\theta), the general linearized KKT equation is

\[
K
\begin{bmatrix}x_\theta\\\lambda_\theta\end{bmatrix}
=-
\begin{bmatrix}
H_\theta x+g_\theta+A_\theta^T\lambda\\
A_\theta x-b_\theta
\end{bmatrix}.
\]

`solve_linear_response` accepts the already assembled right-hand side. For a
fixed intensive control, (dx/dh) uses upper right-hand side (B^T); for an
equality target, (dx/db) uses lower right-hand side (I).

## Stability and rank

The Hessian need not be globally positive definite. It must be positive
definite on `null(A)`. Equality rows must be full row rank. Consistent redundant
rows and inconsistent targets are reported as different errors.
