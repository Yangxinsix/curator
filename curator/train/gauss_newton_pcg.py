"""Exact-residual LM with Nyström-preconditioned, verified inner CG solves.

The default original-objective acceptance requires true relative linear residual
<= 1e-3. An underconverged solve can refresh a stale sketch or raise damping, but
cannot update parameters or lower damping. The residual function/data remain
fixed throughout each call. All sketch construction and retry work is counted.

Defaults: rank=32, max_cg=50, cg_rtol=1e-3, cg_atol=0, max_retries=6,
damping_init=1, damping_min=1e-6, damping_max=1e12, acceptance_eta=1e-4,
refresh_interval=8, refresh_cg_fraction=.8, scale_probes=2, seed=0,
derivative_backend='autograd', deadline=None (absolute time.monotonic()).
Damping settings multiply the initial mean-GN-diagonal estimate, obtained with
the same independent Rademacher probes as the original LM implementation.
"""

import math
import time

import torch

try:
    from curator.train.gauss_newton import linearize_residual
except ModuleNotFoundError as error:
    if error.name not in ("curator", "curator.train", "curator.train.gauss_newton"):
        raise
    from lm import linearize_residual  # Standalone tests add the validated module path.


def _check_deadline(deadline, tensor=None):
    if deadline is not None:
        if tensor is not None and tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        if time.monotonic() >= deadline:
            raise TimeoutError


def build_nystrom(matvec, x, rank, seed=0, deadline=None):
    """Stable shifted Nyström approximation of an undamped PSD operator.

    This is Algorithm 2.1 of Frangella, Tropp & Udell (SIMAX 2023): QR random
    probes, shifted Cholesky, thin SVD, then subtract the numerical shift.
    https://tropp.caltech.edu/papers/FTU23-Randomized-Nystrom-SIMAX.pdf
    """
    _check_deadline(deadline)
    rank = min(int(rank), x.numel())
    generator = torch.Generator(device=x.device).manual_seed(int(seed))
    omega = torch.randn((x.numel(), rank), dtype=x.dtype, device=x.device,
                        generator=generator)
    omega = torch.linalg.qr(omega, mode="reduced")[0]
    _check_deadline(deadline, omega)
    columns = []
    for column in omega.unbind(1):
        _check_deadline(deadline)
        columns.append(matvec(column).detach())
        _check_deadline(deadline, columns[-1])
    sample = torch.stack(columns, dim=1)
    if not bool(torch.isfinite(sample).all()):
        raise FloatingPointError("Nonfinite Nyström operator sample")
    shift = max(torch.finfo(x.dtype).eps * float(sample.norm()), torch.finfo(x.dtype).tiny)
    # The small shift also protects a mathematically rank-deficient compression.
    for _ in range(8):
        _check_deadline(deadline)
        shifted = sample + shift * omega
        compressed = omega.T @ shifted
        compressed = (compressed + compressed.T) / 2
        lower, status = torch.linalg.cholesky_ex(compressed, check_errors=False)
        if int(status) == 0:
            break
        shift *= 10
    else:
        raise FloatingPointError("Nyström compression is not numerically PSD")
    # lower @ lower.T = compressed; factor = shifted @ lower**(-T).
    factor = torch.linalg.solve_triangular(lower, shifted.T, upper=False).T
    vectors, singular, _ = torch.linalg.svd(factor, full_matrices=False)
    eigenvalues = (singular.square() - shift).clamp_min(0)
    _check_deadline(deadline, eigenvalues)
    return {"vectors": vectors.detach(), "eigenvalues": eigenvalues.detach(),
            "rank": rank, "shift": shift}


def apply_nystrom(vector, sketch, damping):
    """Apply the SPD inverse preconditioner, never an approximate GN operator.

    FTU2023 Eq. 5.3: P^-1 = I + U[((s_min+mu)/(s+mu))-1]U.T.
    Zero sketch eigenvalues and the orthogonal complement are retained.
    """
    vectors, eigenvalues = sketch["vectors"], sketch["eigenvalues"]
    projection = vectors.T @ vector
    scaling = (eigenvalues[-1] + damping) / (eigenvalues + damping)
    return vector + vectors @ ((scaling - 1) * projection)


def pcg(matvec, rhs, precondition, max_cg=50, rtol=1e-3, atol=0., deadline=None):
    """PCG from zero, returning recursive diagnostics; caller verifies A @ step."""
    step = torch.zeros_like(rhs)
    residual = rhs.clone()
    rhs_norm = float(rhs.norm())
    target = max(atol, rtol * rhs_norm)
    info = {"cg_iterations": 0, "cg_stop": "max_cg",
            "cg_recursive_relative_residual": 1. if rhs_norm else 0.}
    if rhs_norm <= target:
        info["cg_stop"] = "recursive_converged"
        return step, info
    try:
        _check_deadline(deadline)
        transformed = precondition(residual)
        direction = transformed.clone()
        dot = torch.dot(residual, transformed)
        for _ in range(max_cg):
            _check_deadline(deadline)
            product = matvec(direction)
            curvature = torch.dot(direction, product)
            if not bool(torch.isfinite(curvature)) or float(curvature) <= 0 or float(dot) <= 0:
                info["cg_stop"] = "nonpositive_curvature"
                break
            alpha = dot / curvature
            step = step + alpha * direction
            residual = residual - alpha * product
            residual_norm = float(residual.norm())
            info["cg_iterations"] += 1
            info["cg_recursive_relative_residual"] = residual_norm / rhs_norm
            if not math.isfinite(residual_norm):
                info["cg_stop"] = "nonfinite_cg"
                info["cg_recursive_relative_residual"] = None
                break
            if residual_norm <= target:
                info["cg_stop"] = "recursive_converged"
                break
            transformed = precondition(residual)
            new_dot = torch.dot(residual, transformed)
            direction = transformed + (new_dot / dot) * direction
            dot = new_dot
    except TimeoutError:
        info["cg_stop"] = "deadline"
    return step.detach(), info


def lm_pcg_step(residual_fn, x, state=None, config=None):
    """Return detached parameters, reusable sketch/state, JSON-safe diagnostics.

    Input parameters/state are not mutated. State tensors contain only the
    detached low-rank sketch, never an autograd tape. A deadline-interrupted
    construction leaves any old complete sketch intact.
    Matvec counters count completed products; primitive JVP/VJP counters also
    include work issued before a deadline interrupts the enclosing product.
    """
    config, state = dict(config or {}), dict(state or {})
    x = x.detach()
    if x.ndim != 1 or not x.is_floating_point() or not x.numel():
        raise ValueError("x must be a nonempty one-dimensional floating tensor")
    rank = min(int(config.get("rank", 32)), x.numel())
    max_cg, retries = int(config.get("max_cg", 50)), int(config.get("max_retries", 6))
    rtol, atol = float(config.get("cg_rtol", 1e-3)), float(config.get("cg_atol", 0.))
    if min(rank, max_cg, retries) < 1 or not 0 < rtol < 1 or atol < 0:
        raise ValueError("rank/iteration limits must be positive; 0 < cg_rtol < 1; cg_atol >= 0")
    deadline = config.get("deadline")
    backend = config.get("derivative_backend", "autograd")
    info = dict.fromkeys(("cg_iterations", "jvp_calls", "vjp_calls", "residual_evals",
                          "reverse_sweeps", "normal_matvecs", "preconditioner_builds",
                          "preconditioner_refreshes", "preconditioner_matvecs",
                          "preconditioner_apply_calls", "verification_matvecs",
                          "linear_failures", "rejected_trials"), 0)
    info.update(accepted=False, status="not_started", loss_before=None, loss_after=None,
                gradient_norm=None, damping_before=state.get("damping"),
                damping_after=state.get("damping"), derivative_backend=backend,
                preconditioner_seconds=0., linear_solve_seconds=0., trials=[], builds=[])
    accepted_steps = int(state.get("accepted_steps", 0))

    try:
        _check_deadline(deadline)
        residual, raw_jvp, raw_vjp = linearize_residual(residual_fn, x, backend)
        info["residual_evals"] += 1
        if residual.ndim != 1 or not residual.numel():
            raise ValueError("residual_fn must return a nonempty vector")
        loss = float(.5 * torch.dot(residual, residual))
        if not math.isfinite(loss):
            info["status"] = "nonfinite_residual"
            return x, state, info
        info["loss_before"] = info["loss_after"] = loss

        def jvp(vector):
            _check_deadline(deadline)
            info["jvp_calls"] += 1
            if backend == "autograd":
                info["reverse_sweeps"] += 2
            else:
                info["residual_evals"] += 1
            result = raw_jvp(vector)
            _check_deadline(deadline, result)
            return result

        def vjp(vector):
            _check_deadline(deadline)
            info["vjp_calls"] += 1
            info["reverse_sweeps"] += 1
            result = raw_vjp(vector)
            _check_deadline(deadline, result)
            return result

        def normal(vector, phase="cg"):
            result = vjp(jvp(vector))
            info["normal_matvecs"] += 1
            if phase == "build":
                info["preconditioner_matvecs"] += 1
            elif phase == "verify":
                info["verification_matvecs"] += 1
            return result

        gradient = vjp(residual)
        gradient_norm = float(gradient.norm())
        info["gradient_norm"] = gradient_norm if math.isfinite(gradient_norm) else None
        if not math.isfinite(gradient_norm) or gradient_norm == 0:
            info["status"] = "zero_gradient" if gradient_norm == 0 else "nonfinite_gradient"
            return x, state, info
        if "scale" not in state:
            probes = int(config.get("scale_probes", 2))
            if probes < 1:
                raise ValueError("scale_probes must be positive")
            generator = torch.Generator(device=x.device).manual_seed(int(config.get("seed", 0)))
            scale = 0.
            for _ in range(probes):
                vector = torch.randint(2, x.shape, generator=generator, device=x.device)
                vector = (2 * vector.to(x.dtype) - 1) / math.sqrt(x.numel())
                product = jvp(vector)
                scale += float(torch.dot(product, product)) / probes
            if scale <= torch.finfo(x.dtype).tiny:
                product = jvp(gradient / gradient_norm)
                scale = float(torch.dot(product, product))
            if not math.isfinite(scale) or scale <= 0:
                info["status"] = "nonfinite_scale"
                return x, state, info
            state["scale"] = scale
        scale = float(state["scale"])
        lower = float(config.get("damping_min", 1e-6)) * scale
        upper = float(config.get("damping_max", 1e12)) * scale
        damping = float(state.get("damping", float(config.get("damping_init", 1.)) * scale))
        if not 0 < lower <= upper < math.inf or not math.isfinite(damping) or damping <= 0:
            raise ValueError("Damping and its bounds must be finite and positive")
        damping = min(upper, max(lower, damping))
        info.update(damping_before=damping, gn_scale=scale, damping_floor=lower)

        def rebuild():
            started = time.monotonic()
            previous_calls = info["preconditioner_matvecs"]
            build = {"rank": rank, "status": "building"}
            info["builds"].append(build)
            info["preconditioner_builds"] += 1
            info["preconditioner_refreshes"] += int("preconditioner" in state)
            build_id = int(state.get("builds_completed", 0)) + 1
            try:
                sketch = build_nystrom(lambda v: normal(v, "build"), x, rank,
                                       int(config.get("seed", 0)) + 104729 * build_id, deadline)
                _check_deadline(deadline, sketch["eigenvalues"])
                sketch.update(built_at_step=accepted_steps, build_id=build_id)
                state.update(preconditioner=sketch, builds_completed=build_id,
                             refresh_requested=False)
                build.update(status="complete", largest_eigenvalue=float(sketch["eigenvalues"][0]),
                             smallest_eigenvalue=float(sketch["eigenvalues"][-1]), shift=sketch["shift"])
            except TimeoutError:
                build["status"] = "deadline"
                raise
            finally:
                seconds = time.monotonic() - started
                info["preconditioner_seconds"] += seconds
                build.update(seconds=seconds, matvecs=info["preconditioner_matvecs"] - previous_calls)

        sketch = state.get("preconditioner")
        if (sketch is None or sketch["rank"] != rank or state.get("refresh_requested", False)
                or accepted_steps - sketch["built_at_step"] >= int(config.get("refresh_interval", 8))):
            rebuild()

        for _ in range(retries):
            _check_deadline(deadline)
            sketch = state["preconditioner"]

            def precondition(vector):
                info["preconditioner_apply_calls"] += 1
                return apply_nystrom(vector, sketch, damping)

            started = time.monotonic()
            step, trial = pcg(lambda v: normal(v) + damping * v, -gradient, precondition,
                              max_cg, rtol, atol, deadline)
            info["linear_solve_seconds"] += time.monotonic() - started
            info["cg_iterations"] += trial["cg_iterations"]
            trial.update(damping=damping, preconditioner_build_id=sketch["build_id"],
                         linear_converged=False, cg_status=trial["cg_stop"],
                         cg_relative_residual=None, rho=None)
            info["trials"].append(trial)
            _check_deadline(deadline)
            if trial["cg_stop"] == "deadline":
                raise TimeoutError
            normal_step = normal(step, "verify")
            true_norm = float((normal_step + damping * step + gradient).norm())
            relative = true_norm / gradient_norm
            converged = math.isfinite(true_norm) and true_norm <= max(atol, rtol * gradient_norm)
            trial.update(cg_relative_residual=relative if math.isfinite(relative) else None,
                         linear_converged=converged, cg_status="converged" if converged else "underconverged")
            if not converged:
                info["linear_failures"] += 1
                info["status"] = "linear_failure"
                if sketch["built_at_step"] < accepted_steps:
                    rebuild()
                    continue
                if damping >= upper:
                    info["status"] = "damping_limit"
                    break
                damping = min(upper, damping * max(2., min(10., relative / rtol))
                              if math.isfinite(relative) else upper)
                continue

            predicted = float(-torch.dot(gradient, step) - .5 * torch.dot(step, normal_step))
            trial["predicted_reduction"] = predicted if math.isfinite(predicted) else None
            _check_deadline(deadline)
            if math.isfinite(predicted) and predicted > 0:
                candidate = x + step
                candidate_residual = residual_fn(candidate).detach()
                info["residual_evals"] += 1
                candidate_loss = float(.5 * torch.dot(candidate_residual, candidate_residual))
                actual, rho = loss - candidate_loss, (loss - candidate_loss) / predicted
                trial.update(loss=candidate_loss if math.isfinite(candidate_loss) else None,
                             rho=rho if math.isfinite(rho) else None)
                _check_deadline(deadline)
                if (math.isfinite(candidate_loss) and math.isfinite(rho) and actual > 0
                        and rho > float(config.get("acceptance_eta", 1e-4))):
                    if rho > .75 and trial["cg_iterations"] <= max_cg / 2:
                        damping = max(lower, .5 * damping)
                    elif rho < .25:
                        damping = min(upper, 2 * damping)
                    state.update(accepted_steps=accepted_steps + 1,
                                 refresh_requested=trial["cg_iterations"] >=
                                 float(config.get("refresh_cg_fraction", .8)) * max_cg)
                    info.update(accepted=True, status="accepted", loss_after=candidate_loss,
                                actual_reduction=actual, predicted_reduction=predicted, rho=rho,
                                cg_relative_residual=relative, linear_converged=True)
                    x = candidate.detach()
                    break
            info["rejected_trials"] += 1
            info["status"] = "rejected"
            if damping >= upper:
                info["status"] = "damping_limit"
                break
            damping = min(upper, 2 * damping)
        state["damping"] = damping
        info["damping_after"] = damping
    except TimeoutError:
        info["status"] = "deadline"
        if "damping" in locals():
            state["damping"] = damping
            info["damping_after"] = damping
    return x, state, info
