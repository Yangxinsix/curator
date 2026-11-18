"""Matrix-free damped Gauss--Newton for a pure, normalized residual function.

``lm_step(residual_fn, x, state, config)`` minimizes ``0.5 * residual_fn(x)^2``.
The callable must keep the data fixed throughout a step. Ordinary autograd is
the default backend, including an exact reverse-over-reverse JVP compatible
with native TorchScript/e3nn operations. Only the residual's first derivative
is used: a residual containing an energy gradient requires energy Hessian
products, not third derivatives.

Config keys (all optional): max_cg=30, cg_rtol=1e-3, cg_atol=0, max_retries=6,
acceptance_eta=1e-4, damping_init=1e-3, damping_min=1e-12, damping_max=1e12,
scale_probes=2, seed=0, deadline=None, derivative_backend='autograd'. The optional
'torch_func' backend uses forward-mode JVPs. Damping config values multiply an initial
deterministic estimate of trace(J.T @ J) / parameter_count. State stores absolute
damping and this scale. ``deadline`` uses time.monotonic(); an in-flight torch
operation cannot be interrupted, but the next operation will not be started.
"""

import math
import time

import torch


def linearize_residual(residual_fn, x, backend="autograd"):
    """Return detached residual and exact JVP/VJP closures at fixed parameters.

    The closures retain one primal reverse-mode tape until they are discarded.
    Products are detached; callers cannot differentiate through the optimizer.
    The autograd JVP differentiates J.T @ u with respect to the auxiliary u,
    never with respect to parameters, so it computes J @ v without adding the
    residual Hessian. It needs two reverse sweeps and no new primal evaluation.
    """
    if backend == "torch_func":
        x = x.detach()
        residual, pullback = torch.func.vjp(residual_fn, x)

        def jvp(vector):
            return torch.func.jvp(residual_fn, (x,), (vector.detach(),))[1].detach()

        def vjp(vector):
            return pullback(vector.detach(), create_graph=False)[0].detach()

    elif backend == "autograd":
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            residual = residual_fn(x)

        def vjp(vector):
            if not residual.requires_grad:
                return torch.zeros_like(x)
            product, = torch.autograd.grad(residual, x, grad_outputs=vector.detach(),
                                           retain_graph=True, create_graph=False,
                                           allow_unused=True)
            return torch.zeros_like(x) if product is None else product.detach()

        def jvp(vector):
            if not residual.requires_grad:
                return torch.zeros_like(residual)
            with torch.enable_grad():
                cotangent = torch.zeros_like(residual, requires_grad=True)
                transpose_product, = torch.autograd.grad(
                    residual, x, grad_outputs=cotangent, retain_graph=True,
                    create_graph=True, allow_unused=True)
                if transpose_product is None or not transpose_product.requires_grad:
                    return torch.zeros_like(residual)
                product, = torch.autograd.grad(
                    transpose_product, cotangent, grad_outputs=vector.detach(),
                    retain_graph=True, create_graph=False, allow_unused=True)
            return torch.zeros_like(residual) if product is None else product.detach()

    else:
        raise ValueError("derivative_backend must be 'autograd' or 'torch_func'")
    return residual.detach(), jvp, vjp


def lm_step(residual_fn, x, state=None, config=None):
    """Return detached new parameters, new state, and JSON-serializable metrics.

    No input tensors or dictionaries are mutated. Rejected/interrupted trials
    return the original parameters. All candidates are accepted against the
    actual original residual objective, never against a surrogate loss.
    """
    config = {} if config is None else config
    state = dict(state or {})
    x = x.detach()
    if x.ndim != 1 or not x.is_floating_point() or x.numel() == 0:
        raise ValueError("x must be a nonempty one-dimensional floating tensor")
    max_cg = int(config.get("max_cg", 30))
    max_retries = int(config.get("max_retries", 6))
    if max_cg < 1 or max_retries < 1:
        raise ValueError("max_cg and max_retries must be positive")
    deadline = config.get("deadline")
    backend = config.get("derivative_backend", "autograd")
    info = {
        "accepted": False, "status": "not_started", "loss_before": None,
        "loss_after": None, "gradient_norm": None, "cg_iterations": 0,
        "residual_evals": 0, "jvp_calls": 0, "vjp_calls": 0,
        "reverse_sweeps": 0, "derivative_backend": backend,
        "trials": [], "damping_before": state.get("damping"),
        "damping_after": state.get("damping"),
    }

    def expired():
        return deadline is not None and time.monotonic() >= deadline

    if expired():
        info["status"] = "deadline"
        return x, state, info

    residual, linear_jvp, linear_vjp = linearize_residual(residual_fn, x, backend)
    info["residual_evals"] += 1
    residual = residual.detach()
    if residual.ndim != 1 or residual.numel() == 0:
        raise ValueError("residual_fn must return a nonempty one-dimensional tensor")
    loss = float(0.5 * torch.dot(residual, residual))
    if not math.isfinite(loss):
        info["status"] = "nonfinite_residual"
        return x, state, info
    info["loss_before"] = info["loss_after"] = loss

    def jvp(vector):
        info["jvp_calls"] += 1
        if backend == "torch_func":
            info["residual_evals"] += 1
        else:
            info["reverse_sweeps"] += 2
        return linear_jvp(vector)

    def vjp(vector):
        info["vjp_calls"] += 1
        info["reverse_sweeps"] += 1
        # Detaching the cotangent is essential: this is J.T @ v, not a
        # derivative of a residual-dependent v (which would add Hessian terms).
        return linear_vjp(vector)

    if expired():
        info["status"] = "deadline"
        return x, state, info
    gradient = vjp(residual)
    grad_norm = float(torch.linalg.vector_norm(gradient))
    info["gradient_norm"] = grad_norm if math.isfinite(grad_norm) else None
    if not math.isfinite(grad_norm) or grad_norm == 0:
        info["status"] = "zero_gradient" if grad_norm == 0 else "nonfinite_gradient"
        return x, state, info

    if "scale" not in state:
        probes = int(config.get("scale_probes", 2))
        if probes < 1:
            raise ValueError("scale_probes must be positive")
        generator = torch.Generator(device=x.device).manual_seed(int(config.get("seed", 0)))
        scale = 0.0
        for _ in range(probes):
            if expired():
                info["status"] = "deadline"
                return x, state, info
            vector = torch.randint(2, x.shape, generator=generator, device=x.device)
            vector = (2 * vector.to(x.dtype) - 1) / math.sqrt(x.numel())
            product = jvp(vector)
            scale += float(torch.dot(product, product)) / probes
        if expired():
            info["status"] = "deadline"
            return x, state, info
        if scale <= torch.finfo(x.dtype).tiny and not expired():
            # A few random probes can all land in a null space. A nonzero
            # gradient cannot lie entirely in ker(J), so use it as a fallback.
            product = jvp(gradient / grad_norm)
            scale = float(torch.dot(product, product))
        if not math.isfinite(scale) or scale <= 0:
            info["status"] = "nonfinite_scale"
            return x, state, info
        state["scale"] = scale
    scale = state["scale"]
    damping_min = float(config.get("damping_min", 1e-12)) * scale
    damping_max = float(config.get("damping_max", 1e12)) * scale
    damping = float(state.get("damping", float(config.get("damping_init", 1e-3)) * scale))
    if not 0 < damping_min <= damping <= damping_max < math.inf:
        raise ValueError("damping must lie within finite positive damping bounds")
    nu = float(state.get("nu", 2.0))
    info["damping_before"] = damping
    info["gn_scale"] = scale
    target = max(float(config.get("cg_atol", 0.0)), float(config.get("cg_rtol", 1e-3)) * grad_norm)

    for _ in range(max_retries):
        if expired():
            info["status"] = "deadline"
            break
        step = torch.zeros_like(x)
        remainder = -gradient.clone()
        direction = remainder.clone()
        remainder_sq = torch.dot(remainder, remainder)
        cg_status = "max_cg"
        iterations = 0
        for _ in range(max_cg):
            if expired():
                cg_status = "deadline"
                break
            product = jvp(direction)
            if expired():
                cg_status = "deadline"
                break
            product = vjp(product) + damping * direction
            curvature = torch.dot(direction, product)
            if not bool(torch.isfinite(curvature)) or float(curvature) <= 0:
                cg_status = "nonpositive_curvature"
                break
            alpha = remainder_sq / curvature
            step = step + alpha * direction
            remainder = remainder - alpha * product
            new_sq = torch.dot(remainder, remainder)
            iterations += 1
            if not bool(torch.isfinite(new_sq)):
                cg_status = "nonfinite_cg"
                break
            if math.sqrt(float(new_sq)) <= target:
                remainder_sq = new_sq
                cg_status = "converged"
                break
            direction = remainder + (new_sq / remainder_sq) * direction
            remainder_sq = new_sq
        info["cg_iterations"] += iterations
        relative_residual = float(torch.linalg.vector_norm(remainder)) / grad_norm
        trial = {"damping": damping, "cg_iterations": iterations, "cg_status": cg_status,
                 "cg_relative_residual": relative_residual if math.isfinite(relative_residual) else None}
        info["trials"].append(trial)
        if expired() or cg_status == "deadline":
            info["status"] = "deadline"
            break
        if cg_status in ("nonpositive_curvature", "nonfinite_cg"):
            predicted = -math.inf
        else:
            product = jvp(step)
            # Use the undamped least-squares model, valid even for truncated CG.
            predicted = float(-torch.dot(gradient, step) - 0.5 * torch.dot(product, product))
        trial["predicted_reduction"] = predicted if math.isfinite(predicted) else None
        if expired():
            info["status"] = "deadline"
            break
        if math.isfinite(predicted) and predicted > 0:
            candidate = x + step
            candidate_residual = residual_fn(candidate).detach()
            info["residual_evals"] += 1
            candidate_loss = float(0.5 * torch.dot(candidate_residual, candidate_residual))
            actual = loss - candidate_loss
            rho = actual / predicted
            trial.update(loss=candidate_loss if math.isfinite(candidate_loss) else None,
                         rho=rho if math.isfinite(rho) else None)
            if expired():
                info["status"] = "deadline"
                break
            if math.isfinite(candidate_loss) and actual > 0 and rho > float(config.get("acceptance_eta", 1e-4)):
                damping *= 1 / 3 if rho >= 1 else max(1 / 3, 1 - (2 * rho - 1) ** 3)
                damping = min(damping_max, max(damping_min, damping))
                nu = 2.0
                info.update(accepted=True, status="accepted", loss_after=candidate_loss,
                            actual_reduction=actual, predicted_reduction=predicted, rho=rho)
                x = candidate.detach()
                break
        if damping >= damping_max:
            info["status"] = "damping_limit"
            break
        damping = min(damping_max, damping * nu)
        nu = min(1024.0, 2 * nu)
        info["status"] = "rejected"
    state.update(damping=damping, nu=nu)
    info["damping_after"] = damping
    return x, state, info
