"""Differentiable CPU/GPU NUFFT action for the symmetric periodic q-EFA kernel."""
import math
import warnings

import torch

from .reciprocal import reciprocal_grid


def _transform(points, data, shape, eps):
    """A shape requests a type-1 transform; None requests its type-2 adjoint."""
    data = data.resolve_conj().resolve_neg().contiguous()
    if points.device.type == "cpu":
        try:
            import finufft
        except ImportError as exc:
            raise ImportError("The CPU reciprocal NUFFT backend requires finufft.") from exc
        axes = points.detach().numpy()
        values = data.detach().numpy()
        options = dict(eps=eps, nthreads=torch.get_num_threads())
        if shape is None:
            result = finufft.nufft3d2(*axes, values, isign=1, **options)
        else:
            result = finufft.nufft3d1(*axes, values, shape, isign=-1, **options)
        return torch.from_numpy(result)
    if points.device.type != "cuda":
        raise ValueError("Reciprocal NUFFT supports CPU and CUDA tensors.")
    try:
        import cufinufft
    except (ImportError, OSError) as exc:
        raise ImportError("The CUDA reciprocal NUFFT backend requires a CUDA-compatible cufinufft.") from exc
    # cuFINUFFT consumes torch tensors directly via their device pointers.
    # Explicitly use the caller's device and stream, including for backward.
    with torch.cuda.device(points.device):
        options = dict(eps=eps, gpu_device_id=points.device.index,
                       gpu_stream=torch.cuda.current_stream(points.device).cuda_stream)
        axes = points.detach().unbind()
        if shape is None:
            return cufinufft.nufft3d2(*axes, data.detach(), isign=1, **options)
        return cufinufft.nufft3d1(*axes, data.detach(), shape, isign=-1, **options)


def _modes(shape, axis, tensor):
    """Broadcast centered mode indices for one component of phase differentiation."""
    n = shape[axis]
    broadcast = [1, 1, 1]
    broadcast[axis] = n
    modes = torch.arange(n, device=tensor.device, dtype=tensor.real.dtype) - n // 2
    return modes.view(broadcast)


# Adjoint transforms below call the autograd-aware operations recursively. This
# is essential for mixed coordinate/parameter derivatives when training forces.
class _Type1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, values, shape, eps):
        ctx.save_for_backward(points, values)
        ctx.shape, ctx.eps = shape, eps
        return _transform(points, values, shape, eps)

    @staticmethod
    def backward(ctx, grad):
        points, values = ctx.saved_tensors
        dpoints = dvalues = None
        if ctx.needs_input_grad[0]:
            derivatives = []
            for axis in range(3):
                ramp = _modes(ctx.shape, axis, grad)
                field = _Type2.apply(points, grad * ramp, ctx.eps)
                derivatives.append(((-1j * values) * field.conj()).real.sum(0))
            dpoints = torch.stack(derivatives)
        if ctx.needs_input_grad[1]:
            dvalues = _Type2.apply(points, grad, ctx.eps)
        return dpoints, dvalues, None, None


class _Type2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, spectrum, eps):
        ctx.save_for_backward(points, spectrum)
        ctx.eps = eps
        return _transform(points, spectrum, None, eps)

    @staticmethod
    def backward(ctx, grad):
        points, spectrum = ctx.saved_tensors
        shape = tuple(spectrum.shape[-3:])
        dpoints = dspectrum = None
        if ctx.needs_input_grad[0]:
            derivatives = []
            for axis in range(3):
                ramp = _modes(shape, axis, spectrum)
                field = _Type2.apply(points, spectrum * ramp, ctx.eps)
                derivatives.append((grad.conj() * (1j * field)).real.sum(0))
            dpoints = torch.stack(derivatives)
        if ctx.needs_input_grad[1]:
            dspectrum = _Type1.apply(points, grad, shape, ctx.eps)
        return dpoints, dspectrum, None


def nufft_aggregate(query, key, value, positions, cell, cutoff, reference_volume):
    """Apply the full-shell charge kernel to [N,V] values, using normalized [N,D] Q/K."""
    if positions.dtype not in (torch.float32, torch.float64):
        raise ValueError("Reciprocal NUFFT positions require float32 or float64.")
    if positions.is_cuda and torch.are_deterministic_algorithms_enabled():
        message = ("CUDA NUFFT spreading uses atomic accumulation. Select the explicit "
                   "reciprocal backend when deterministic algorithms are required.")
        if not torch.is_deterministic_algorithms_warn_only_enabled():
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    _, weights = reciprocal_grid(cell, cutoff, reference_volume)
    dtype = torch.complex128 if positions.dtype == torch.float64 else torch.complex64
    eps = 1e-11 if positions.dtype == torch.float64 else 2e-6
    frac = (positions - positions.mean(0)) @ torch.linalg.inv(cell)
    points = ((frac - (frac + 0.5).floor()) * (2 * math.pi)).T.contiguous()
    shape = tuple(weights.shape)

    # Bound the number of simultaneous complex fields on both particles and grid.
    channels = min(128, max(1, (1 << 20) // max(len(query), weights.numel())))
    key_step = min(channels, key.shape[1])
    value_step = max(1, channels // key_step)
    blocks = []
    for part in value.split(value_step, dim=1):
        output = torch.zeros_like(part)
        for q, k in zip(query.split(key_step, dim=1), key.split(key_step, dim=1)):
            sources = (k.T[:, None, :] * part.T[None, :, :]).flatten(0, 1).to(dtype)
            spectrum = _Type1.apply(points, sources, shape, eps)
            field = _Type2.apply(points, spectrum * weights, eps).real
            output = output + torch.einsum("nd,dvn->nv", q, field.view(q.shape[1], part.shape[1], -1))
        blocks.append(output)
    return torch.cat(blocks, dim=1)
