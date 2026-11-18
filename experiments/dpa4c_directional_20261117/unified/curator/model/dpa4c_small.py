"""Differentiable CUDA contractions for the C32/L3/R4 DPA4C profile."""

from collections import defaultdict
from functools import cache
from itertools import product
from math import sqrt
from pathlib import Path

import numpy as np
import torch


@cache
def _extension():
    from torch.utils.cpp_extension import load

    return load(
        name="curator_dpa4c_small_cuda_v1",
        sources=[str(Path(__file__).parents[1] / "native/dpa4c_small.cu")],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
    )


class PolynomialMap:
    """Immutable sparse polynomial coefficients with reusable transposed derivatives."""

    def __init__(self, inputs, rows):
        self.inputs = inputs
        self.rows = []
        for row in rows:
            merged = defaultdict(float)
            for coefficient, factors in row:
                merged[tuple(sorted(factors))] += coefficient
            self.rows.append([(v, k) for k, v in merged.items() if v != 0])

    @cache
    def packed(self, device, dtype):
        offsets, indices, coefficients = [0], [], []
        for row in self.rows:
            for value, factors in row:
                assert len(factors) <= 5
                indices.append(factors + (-1,) * (5 - len(factors)))
                coefficients.append(value)
            offsets.append(len(indices))
        return (
            torch.tensor(offsets, dtype=torch.int32, device=device),
            torch.tensor(indices, dtype=torch.int32, device=device).reshape(-1, 5),
            torch.tensor(coefficients, dtype=dtype, device=device),
        )

    @cache
    def vjp(self):
        # The new columns hold the incoming adjoint. Differentiating the
        # resulting polynomial again retains both geometry and parameter paths.
        rows = [[] for _ in range(self.inputs)]
        for output, row in enumerate(self.rows):
            for value, factors in row:
                for slot, variable in enumerate(factors):
                    rows[variable].append(
                        (
                            value,
                            factors[:slot]
                            + factors[slot + 1 :]
                            + (self.inputs + output,),
                        )
                    )
        return PolynomialMap(self.inputs + len(self.rows), rows)


class _Polynomial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, plan):
        ctx.plan = plan
        ctx.save_for_backward(x)
        return _extension().evaluate(x.contiguous(), *plan.packed(x.device, x.dtype))

    @staticmethod
    def backward(ctx, adjoint):
        (x,) = ctx.saved_tensors
        return _Polynomial.apply(torch.cat((x, adjoint), dim=1), ctx.plan.vjp()), None


@cache
def _messages():
    # Columns: unit direction x/y/z, cutoff, then 32 enveloped radial amplitudes.
    x, y, z, f = range(4)
    a, b, c = sqrt(3), sqrt(5 / 8), sqrt(3 / 8)
    basis = [
        [(1, ())],
        [(1, (x,))],
        [(1, (y,))],
        [(1, (z,))],
        [(a, (x, y))],
        [(a, (y, z))],
        [(1, (z, z)), (-0.5, (x, x)), (-0.5, (y, y))],
        [(a, (x, z))],
        [(0.5 * a, (x, x)), (-0.5 * a, (y, y))],
        [(3 * b, (x, x, y)), (-b, (y, y, y))],
        [(sqrt(15), (x, y, z))],
        [(4 * c, (y, z, z)), (-c, (x, x, y)), (-c, (y, y, y))],
        [(1, (z, z, z)), (-1.5, (x, x, z)), (-1.5, (y, y, z))],
        [(4 * c, (x, z, z)), (-c, (x, x, x)), (-c, (x, y, y))],
        [(0.5 * sqrt(15), (x, x, z)), (-0.5 * sqrt(15), (y, y, z))],
        [(b, (x, x, x)), (-3 * b, (x, y, y))],
    ]
    rows = [[(1, (f, f))], [(1, (f, f, f, f))]]
    for degree, width in enumerate((32, 8, 4, 1)):
        for angular in basis[degree**2 : (degree + 1) ** 2]:
            for channel in range(width):
                scale = (4 + channel,) + ((f,) if degree else ())
                rows.append([(v, k + scale) for v, k in angular])
    return PolynomialMap(36, rows)


@cache
def _invariant_plan():
    from curator.model.dpa4c import _gaunt_coupling, _packed_l2_to_stf, _probe_entries

    rows = []
    # Angular blocks are stored as [magnetic index, channel] in each row.
    sizes = (8, 4, 1)
    start = 0
    for degree, width in enumerate(sizes, 1):
        for i in range(width):
            for j in range(i, width):
                scale = 1 if i == j else sqrt(2)
                rows.append(
                    [
                        (scale, (start + m * width + i, start + m * width + j))
                        for m in range(2 * degree + 1)
                    ]
                )
        start += width * (2 * degree + 1)

    ranks = [4, 2, 1]
    vector = np.arange(start, start + 12).reshape(3, 4)
    tensor = np.arange(start + 12, start + 22).reshape(5, 2)
    octupole = np.arange(start + 22, start + 29).reshape(7, 1)
    slots = (vector, tensor, octupole)
    stf = _packed_l2_to_stf(torch.eye(5, dtype=torch.float64)).numpy()
    c112 = -stf.transpose(1, 2, 0) / sqrt(5)
    c222 = -sqrt(12 / 35) * np.einsum("aij,bjk,cki->abc", stf, stf, stf)
    for degrees in ((1, 1, 2), (1, 2, 3), (2, 2, 2), (2, 3, 3)):
        if degrees == (1, 1, 2):
            coupling = c112
        elif degrees == (2, 2, 2):
            coupling = c222
        else:
            coupling = _gaunt_coupling(degrees, 3).numpy()
        nz = np.argwhere(abs(coupling) > 1e-13)
        probes, _ = _probe_entries(degrees, ranks)
        for probe in probes.tolist():
            if degrees[0] == degrees[2]:
                scale = {1: 1.0, 2: sqrt(3), 3: sqrt(6)}[len(set(probe))]
            elif degrees[0] == degrees[1]:
                scale = 1.0 if probe[0] == probe[1] else sqrt(2)
            elif degrees[1] == degrees[2]:
                scale = 1.0 if probe[1] == probe[2] else sqrt(2)
            else:
                scale = 1.0
            rows.append(
                [
                    (
                        scale * coupling[tuple(m)],
                        tuple(
                            int(slots[d - 1][j, i])
                            for d, j, i in zip(degrees, m, probe)
                        ),
                    )
                    for m in nz
                ]
            )
    # |Qv|^2 in the same tensor-major, vector-minor order as Curator.
    qtq = np.einsum("aij,bik->ajbk", stf, stf)
    nz = np.argwhere(abs(qtq) > 1e-13)
    for t, v in product(range(2), range(4)):
        rows.append(
            [
                (
                    qtq[a, j, b, k],
                    (
                        int(tensor[a, t]),
                        int(vector[j, v]),
                        int(tensor[b, t]),
                        int(vector[k, v]),
                    ),
                )
                for a, j, b, k in nz
            ]
        )
    return PolynomialMap(start + 29, rows)


def descriptor(model, edge_idx, edge_diff, atom_types):
    """Exact radial network and differentiable angular messages and invariants."""
    distance = (edge_diff.square().sum(-1) + 1e-14).sqrt()
    cutoff = model._cutoff(distance)[:, None]
    radial, modes = model._radial_profiles(distance)
    scale, shift, mixing = model._pair_coefficients()
    pair = atom_types[edge_idx[:, 0]] * model.num_elements + atom_types[edge_idx[:, 1]]
    amplitude = (
        scale[pair] * radial + shift[pair] + (mixing[pair] * modes[:, None]).sum(-1)
    )
    inputs = torch.cat((edge_diff / distance[:, None], cutoff, amplitude * cutoff), -1)
    messages = _Polynomial.apply(inputs, _messages())
    reduced = messages.new_zeros((len(atom_types), messages.shape[1])).index_add_(
        0, edge_idx[:, 0], messages
    )
    divisors = (reduced[:, :2] + 0.25).sqrt()
    moments = torch.cat(
        (reduced[:, 2:34] / divisors[:, :1], reduced[:, 34:] / divisors[:, 1:]), 1
    )
    count = len(atom_types)
    vector = moments[:, 32:56].reshape(count, 3, 8)
    tensor = moments[:, 56:76].reshape(count, 5, 4)
    octupole = moments[:, 76:].reshape(count, 7, 1)

    def align(value, matrix):
        return value @ (
            torch.eye(matrix.shape[0], dtype=value.dtype, device=value.device) + matrix
        )

    vector = align(vector, model.alignment_1)
    tensor = align(tensor, model.alignment_2)
    blocks = (
        vector,
        tensor,
        octupole,
        vector @ model.projection_1,
        tensor @ model.projection_2,
        octupole @ model.projection_3,
    )
    invariants = _Polynomial.apply(
        torch.cat([v.flatten(1) for v in blocks], 1), _invariant_plan()
    )
    raw = torch.cat(
        (model.element_embedding(atom_types), moments[:, :32], divisors, invariants), 1
    )
    return (raw - model.calibration_mean) * model.calibration_inv_std, moments
