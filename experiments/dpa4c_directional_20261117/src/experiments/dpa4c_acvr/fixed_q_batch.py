"""Batched fixed-total-charge molecular energy, charge, forces and stationarity."""

import math
import torch
from torch.nn import functional as F
from curator.data import properties as p
from multiboundary import _readout_charge
from qefa_cuda import MolecularKernels


def reference_factors(e, query, key, pos_row):
    b, n, _ = pos_row.shape
    xyz = pos_row - pos_row.mean(1, keepdim=True)
    phase = (xyz @ e.attention.grid.T)[..., None] * e.attention.frequencies
    cos = phase.cos().unsqueeze(-2)
    sin = phase.sin().unsqueeze(-2)

    def rotary(x):
        v = x.reshape(b, n, 1, x.shape[-2], x.shape[-1])
        a, c = v[..., ::2], v[..., 1::2]
        return torch.stack((a * cos - c * sin, a * sin + c * cos), dim=-1).flatten(-3)

    # Flatten the quadrature directions/channels, contract over atoms once for each row.
    qr = (
        rotary(query)
        * e.attention.weights[None, None, :, None]
        / math.sqrt(query.shape[-2] * query.shape[-1])
    ).flatten(2)
    kr = rotary(key).flatten(2)

    return qr, kr


def panel_energy_charge(head, data, *, return_partial=False, geometry_correction=None):
    # Equal atom count, one fixed-total-charge zero-field condition per geometry.
    e = head.internal_energy
    prop = head.proposal
    pos = data[p.positions]
    n_atoms = data[p.n_atoms].reshape(-1)
    b = len(n_atoms)
    n = pos.shape[0] // b
    var = data["variational"]
    capturing = pos.is_cuda and torch.cuda.is_current_stream_capturing()
    periodic = hasattr(e.attention, "prepare_periodic_batch")
    if not capturing and not (
        n > 0
        and bool((n_atoms == n).all())
        and bool((data[p.pbc] == periodic).all())
        and torch.equal(
            var["condition_geometry_index"], torch.arange(b, device=pos.device)
        )
    ):
        raise ValueError(
            "The batched charge head requires a homogeneous fixed-Q batch matching its PBC backend"
        )
    for bc in var["boundary_spec"]:
        if bc["controls"] or len(bc["equalities"]) != 1 or bc.get("environment"):
            raise ValueError("The batched charge head requires fixed-charge-only conditions")
    total = data[p.total_charge].reshape(b)
    h = e.feature_spec.flatten(data)
    features = h.reshape(b, n, -1)
    norm_h = e.feature_norm(h)
    projection_context = {"_charge_efa_input": norm_h}
    tensor = getattr(e, "tensor_branch", None)
    if tensor is not None:
        projection_context["_dpa_moments"] = data[p.node_embedding]
    query, key, value = e._project_features(h, projection_context)
    pos_row = pos.reshape(b, n, 3)
    optimized = periodic or isinstance(
        e.attention, MolecularKernels
    ) and e.attention.supports_fused(query)
    if optimized:
        if periodic:
            communication = e.attention.prepare_periodic_batch(
                query, key, pos_row, data[p.cell].reshape(b, 3, 3)
            )
        else:
            communication = e.attention.prepare_batch(query, key, pos_row)
        apply = communication.apply_batch

        def transpose(v):
            return communication.apply_batch(v, transpose=True)
    else:
        qr, kr = reference_factors(e, query, key, pos_row)

        def apply(v):
            return qr @ (kr.mT @ v)

        def transpose(v):
            return kr @ (qr.mT @ v)

    offset = e.charge_affine.weight[:, e.hidden_features :]
    cond = (
        torch.stack((total, total / n), -1)[:, None, :].expand(b, n, 2)
        / prop.charge_scale
    )
    if periodic:
        cell = data[p.cell].reshape(b, 3, 3)
        # The scalar triple product is capture-safe, including on older CUDA builds.
        volume = (cell[:, 0] * torch.linalg.cross(cell[:, 1], cell[:, 2])).sum(-1).abs()
        cond = prop._periodic_condition(cond, volume)
    potential = pos.new_zeros(b, n, 2)
    local = (
        prop.input(features)
        + prop.boundary_input(cond)
        + prop.potential_input(potential)
    )
    geom = apply(value.reshape(b, n, -1))
    drive, bias, charge_value = e._coefficients(
        norm_h.reshape(b, n, -1), geom, transpose
    )
    direction = getattr(e, "direction_field", None)
    if direction is not None:
        dc = direction(features, pos_row, e.attention, cell[0] if periodic else None,
                       cached_waves=periodic)
        drive = drive + dc
    local = prop._inject(local, prop._physical_input(drive, bias, charge_value))

    def fuse(module, loc, glob):
        return module(loc.flatten(0, 1), glob.flatten(0, 1)).reshape(b, n, -1)

    first = fuse(prop.fusion, local, apply(prop.norm(local)))
    last = fuse(prop.extra_fusion, first, transpose(prop.norm(first)))
    factors = prop.response_factors(prop._invariants(features, last, cond))
    charge = _readout_charge(
        drive,
        factors,
        total,
        torch.ones_like(total, dtype=torch.bool),
        prop.operator_epsilon,
    )
    response = F.linear(torch.cat((apply(charge * charge_value), charge), -1), offset)
    lin = torch.sigmoid(bias) * response
    curve = F.softplus(bias + response) - F.softplus(bias) - lin
    terms = (
        e.positive_output_weights * (e.driving_scale * lin + e.curvature_scale * curve)
    ).sum((1, 2))
    U = (
        data[p.energy].reshape(b)
        + terms
        + 0.5 * e.hardness * charge.square().sum((1, 2))
    )
    if direction is not None:
        U = U + (charge * dc).sum((1, 2))
    if tensor is not None and tensor.ranks:
        left, right = (communication.q, communication.k) if optimized else (qr, kr)
        readout_input = {"normalized_features": norm_h.reshape(b, n, -1)} if hasattr(
            tensor, "chemical_gate"
        ) else {}
        U = U + tensor.orient_energy(
            data[p.node_embedding].reshape(b, n, -1), left, right, **readout_input
        )
    if geometry_correction is not None:
        left, right = (communication.q, communication.k) if optimized else (qr, kr)
        U = U + geometry_correction(data, left, right)
    if return_partial and optimized:
        # At fixed geometry response is linear in q. Its adjoint gives dU/dq,
        # retaining the graph of q and of every physical coefficient.
        total_slope = torch.where(
            bias + response > 20, torch.ones_like(bias), torch.sigmoid(bias + response)
        )
        slope = e.positive_output_weights * (
            (e.driving_scale - e.curvature_scale) * torch.sigmoid(bias)
            + e.curvature_scale * total_slope
        )
        source = slope @ offset
        dq = (
            e.hardness * charge
            + source[..., -1:]
            + (charge_value * transpose(source[..., :-1])).sum(-1, keepdim=True)
        )
        if direction is not None:
            dq = dq + dc
        return U, charge, dq
    if return_partial:
        return U, charge, None
    return U, charge


def panel_forward(head, data):
    U, charge, dq = panel_energy_charge(head, data, return_partial=True)
    b, n, _ = charge.shape
    pos = data[p.positions]
    var = data["variational"]
    # Earlier node (positions) receives its full path through the later node charge.
    # Simultaneously obtain the partial derivative at charge for the stationarity target.
    if dq is None:
        dR, dq = torch.autograd.grad(
            U.sum(), (pos, charge), create_graph=True, retain_graph=True
        )
    else:
        dR = torch.autograd.grad(U.sum(), pos, create_graph=True, retain_graph=True)[0]
    resid = dq - dq.mean(1, keepdim=True)
    state_ptr = var["condition_state_ptr"]

    def pack(tensor, unit, shape):
        stride = tensor.numel() // b
        return {
            "values": tensor.reshape(-1),
            "ptr": torch.arange(b + 1, device=pos.device) * stride,
            "shape": (shape,) * b,
            "unit": unit,
        }

    labels = {
        "condition_energy": pack(U, "eV", (1,)),
        "condition_forces": pack(-dR, "eV/Angstrom", (n, 3)),
        "variable_value": pack(charge, "e", (n, 1)),
    }
    data["variational_prediction"] = {
        "value_space": "physical",
        "labels": labels,
        "residuals": {"stationarity": {"values": resid.reshape(-1), "ptr": state_ptr}},
        "metadata": {
            "state_semantics": "variational_internal",
            "energy_convention": "internal",
            "force_potential": "thermodynamic",
        },
        "condition_geometry_index": var["condition_geometry_index"],
    }
    return data
