"""Charge constraints and external work for [q, px, py, pz] states.

Charge is measured in e, dipoles in e Angstrom; conjugate quantities are V
and V/Angstrom. This is a finite-system boundary adapter, not an electrostatic
interaction model. Intrinsic interactions remain entirely in the learned energy.
"""
from dataclasses import dataclass, fields, replace
import torch
from curator.data import properties
from curator.data.variational_data import canonical_boundary_hash, condition_input
from curator.variational.compiler import compile_boundary
from curator.variational.functionals import FunctionalProblem
from curator.variational.spec import BoundarySpec, ConstraintSpec, GroupSpec, VariableSpec, VariableLayout


@dataclass(frozen=True)
class MultipoleBoundary:
    charge_boundary: object
    external_conjugate: torch.Tensor
    matrix: torch.Tensor
    target: torch.Tensor

    @classmethod
    def compile(cls, spec, positions, atom_fields=None, *, potential=None,
                uniform_field=None, local_field=None, control_value=None,
                constraint_target=None):
        n = len(positions)
        if spec.variable.components != 1 or spec.variable.unit != 'e' or spec.variable.conjugate_unit != 'V':
            raise ValueError('multipole boundaries require one physical charge per atom (e, V)')
        layout = VariableLayout(spec.variable, n, atom_fields or {}, dtype=positions.dtype, device=positions.device)
        compiled = compile_boundary(spec, layout)
        if compiled.environment_matrix.shape[0]:
            raise ValueError('multipole v1 supports charge constraints and voltage controls, not Robin environments')
        updates = {}
        for name, value in [('control_value', control_value), ('constraint_target', constraint_target)]:
            if value is not None:
                value = value.to(positions)
                if value.shape != getattr(compiled, name).shape:
                    raise ValueError(name + ' shape differs from boundary specification')
                updates[name] = value
        compiled = replace(compiled, **updates)
        phi = positions.new_zeros(n) if potential is None else potential.to(positions)
        field = positions.new_zeros(3) if uniform_field is None else uniform_field.to(positions)
        local = positions.new_zeros((n, 3)) if local_field is None else local_field.to(positions)
        if phi.shape != (n,) or field.shape != (3,) or local.shape != (n, 3):
            raise ValueError('external potential/field must have shapes [N], [3], [N,3]')
        if not all(torch.isfinite(v).all() for v in (phi, field, local)):
            raise ValueError('external controls must be finite')
        # W_ext = q phi - E.(q R + p); local E only couples to local p.
        # Spatially varying phi tensors must retain their coordinate graph.
        conjugate = torch.cat(((-phi + positions @ field)[:, None], local + field[None, :]), dim=1)
        matrix = positions.new_zeros((compiled.constraint_matrix.shape[0], 4 * n))
        matrix[:, 0::4] = compiled.constraint_matrix
        return cls(compiled, conjugate, matrix, compiled.constraint_target)

    @property
    def n_atoms(self):
        return self.external_conjugate.shape[0]

    def problem(self, energy):
        return FunctionalProblem(energy, 4 * self.n_atoms, self.matrix, self.target)

    def project(self, state):
        return self.problem(lambda x: x.sum() * 0).project_state(state.reshape(-1)).reshape(self.n_atoms, 4)

    def tangent(self, vector):
        vector = vector.reshape(-1)
        if self.matrix.shape[0]:
            vector = vector - self.matrix.T @ torch.linalg.solve(self.matrix @ self.matrix.T, self.matrix @ vector)
        return vector.reshape(self.n_atoms, 4)

    def proposal_inputs(self):
        field = self.external_conjugate.clone()
        compiled = self.charge_boundary
        if compiled.control_matrix.shape[0]:
            field = field + torch.cat(((compiled.control_matrix.T @ compiled.control_value)[:, None], field.new_zeros((self.n_atoms, 3))), dim=1)
        return {
            'origin': self.project(field.new_zeros((self.n_atoms, 4))),
            'network_field': self.tangent(field),
            'constraint_matrix': self.matrix,
            'constraint_target': self.target,
        }

    def work(self, state):
        state = state.reshape(self.n_atoms, 4)
        return self.charge_boundary.energy(state[:, 0]) - (self.external_conjugate * state).sum()

    def detached(self):
        compiled = replace(self.charge_boundary, **{
            f.name: getattr(self.charge_boundary, f.name).detach()
            for f in fields(self.charge_boundary)
            if isinstance(getattr(self.charge_boundary, f.name), torch.Tensor)
        })
        return MultipoleBoundary(compiled, self.external_conjugate.detach(), self.matrix.detach(), self.target.detach())


def boundary_for_geometry(data, geometry, start, stop):
    """One condition per geometry; structured charge controls remain authoritative."""
    positions = data[properties.positions][start:stop]
    payload = data.get('variational')
    condition = None
    if payload is not None:
        from curator.layer._variational_output import _boundary_from_dict
        matches = (payload['condition_geometry_index'] == geometry).nonzero().flatten()
        if len(matches) != 1:
            raise ValueError('multipole energy adapter requires exactly one condition per geometry')
        condition = int(matches[0])
        raw = payload['boundary_spec'][condition]
        if canonical_boundary_hash(raw) != payload['boundary_hash'][condition]:
            raise ValueError('boundary hash mismatch')
        spec = _boundary_from_dict(raw)
    else:
        constraints = ()
        if properties.total_charge in data:
            charge = float(data[properties.total_charge].reshape(-1)[geometry])
            constraints = (ConstraintSpec.group_sum('total_charge', 'all', charge, unit='e'),)
        spec = BoundarySpec(VariableSpec('charge', 'charge', unit='e', conjugate_unit='V'), groups=(GroupSpec.all(),), equalities=constraints)
    atom_fields = {k: v[start:stop] for k, v in data.items()
                   if isinstance(v, torch.Tensor) and v.ndim and v.shape[0] == len(data[properties.positions])}
    overrides = {}
    if condition is not None:
        for name in ('constraint_target', 'control_value'):
            overrides[name] = condition_input(payload, name, condition)
    uniform = data.get('external_field')
    if uniform is not None:
        uniform = uniform.reshape(-1, 3)[geometry]
    if condition is not None:
        supplied = condition_input(payload, 'external_field', condition)
        if supplied is not None:
            if supplied.numel() != 3:
                raise ValueError('multipole external_field must be a physical electric vector with 3 components')
            uniform = supplied.reshape(3)
    potential = data.get('external_potential')
    local = data.get('local_electric_field')
    return MultipoleBoundary.compile(spec, positions, atom_fields,
        potential=None if potential is None else potential[start:stop],
        uniform_field=uniform, local_field=None if local is None else local[start:stop], **overrides)
