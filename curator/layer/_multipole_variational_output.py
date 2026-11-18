"""Compose learned q/dipole energy, feasible proposal and routed stationarity.

E/F supervision differentiates the entire proposal path. The residual replay
uses detached geometry and energy parameters, updating proposal parameters only.
No atomic charge or dipole labels are read. Ordinary GradientOutput computes
conservative forces after this adapter has added physical-unit energy.
"""
import torch
from torch import nn
from curator.data import properties
from curator.model.features import scalar_vector_features
from curator.variational.multipole_boundary import boundary_for_geometry


class MultipoleVariationalOutput(nn.Module):
    repr_callback = True

    def __init__(self, energy_model=None, proposal=None, charge_residual_scale=1.,
                 dipole_residual_scale=1., compute_residuals=True, state_mode='predicted'):
        super().__init__()
        from ._multipole_state_energy import NeuralMultipoleStateEnergy
        from ._multipole_state_proposal import MultipoleStateProposal
        self.energy_model = energy_model if energy_model is not None else NeuralMultipoleStateEnergy()
        self.proposal = proposal if proposal is not None else MultipoleStateProposal()
        if charge_residual_scale <= 0 or dipole_residual_scale <= 0:
            raise ValueError('residual scales must be positive fixed physical quantities')
        if state_mode not in ('predicted', 'reference'):
            raise ValueError('state_mode must be predicted or reference')
        self.charge_residual_scale = float(charge_residual_scale)
        self.dipole_residual_scale = float(dipole_residual_scale)
        self.compute_residuals = bool(compute_residuals)
        self.state_mode = state_mode
        self.feature_spec = None
        self.model_outputs = [properties.energy, 'multipole_energy', 'multipole_state',
            'multipole_charge_residual', 'multipole_dipole_residual',
            'multipole_charge_rms', 'multipole_dipole_rms', 'multipole_feasibility']
        self.register_buffer('_dtype_anchor', torch.empty(0))

    def bind(self, feature_spec):
        if self.feature_spec is not None:
            if self.feature_spec != feature_spec:
                raise ValueError('multipole feature specification changed')
            return self
        self.feature_spec = feature_spec
        for module in (self.energy_model, self.proposal):
            module.bind(feature_spec=feature_spec)
            module.to(self._dtype_anchor)
        return self

    def register_repr_callback(self, model):
        self.bind(model.representation.response_feature_spec())
        reference = next(model.representation.parameters())
        self.to(device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _rms(value):
        return value.square().mean().clamp_min(1.e-30).sqrt()

    def _residual(self, context, boundary, predicted):
        context = {k: v.detach() for k, v in context.items()}
        boundary = boundary.detached()
        state = boundary.project(self.proposal(context))
        if not torch.allclose(state.detach(), predicted.detach(), rtol=1.e-6, atol=1.e-7):
            raise RuntimeError('physical and residual proposal paths disagree')
        if not state.requires_grad:
            state = state.detach().requires_grad_(True)
        params = {k: v.detach() for k, v in self.energy_model.named_parameters()}
        buffers = {k: v.detach().clone() for k, v in self.energy_model.named_buffers()}
        energy = torch.func.functional_call(self.energy_model, (params, buffers), (context, state)) + boundary.work(state)
        gradient = torch.autograd.grad(energy, state, create_graph=self.training, retain_graph=True)[0]
        return boundary.tangent(gradient)

    def forward(self, data):
        if self.feature_spec is None:
            raise RuntimeError('bind multipole output before constructing optimizer')
        counts = data[properties.n_atoms].reshape(-1)
        features = self.feature_spec.flatten(data)
        if int(counts.sum()) != len(features) or bool((counts <= 0).any()):
            raise ValueError('atom counts do not partition representation features')
        scalar, vector = scalar_vector_features(features, self.feature_spec.irreps)
        batch = data.get(properties.image_idx)
        if batch is not None:
            expected = torch.repeat_interleave(torch.arange(len(counts), device=counts.device), counts)
            if not torch.equal(batch, expected):
                raise ValueError('multipole output requires structure-contiguous atom order')
        if self.training and self.state_mode != 'predicted':
            raise ValueError('reference mode is evaluation only; training retains the full proposal path')
        energies, states, charge_rms, dipole_rms, feasibility = [], [], [], [], []
        start = 0
        with torch.enable_grad():
            for geometry, count_tensor in enumerate(counts):
                stop = start + int(count_tensor)
                boundary = boundary_for_geometry(data, geometry, start, stop)
                context = {'features': features[start:stop], 'scalar': scalar[start:stop],
                    'vector': vector[start:stop], 'positions': data[properties.positions][start:stop],
                    'pbc': data[properties.pbc].reshape(-1, 3)[geometry] if properties.pbc in data else torch.zeros(3, dtype=torch.bool, device=features.device),
                    **boundary.proposal_inputs()}
                if hasattr(self.energy_model, 'prepare_context'):
                    context = self.energy_model.prepare_context(context)
                predicted = boundary.project(self.proposal(context))
                state = predicted
                if self.state_mode == 'reference':
                    from curator.variational.functionals import solve_functional_reference
                    from curator.variational.nonlinear import NewtonKKTOptions
                    problem = boundary.problem(lambda x: self.energy_model(context, x.reshape(-1, 4)) + boundary.work(x))
                    solved = solve_functional_reference(problem, state.detach().reshape(-1), options=NewtonKKTOptions(stationarity_atol=1.e-6, stationarity_rtol=1.e-6))
                    if not solved.converged or not solved.stable:
                        raise RuntimeError('nonlinear reference state is not a certified minimum: ' + solved.status)
                    state = solved.state.detach().reshape(-1, 4)
                correction = self.energy_model(context, state) + boundary.work(state)
                if self.state_mode == 'reference':
                    correction = correction + solved.multipliers.detach() @ problem.constraints(state.reshape(-1))
                if not torch.isfinite(correction) or not torch.isfinite(state).all():
                    raise RuntimeError('nonfinite learned multipole energy/state')
                energies.append(correction)
                states.append(state)
                violation = boundary.matrix @ state.reshape(-1) - boundary.target
                feasibility.append(violation.abs().max().detach() if violation.numel() else features.new_zeros(()))
                if self.compute_residuals:
                    if self.state_mode == 'predicted':
                        residual = self._residual(context, boundary, predicted)
                    else:
                        residual = problem.stationarity(state.reshape(-1)).stationarity.reshape(-1, 4)
                    charge_rms.append(self._rms(residual[:, 0]))
                    dipole_rms.append(self._rms(residual[:, 1:]))
                else:
                    charge_rms.append(features.new_full((), float('nan')))
                    dipole_rms.append(features.new_full((), float('nan')))
                start = stop
        correction = torch.stack(energies)
        data['multipole_energy'] = correction
        data[properties.energy] = data[properties.energy] + correction.reshape_as(data[properties.energy])
        data['multipole_state'] = torch.cat(states)
        data['multipole_charge_rms'] = torch.stack(charge_rms)
        data['multipole_dipole_rms'] = torch.stack(dipole_rms)
        data['multipole_charge_residual'] = data['multipole_charge_rms'] / self.charge_residual_scale
        data['multipole_dipole_residual'] = data['multipole_dipole_rms'] / self.dipole_residual_scale
        data['multipole_feasibility'] = torch.stack(feasibility)
        return data
