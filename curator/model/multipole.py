"""Opt-in assembly and parameter ownership for the global-state ACVR adapter."""
from collections import OrderedDict
import torch
from curator.model.base import NeuralNetworkPotential, ParameterGroup
from curator.model.longrange_benchmark import augment_energy_model
from curator.layer._multipole_variational_output import MultipoleVariationalOutput


class MultipolePotential(NeuralNetworkPotential):
    def forward(self, data):
        if torch.is_inference_mode_enabled():
            raise RuntimeError("multipole forces require inference_mode=False")
        with torch.enable_grad():
            return self._forward_modules(data)

    def parameter_groups(self):
        heads = [m for m in self.output_modules if isinstance(m, MultipoleVariationalOutput)]
        if len(heads) != 1:
            raise ValueError("MultipolePotential requires exactly one state energy output")
        energy = list(heads[0].energy_model.parameters())
        proposal = list(heads[0].proposal.parameters())
        energy_ids = {id(p) for p in energy}
        proposal_ids = {id(p) for p in proposal}
        if energy_ids & proposal_ids:
            raise ValueError("energy and proposal must not share trainable parameters")
        owned = energy_ids | proposal_ids
        groups = []
        for group in super().parameter_groups():
            params = [p for p in group.params if id(p) not in owned]
            if params:
                groups.append(ParameterGroup(group.name, params, group.defaults))
        groups.extend([ParameterGroup("state_energy", energy), ParameterGroup("proposal", proposal)])
        return groups

    def module_groups(self):
        groups = OrderedDict(super().module_groups())
        heads = [m for m in self.output_modules if isinstance(m, MultipoleVariationalOutput)]
        if len(heads) != 1:
            raise ValueError('MultipolePotential requires exactly one state energy output')
        groups['output_modules'] = [m for m in self.output_modules if m is not heads[0]]
        groups['state_energy'] = [heads[0].energy_model]
        groups['proposal'] = [heads[0].proposal]
        return groups


def augment_multipole_model(base_model, adapter=None):
    adapter = MultipoleVariationalOutput() if adapter is None else adapter
    augmented = augment_energy_model(base_model, adapter)
    model = MultipolePotential(representation=augmented.representation,
        input_modules=list(augmented.input_modules), output_modules=list(augmented.output_modules),
        model_outputs=list(augmented.model_outputs), heads=getattr(augmented, 'heads', None))
    model._initialized = augmented._initialized
    if hasattr(augmented, '_acvr_energy_reference'):
        model._acvr_energy_reference = augmented._acvr_energy_reference
    model.train(base_model.training)
    return model
