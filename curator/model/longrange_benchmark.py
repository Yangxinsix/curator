"""Opt-in composition of physical-energy Curator models and long-range adapters."""
import copy

import torch
from curator.data import properties
from curator.layer import PairwiseDistance, GlobalRescaleShift
from curator.layer._force_output import ForceOutput
from curator.layer._grad_output import GradientOutput
from curator.layer._physical_energy_rescale import PhysicalEnergyRescale
from curator.model.base import NeuralNetworkPotential


def augment_energy_model(base_model, adapter):
    """Return an independent model with an additive energy adapter.

    Insert after physical rescaling and before the conservative force output.
    The original model and its parameters are never mutated.  Existing ACVR
    variational outputs are deliberately not accepted as a short-range baseline.
    Stress and deployment are outside this energy/force benchmark contract.
    """
    if not isinstance(base_model, NeuralNetworkPotential):
        raise TypeError("base_model must be a Curator NeuralNetworkPotential")
    if not isinstance(adapter, torch.nn.Module):
        raise TypeError("adapter must be a torch.nn.Module")
    if any(getattr(m, "joint_variational_output", False)
           for m in base_model.output_modules):
        raise ValueError("Use the pure short-range baseline, not an ACVR model")
    original_outputs = list(base_model.output_modules)
    forces = [m for m in original_outputs
              if isinstance(m, ForceOutput) and m.produces_forces]
    if len(forces) != 1 or forces[0] is not original_outputs[-1]:
        raise ValueError("Baseline must end in one conservative GradientOutput")
    if not isinstance(forces[0], GradientOutput):
        raise ValueError("Direct-force baselines cannot supply conservative adapter forces")
    if any(key in base_model.model_outputs for key in ("stress", "virial")):
        raise ValueError("This benchmark assembly supports energy and forces, not stress")
    clone = copy.deepcopy(base_model)
    output_modules = list(clone.output_modules[:-1])
    for index, module in enumerate(output_modules):
        if isinstance(module, GlobalRescaleShift):
            output_modules[index] = PhysicalEnergyRescale(module)
    if not any(isinstance(m, PhysicalEnergyRescale) for m in output_modules):
        raise ValueError("An explicit physical-energy rescale is required")
    inputs = list(clone.input_modules)
    pair_modules = [m for m in inputs if isinstance(m, PairwiseDistance)]
    if not pair_modules:
        raise ValueError("Baseline requires PairwiseDistance coordinate preprocessing")
    for module in pair_modules:
        module.compute_distance_from_R = True
        module.compute_forces = True
    head = copy.deepcopy(adapter)
    if hasattr(head, "bind"):
        head.bind(clone.representation.response_feature_spec())
    reference = next(clone.parameters(), None)
    if reference is not None:
        head.to(device=reference.device, dtype=reference.dtype)
    output_modules.extend([
        head,
        GradientOutput(grad_on_edge_diff=False, grad_on_positions=True,
                       model_outputs=[properties.forces]),
    ])
    model = NeuralNetworkPotential(
        representation=clone.representation, input_modules=inputs,
        output_modules=output_modules,
        model_outputs=[properties.energy, properties.forces],
        heads=copy.deepcopy(getattr(clone, "heads", None)),
    )
    model._initialized = getattr(clone, "_initialized", False)
    if hasattr(clone, "_acvr_energy_reference"):
        model._acvr_energy_reference = copy.deepcopy(clone._acvr_energy_reference)
    model.train(base_model.training)
    return model

