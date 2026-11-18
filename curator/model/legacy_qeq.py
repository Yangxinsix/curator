"""Preserve a pretrained MACE energy readout while appending legacy direct q."""
import copy
import torch
from e3nn import o3
from curator.layer._atomwise_nn import MACEAtomwiseNN


def append_legacy_charge_readout(readout, charge_initial_scale=0.01):
    """Expand each existing scalar output 1 -> 2, sharing the original trunk.

    Column zero is copied exactly, including e3nn path normalization. New charge
    columns use the same layerwise sum and final shared MLP as the old two-output
    MACE architecture. Callers provide the initialization RNG seed.
    """
    if not isinstance(readout, MACEAtomwiseNN) or readout.separate_heads:
        raise TypeError('legacy expansion requires a shared MACEAtomwiseNN')
    result = MACEAtomwiseNN(num_interactions=readout.num_interactions,
        hidden_irreps=readout.hidden_irreps, MLP_irreps=readout.MLP_irreps,
        out_features=2, heads=['energy', 'atomic_charge'])
    result.to(next(readout.parameters()))
    result.readout_mlp[0].load_state_dict(readout.readout_mlp[0].state_dict())
    old_layers = list(readout.readouts[:-1]) + [readout.readout_mlp[-1]]
    new_layers = list(result.readouts[:-1]) + [result.readout_mlp[-1]]
    with torch.no_grad():
        for old, new in zip(old_layers, new_layers):
            old_linear, new_linear = old.linear, new.linear
            if old_linear.irreps_out != o3.Irreps('1x0e') or new_linear.irreps_out != o3.Irreps('2x0e'):
                raise ValueError('expected scalar energy -> energy/charge expansion')
            new_linear.weight.mul_(charge_initial_scale)
            if len(old_linear.instructions) != len(new_linear.instructions):
                raise ValueError('expanded e3nn instruction layout differs')
            for index, (a, b) in enumerate(zip(old_linear.instructions, new_linear.instructions)):
                if (a.i_in, a.i_out, a.path_weight) != (b.i_in, b.i_out, b.path_weight):
                    raise ValueError('expanded e3nn path normalization differs')
                source = old_linear.weight_view_for_instruction(index)
                destination = new_linear.weight_view_for_instruction(index)
                destination[..., :1].copy_(source)
            if old_linear.bias.numel() or new_linear.bias.numel():
                raise ValueError('this adapter expects the pretrained bias-free readout')
    return result
