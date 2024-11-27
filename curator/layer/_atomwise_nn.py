import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Union, Optional, Callable, Dict
from e3nn import o3
from e3nn.nn import Activation
from curator.data.properties import activation_fn
import warnings

class Dense(nn.Module):
    r"""
    Fully connected linear layer with activation function.

    .. math::
       y = activation(x W^T + b)
    """

    def __init__(
        self,
        in_features: Union[int, o3.Irreps],
        out_features: Union[int, o3.Irreps],
        activation: Union[Callable, nn.Module] = None,
        use_e3nn: bool = False,
        *args,
        **kwargs,
    ):
        """
        Args:
            in_features: number of input feature :math:`x`.
            out_features: umber of output features :math:`y`.
            bias: If False, the layer will not adapt bias :math:`b`.
            activation: if None, no activation function is used.
        """
        if use_e3nn:
            self.linear = o3.Linear(in_features, out_features, *args, **kwargs)
            self.activation = Activation(irreps_in=in_features, acts=[activation or nn.Identity()])
        else:
            self.linear = nn.Linear(in_features, out_features, *args, **kwargs)
            self.activation = activation or nn.Identity()

    def forward(self, input: torch.Tensor):
        y = self.linear(input)
        y = self.activation(y)
        return y

class AtomwiseNN(nn.Module):
    # 1. Output structure-based property: per_atom=False, aggregation_mode='mean' or 'sum'
    # 2. Output structure-based property + per-atom property: per_atom=True, aggregation_mode='mean' or 'sum'
    # 3. Output per-atom property: per_atom=False, aggregation_mode = None
    def __init__(
        self,
        in_features: Union[int, o3.Irreps],
        out_features: Union[int, o3.Irreps],
        n_hidden: Union[List[int], List[o3.Irreps], int, o3.Irreps, None],
        n_hidden_layers: int = 1,
        use_e3nn: bool = False,
        activation: Union[Callable, nn.Module, str, List[Callable], List[nn.Module], List[str]] = 'silu',
        output_keys: Union[List[str], List[Dict]] = ["energy"],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.use_e3nn = use_e3nn
        self.output_keys = output_keys

        # number of neurons
        n_neurons = [in_features]
        if n_hidden is None:
            for _ in range(n_hidden_layers):
                n_neurons.append(in_features)
        elif isinstance(n_hidden, list):
            if len(n_hidden) != n_hidden_layers:
                self.n_hidden_layers = n_hidden
                warnings.warn(f"n_hidden ({len(n_hidden)}) does not equal to n_hidden_layers ({n_hidden_layers}).")
            for neuron in n_hidden:
                n_neurons.append(neuron)
        else:
            for _ in range(n_hidden_layers):
                n_neurons.append(n_hidden)
        n_neurons.append(out_features)

        # activations
        acts = []
        if isinstance(activation, list):
            acts = [activation_fn[act] if isinstance(act, str) else act for act in activation] + [None]
            acts = [activation_fn[activation] for _ in range(n_hidden_layers)] + [None]
        else:
            acts = [activation_fn[activation] if isinstance(activation, str) else activation for _ in range(n_hidden_layers)] + [None]

        # layers
        layers = [Dense(n_neurons[i], n_neurons[i+1], acts[i]) for i in self.n_hidden_layers + 1]
        self.readout_mlp = nn.Sequential(**layers)

        # output keys
        n_out = out_features if isinstance(out_features, int) else out_features.dim
        self.output_keys = output_keys
        # 1. Output structure-based property: per_atom=False, aggregation_mode='mean' or 'sum'
        # 2. Output structure-based property + per-atom property: per_atom=True, aggregation_mode='mean' or 'sum'
        # 3. Output per-atom property: per_atom=False, aggregation_mode = None
        for i, spec in enumerate(self.output_keys):
            if isinstance(spec, str):
                self.output_keys[i] = {
                    'key': spec,
                    'per_atom': False,            # this means per_atom property will be outputed
                    'aggregation_mode': 'mean',   # if this property is per atom, use None and output as is
                    'per_atom_key': spec + '_pa',  # name of per_atom property
                    'split_size': 1,
                }
            else:
                if 'key' not in spec:
                    raise ValueError("No output key is specified!")
                if 'aggregation_mode' not in spec:
                    spec['aggregation_mode'] = 'mean'  # Default to mean aggregation
                if 'per_atom' not in spec:
                    spec['per_atom'] = False
                if 'split_size' not in spec:
                    spec['split_size'] = 1
        self.split_size = [spec['split_size'] for spec in self.output_keys]
        assert sum(self.split_size) == n_out, "The dimensionality of output features does not match number of output keys!"

        def forward(self, input: torch.Tensor):
            pass
            out = self.readout_mlp(input)