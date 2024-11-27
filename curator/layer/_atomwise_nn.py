import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Union, Optional, Callable
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
    def __init__(
        self,
        in_features: Union[int, o3.Irreps],
        out_features: Union[int, o3.Irreps],
        n_hidden: Union[List[int], List[o3.Irreps], int, o3.Irreps, None],
        n_hidden_layers: int = 1,
        use_e3nn: bool = False,
        activation: Union[Callable, nn.Module, str, List[Callable], List[nn.Module], List[str]] = 'silu',
        output_keys: List[str] = ["energy"],
        per_atom_output_keys: Union[List[str], None] = None,
        aggregation_mode: Union[List[str], str] = "sum",     # should be sum or mean
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

        def forward(self, input: torch.Tensor):


