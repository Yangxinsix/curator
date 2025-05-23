import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Union, Optional, Callable, Dict, NamedTuple
from e3nn import o3
from e3nn.nn import Activation
from curator.data.properties import activation_fn
from curator.data import properties
from ._cuequivariance_wrapper import Linear
import warnings
try:
    from torch_scatter import scatter_add, scatter_mean
except ImportError:
    from curator.utils import scatter_add, scatter_mean

class OutputSpec(NamedTuple):
    key: str                         # name of this properties
    per_atom: bool                   # if per_atom property should be output
    aggregation_mode: Optional[str]  # sum, mean or None, when use None output as is, means this property is per atom
    per_atom_key: str                # the key of per_atom property
    split_size: int                  # dim size of this property

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
        super().__init__()
        if use_e3nn:
            assert isinstance(in_features, o3.Irreps), "in_features must be e3nn.o3.Irreps when using e3nn Linear layer!"
            if isinstance(out_features, int):
                out_features = o3.Irreps(f'{out_features}x0e')
            self.linear = Linear(in_features, out_features, *args, **kwargs)
            self.activation = Activation(irreps_in=out_features, acts=[activation])
        else:
            assert isinstance(in_features, int), 'in_features must be interger for torch.nn.Linear layer!'
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
        in_features: Union[int, o3.Irreps, str],
        out_features: Union[int, o3.Irreps, str] = 1,
        n_hidden: Union[List[int], List[o3.Irreps], int, o3.Irreps, None] = None,
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

        if isinstance(in_features, str):
            in_features = o3.Irreps(in_features)
        if isinstance(out_features, str):
            out_features = o3.Irreps(out_features)

        # number of neurons
        n_neurons = [in_features]
        if n_hidden is None:
            for _ in range(n_hidden_layers):
                if use_e3nn:
                    mul = in_features.sort()[0][0].mul // 2
                    mid_neuron = o3.Irreps(f'{mul}x0e')
                else:
                    mid_neuron = in_features
                n_neurons.append(mid_neuron)
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
        layers = [Dense(n_neurons[i], n_neurons[i+1], acts[i], use_e3nn=use_e3nn) for i in range(self.n_hidden_layers + 1)]
        self.readout_mlp = nn.Sequential(*layers)

        # output keys
        n_out = out_features if isinstance(out_features, int) else out_features.dim
        self.output_specs: List[OutputSpec] = []
        # 1. Output structure-based property: per_atom=False, aggregation_mode='mean' or 'sum'
        # 2. Output structure-based property + per-atom property: per_atom=True, aggregation_mode='mean' or 'sum'
        # 3. Output per-atom property: per_atom=False, aggregation_mode = None
        for spec in output_keys:
            if isinstance(spec, str):
                self.output_specs.append(OutputSpec(
                    key=spec,
                    per_atom=False,
                    aggregation_mode='sum',
                    per_atom_key=spec + '_pa',
                    split_size=1, 
                ))
            else:
                if 'key' not in spec:
                    raise ValueError("No output key is specified!")
                self.output_specs.append(OutputSpec(
                    key=spec['key'],
                    per_atom=spec.get('per_atom', False),
                    aggregation_mode=spec.get('aggregation_mode', 'sum'),
                    per_atom_key=spec.get('per_atom_key', spec['key'] + '_pa'),
                    split_size=spec.get('split_size', 1),
                ))

        self.model_outputs = [spec.key for spec in self.output_specs]
        self.split_size: List[int] = [spec.split_size for spec in self.output_specs]
        assert sum(self.split_size) == n_out, "The dimensionality of output features does not match number of output keys!"

    def _compute(self, input: torch.Tensor, index: Optional[torch.Tensor] = None) -> properties.Type:
        out = self.readout_mlp(input)
        return out

    def _parse_outputs(self, out: torch.Tensor, index: Optional[torch.Tensor] = None) -> properties.Type:
        out = out.split(self.split_size, dim=1)
        output_dict: Dict[str, torch.Tensor] = {}
        for i, spec in enumerate(self.output_specs):
            prop = out[i].squeeze()
            key = spec[0]
            per_atom = spec[1]
            aggregation_mode = spec[2]
            per_atom_key = spec[3]

            # per-atom property
            if per_atom:
                output_dict[per_atom_key] = prop
            
            if aggregation_mode is not None:
                if aggregation_mode == 'sum':
                    output_dict[key] = scatter_add(prop, index, dim=0) if index is not None else torch.sum(prop, dim=0)
                if aggregation_mode == 'mean':
                    output_dict[key] = scatter_mean(prop, index, dim=0) if index is not None else torch.mean(prop, dim=0)
                if aggregation_mode == 'None':
                    output_dict[key] = prop
            else:
                output_dict[key] = prop       # output as is

        return output_dict

    def forward(self, data: properties.Type) -> properties.Type:
        if properties.image_idx not in data:
            data[properties.image_idx] = torch.zeros(data[properties.n_atoms].item(), dtype=data[properties.edge_idx].dtype, device=data[properties.edge_idx].device)
        
        input = data[properties.node_feat]
        index = data[properties.image_idx]
        
        out = self._compute(input)
        output_dict = self._parse_outputs(out, index)
        data.update(output_dict)

        return data

class MACEAtomwiseNN(AtomwiseNN):
    """Atomwise feed-forward neural networks for MACE

    Args:
        num_layers: number of message-passing layers in MACE
        hidden_irreps: hidden_irreps in MACE
    """
    def __init__(
        self,
        num_interactions: int,
        hidden_irreps: Union[o3.Irreps, str, None] = None,
        MLP_irreps: Union[o3.Irreps, str, None] = None,
        lmax: int = 2,
        parity: bool = True,
        num_features: Optional[int] = None,
        *args,
        **kwargs,
    ):

        # hidden feature irreps
        if hidden_irreps is not None:
            self.hidden_irreps = o3.Irreps(hidden_irreps) if isinstance(hidden_irreps, str) else hidden_irreps
        else:
            self.hidden_irreps = o3.Irreps(
                [
                    (num_features, (l, p))
                    for p in ((1, -1) if parity else (1,))
                    for l in range(lmax + 1)
                ]
            ).sort()[0].simplify()
            # MACE prohibits some irreps like 0e, 1e to be used
            forbidden_ir = ['0o', '1e', '2o', '3e', '4o']
            self.hidden_irreps = o3.Irreps([irrep for irrep in self.hidden_irreps if str(irrep.ir) not in forbidden_ir])
        
        # MLP irreps
        if MLP_irreps is None:
            num_features = self.hidden_irreps.count(o3.Irrep(0, 1))
            self.MLP_irreps = o3.Irreps([(max(1, num_features // 2), (0, 1))])
        elif isinstance(MLP_irreps, str):
            self.MLP_irreps = o3.Irreps(MLP_irreps)
        else:
            self.MLP_irreps = MLP_irreps

        super().__init__(in_features=o3.Irreps(str(self.hidden_irreps[0])), n_hidden=self.MLP_irreps, use_e3nn=True, *args, **kwargs)
        self.num_interactions = num_interactions

        self.readouts = nn.ModuleList()
        self.in_features_list = []
        for _ in range(num_interactions - 1):
            self.in_features_list.append(self.hidden_irreps.dim)
            self.readouts.append(Dense(self.hidden_irreps, self.out_features, activation=None, use_e3nn=True))

        self.readouts.append(self.readout_mlp)
        self.in_features_list.append(o3.Irreps(str(self.hidden_irreps[0])))

    def _compute(self, input: torch.Tensor, index: Optional[torch.Tensor] = None) -> properties.Type:
        # split node features to list then calculate contributions from different parts
        node_feat_list = torch.split(input, self.in_features_list, dim=-1)
        out_list = []
        for readout, node_feat in zip(self.readouts, node_feat_list):
            out_list.append(readout(node_feat))
    
        out = torch.sum(torch.stack(out_list, dim=0), dim=0)
        return out