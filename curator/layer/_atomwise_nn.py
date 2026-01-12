import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Union, Optional, Callable, Dict, Tuple
from e3nn import o3
from e3nn.nn import Activation
from curator.data.properties import activation_fn, HeadConfig, resolve_heads
from curator.data import properties
from ._cuequivariance_wrapper import Linear
import warnings
try:
    from torch_scatter import scatter_add, scatter_mean
except ImportError:
    from curator.utils import scatter_add, scatter_mean

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
    def __init__(
        self,
        in_features: Union[int, o3.Irreps, str],
        out_features: Union[int, o3.Irreps, str] = 1,
        n_hidden: Union[List[int], List[o3.Irreps], int, o3.Irreps, None] = None,
        n_hidden_layers: int = 1,
        use_e3nn: bool = False,
        activation: Union[Callable, nn.Module, str, List[Callable], List[nn.Module], List[str]] = 'silu',
        heads: Optional[List[Union[HeadConfig, Dict, str]]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if isinstance(in_features, str):
            in_features = o3.Irreps(in_features)
        if isinstance(out_features, str):
            out_features = o3.Irreps(out_features)

        self.in_features = in_features
        self.out_features = out_features
        self.use_e3nn = use_e3nn
        self.n_hidden_layers = n_hidden_layers

        # Setup neuron sizes
        n_neurons = [in_features]
        if n_hidden is None:
            for _ in range(n_hidden_layers):
                mid_neuron = o3.Irreps(f'{in_features.sort()[0][0].mul // 2}x0e') if use_e3nn else in_features
                n_neurons.append(mid_neuron)
        elif isinstance(n_hidden, list):
            if len(n_hidden) != n_hidden_layers:
                self.n_hidden_layers = len(n_hidden)
            n_neurons.extend(n_hidden)
        else:
            n_neurons.extend([n_hidden] * n_hidden_layers)
        n_neurons.append(out_features)

        # Setup activations
        if isinstance(activation, list):
            acts = [activation_fn[act] if isinstance(act, str) else act for act in activation] + [None]
        else:
            acts = [activation_fn[activation] if isinstance(activation, str) else activation for _ in range(self.n_hidden_layers)] + [None]

        self._n_neurons = n_neurons
        self._acts = acts

        self.readout_mlp = self._make_readout()

        n_out = out_features if isinstance(out_features, int) else out_features.dim
        self._n_out = int(n_out)

        # Prepare and store output specifications using HeadConfig
        self.heads: List[HeadConfig] = resolve_heads(heads) if heads is not None else []
        if not self.heads:
            self.heads = [HeadConfig(key="energy", is_atomwise=True, reduction="sum", atomwise_key="atomic_energy")]

        self.model_outputs = [h.key for h in self.heads]
        self.per_atom_flags = [bool(h.write_atomwise) for h in self.heads]
        self.aggregation_modes = [(h.reduction if h.reduction is not None else "sum") for h in self.heads]
        self.per_atom_keys = [(h.atomwise_key or (h.key + "_pa")) for h in self.heads]
        self.split_size = [int(h.dim) for h in self.heads]

        assert sum(self.split_size) == n_out, "Output feature dimensions do not match split sizes!"

    def _compute(self, input: torch.Tensor) -> torch.Tensor:
        return self.readout_mlp(input)

    def _parse_outputs(self, out: torch.Tensor, index: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        out = out.split(self.split_size, dim=1)
        output_dict: Dict[str, torch.Tensor] = {}

        for i, head in enumerate(self.heads):
            prop = out[i].squeeze(-1)
            key = self.model_outputs[i]
            per_atom = self.per_atom_flags[i]
            aggregation_mode = self.aggregation_modes[i]
            per_atom_key = self.per_atom_keys[i]

            if per_atom:
                output_dict[per_atom_key] = prop

            if aggregation_mode == 'sum':
                output_dict[key] = scatter_add(prop, index, dim=0) if index is not None else torch.sum(prop, dim=0)
            elif aggregation_mode == 'mean':
                output_dict[key] = scatter_mean(prop, index, dim=0) if index is not None else torch.mean(prop, dim=0)
            elif aggregation_mode == 'none':
                output_dict[key] = prop

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

    def _make_readout(self) -> nn.Sequential:
        return nn.Sequential(*[
            Dense(self._n_neurons[i], self._n_neurons[i + 1], self._acts[i], use_e3nn=self.use_e3nn)
            for i in range(self.n_hidden_layers + 1)
        ])

    def _get_domain(self, data: properties.Type) -> Optional[str]:
        return None

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

        super().__init__(
            in_features=o3.Irreps(str(self.hidden_irreps[0])),
            n_hidden=self.MLP_irreps,
            use_e3nn=True,
            *args,
            **kwargs,
        )
        self.num_interactions = num_interactions

        self.readouts = nn.ModuleList()
        self.in_features_list = []
        for _ in range(num_interactions - 1):
            self.in_features_list.append(self.hidden_irreps.dim)
            self.readouts.append(Dense(self.hidden_irreps, self.out_features, activation=None, use_e3nn=True))

        self.readouts.append(self.readout_mlp)
        self.in_features_list.append(self.hidden_irreps[0].dim)

    def _compute(self, input: torch.Tensor, index: Optional[torch.Tensor] = None, domain: Optional[str] = None) -> properties.Type:
        # split node features to list then calculate contributions from different parts
        node_feat_list = torch.split(input, self.in_features_list, dim=-1)

        readouts = self.readouts
        out_list = []
        for readout, node_feat in zip(readouts, node_feat_list):
            out_list.append(readout(node_feat))
        out = torch.sum(torch.stack(out_list, dim=0), dim=0)
        return out


class MultiDomainAtomwiseNN(nn.Module):
    """
    Domain-aware wrapper holding one AtomwiseNN-like module per domain.
    Routes atoms by properties.domain_atom and merges outputs with index_select/scatter.
    """

    def __init__(
        self,
        domains: Optional[List[Union[str, int]]] = None,
        readout_cls: Union[type, Callable] = AtomwiseNN,
        heads_by_domain: Optional[Dict[Union[str, int], List[Union[HeadConfig, Dict, str]]]] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.domains = [str(d) for d in domains] if domains else ["0"]

        base_kwargs = dict(kwargs)
        base_heads_by_domain = {}
        if heads_by_domain:
            base_heads_by_domain = {str(k): v for k, v in heads_by_domain.items()}
        base_kwargs.pop("domains", None)
        base_kwargs.pop("heads_by_domain", None)

        self.domain_modules = nn.ModuleDict()
        for dom in self.domains:
            domain_kwargs = dict(base_kwargs)
            heads_for_domain = base_heads_by_domain.get(dom, domain_kwargs.get("heads"))
            if heads_for_domain is not None:
                resolved = resolve_heads(heads_for_domain)
                domain_kwargs["heads"] = heads_for_domain
                domain_kwargs["out_features"] = sum(int(h.dim) for h in resolved)
            self.domain_modules[dom] = readout_cls(*args, **domain_kwargs)

    def _get_domain(self, data: properties.Type) -> str:
        if properties.domain not in data:
            return self.domains[0]
        dom = data[properties.domain]
        if torch.is_tensor(dom):
            if dom.numel() == 0:
                return self.domains[0]
            dom = dom.view(-1)[0].item()
        dom = str(dom)
        if dom in self.domain_modules:
            return dom
        return self.domains[0]

    def _ensure_image_index(self, data: properties.Type) -> None:
        if properties.image_idx not in data:
            data[properties.image_idx] = torch.zeros(
                data[properties.n_atoms].item(),
                dtype=data[properties.edge_idx].dtype,
                device=data[properties.edge_idx].device,
            )

    def _split_domain_labels(
        self, data: properties.Type, n_atoms: int
    ) -> Optional[torch.Tensor]:
        atom_domain = None
        if properties.domain_atom in data:
            atom_domain = data[properties.domain_atom]
        elif properties.domain in data:
            dom = data[properties.domain]
            if torch.is_tensor(dom) and dom.numel() == n_atoms:
                atom_domain = dom
        return atom_domain

    def _per_atom_output_keys(self, module: AtomwiseNN) -> set:
        keys = set()
        for i, key in enumerate(module.model_outputs):
            if module.aggregation_modes[i] == "none":
                keys.add(key)
            if module.per_atom_flags[i]:
                keys.add(module.per_atom_keys[i])
        return keys

    def _compute_outputs_subset(
        self,
        module: AtomwiseNN,
        node_feat: torch.Tensor,
        index: torch.Tensor,
        atom_idx: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        node_feat_sub = node_feat.index_select(0, atom_idx)
        index_sub = index.index_select(0, atom_idx)
        out = module._compute(node_feat_sub)
        return module._parse_outputs(out, index_sub), index_sub

    def _init_combined(
        self,
        output: Dict[str, torch.Tensor],
        module: AtomwiseNN,
        n_atoms: int,
        n_graphs: int,
    ) -> Dict[str, torch.Tensor]:
        per_atom_keys = self._per_atom_output_keys(module)
        combined = {}
        for key, val in output.items():
            if key in per_atom_keys:
                shape = (n_atoms,) + val.shape[1:]
            else:
                shape = (n_graphs,) + val.shape[1:]
            combined[key] = val.new_zeros(shape)
        return combined

    def _scatter_outputs(
        self,
        combined: Dict[str, torch.Tensor],
        output: Dict[str, torch.Tensor],
        module: AtomwiseNN,
        atom_idx: torch.Tensor,
    ) -> None:
        per_atom_keys = self._per_atom_output_keys(module)
        for key, val in output.items():
            if key in per_atom_keys:
                combined[key].index_copy_(0, atom_idx, val)
            else:
                combined[key][: val.shape[0]].add_(val)

    def forward(self, data: properties.Type) -> properties.Type:
        self._ensure_image_index(data)
        node_feat = data[properties.node_feat]
        index = data[properties.image_idx]
        n_atoms = node_feat.shape[0]
        n_graphs = int(index.max().item()) + 1 if index.numel() > 0 else 0

        atom_domain = self._split_domain_labels(data, n_atoms)
        if atom_domain is None:
            dom = self._get_domain(data)
            return self.domain_modules[dom](data)

        atom_domain = atom_domain.to(torch.long)
        combined = None
        matched = torch.zeros((n_atoms,), dtype=torch.bool, device=node_feat.device)

        for dom, module in self.domain_modules.items():
            if not str(dom).isdigit():
                continue
            dom_id = int(dom)
            mask = atom_domain == dom_id
            if not torch.any(mask):
                continue
            atom_idx = mask.nonzero(as_tuple=False).view(-1)
            output, _ = self._compute_outputs_subset(module, node_feat, index, atom_idx)
            if combined is None:
                combined = self._init_combined(output, module, n_atoms, n_graphs)
            self._scatter_outputs(combined, output, module, atom_idx)
            matched[atom_idx] = True

        if combined is None:
            dom = self.domains[0]
            return self.domain_modules[dom](data)

        remaining = ~matched
        if torch.any(remaining):
            atom_idx = remaining.nonzero(as_tuple=False).view(-1)
            fallback_dom = self.domains[0]
            output, _ = self._compute_outputs_subset(self.domain_modules[fallback_dom], node_feat, index, atom_idx)
            self._scatter_outputs(combined, output, self.domain_modules[fallback_dom], atom_idx)

        data.update(combined)
        return data


class MultiDomainMACEAtomwiseNN(MultiDomainAtomwiseNN):
    """
    MultiDomain wrapper specialized for MACEAtomwiseNN.
    """

    def __init__(self, domains: Optional[List[Union[str, int]]] = None, *args, **kwargs):
        super().__init__(domains=domains, readout_cls=MACEAtomwiseNN, *args, **kwargs)
