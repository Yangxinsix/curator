import torch
from torch import nn
from torch.nn import functional as F
from typing import Final, List, Union, Optional, Callable, Dict, Tuple, Literal
from e3nn import o3
from e3nn.nn import Activation
from curator.data.properties import activation_fn, HeadConfig, resolve_heads
from curator.data import properties
from ._ops import Linear, ScalarLinear
import warnings
try:
    from torch_scatter import scatter_add, scatter_mean
except ImportError:
    from curator.utils import scatter_add, scatter_mean

AggregationMode = Literal["sum", "mean", "none"]

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
            self.linear = ScalarLinear(in_features, out_features, *args, **kwargs)
            self.activation = activation or nn.Identity()

    def forward(self, input: torch.Tensor):
        y = self.linear(input)
        y = self.activation(y)
        return y

class AtomwiseNN(nn.Module):
    separate_heads: Final[bool]

    def __init__(
        self,
        in_features: Union[int, o3.Irreps, str],
        out_features: Union[int, o3.Irreps, str] = 1,
        n_hidden: Union[List[int], List[o3.Irreps], int, o3.Irreps, None] = None,
        n_hidden_layers: int = 1,
        use_e3nn: bool = False,
        activation: Union[Callable, nn.Module, str, List[Callable], List[nn.Module], List[str]] = 'silu',
        heads: Optional[List[Union[HeadConfig, Dict, str]]] = None,
        separate_heads: bool = False,
        linear_initializer: Optional[Callable[[nn.Module], None]] = None,
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
        self.separate_heads = bool(separate_heads)

        # Prepare and store output specifications using HeadConfig.
        self.heads: List[HeadConfig] = resolve_heads(heads) if heads is not None else []
        if not self.heads:
            self.heads = [HeadConfig(key="energy", is_atomwise=True, reduction="sum", atomwise_key="atomic_energy")]
        if self.separate_heads and not self.use_e3nn:
            self.out_features = sum(int(h.dim) for h in self.heads)

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
        self._hidden_neurons = list(n_neurons[1:-1])
        self._acts = acts

        if self.separate_heads:
            self.shared_mlp, self.head_modules, self.shared_out_features = self._make_separate_readout()
            self.readout_mlp = self.shared_mlp
        else:
            self.shared_mlp = nn.Identity()
            self.head_modules = nn.ModuleDict()
            self.shared_out_features = self.in_features
            self.readout_mlp = self._make_readout()

        if linear_initializer is not None:
            for module in self.modules():
                if isinstance(module, Dense):
                    linear_initializer(module.linear)

        self._n_out = sum(int(h.dim) for h in self.heads)

        self.model_outputs = [h.key for h in self.heads]
        self.per_atom_flags = [bool(h.write_atomwise) for h in self.heads]
        self.aggregation_modes: List[AggregationMode] = [
            (h.reduction if h.reduction is not None else "none")
            for h in self.heads
        ]
        self.per_atom_keys = [(h.atomwise_key or (h.key + "_pa")) for h in self.heads]
        self.split_size = [int(h.dim) for h in self.heads]

        assert sum(self.split_size) == self._n_out, "Output feature dimensions do not match split sizes!"

    def _compute(self, input: torch.Tensor) -> torch.Tensor:
        if self.separate_heads:
            shared = self.shared_mlp(input)
            outputs = [self.head_modules[h.key](shared) for h in self.heads]
            return torch.cat(outputs, dim=-1)
        return self.readout_mlp(input)

    def _parse_outputs(self, out: torch.Tensor, index: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        out = out.split(self.split_size, dim=1)
        output_dict: Dict[str, torch.Tensor] = {}

        # Iterate over tensor-friendly lists for TorchScript compatibility.
        for i in range(len(self.model_outputs)):
            # Avoid inplace/view conflicts during TorchScript autograd in deploy models.
            prop = out[i].squeeze(-1).clone()
            key = self.model_outputs[i]
            per_atom = self.per_atom_flags[i]
            aggregation_mode = self.aggregation_modes[i]
            per_atom_key = self.per_atom_keys[i]

            if per_atom:
                output_dict[per_atom_key] = prop

            if aggregation_mode == 'sum':
                if index is not None and index.numel() > 0 and index.numel() == prop.shape[0]:
                    output_dict[key] = scatter_add(prop, index, dim=0)
                else:
                    output_dict[key] = torch.sum(prop, dim=0)
            elif aggregation_mode == 'mean':
                if index is not None and index.numel() > 0 and index.numel() == prop.shape[0]:
                    output_dict[key] = scatter_mean(prop, index, dim=0)
                else:
                    output_dict[key] = torch.mean(prop, dim=0) if prop.numel() > 0 else torch.sum(prop, dim=0)
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
        return self._make_mlp(self.in_features, self.out_features)

    def _make_mlp(
        self,
        in_features: Union[int, o3.Irreps],
        out_features: Union[int, o3.Irreps],
        hidden_neurons: Optional[List[Union[int, o3.Irreps]]] = None,
    ) -> nn.Sequential:
        hidden = list(self._hidden_neurons if hidden_neurons is None else hidden_neurons)
        dims: List[Union[int, o3.Irreps]] = [in_features] + hidden + [out_features]
        modules = [
            Dense(dims[i], dims[i + 1], self._acts[i] if i < len(hidden) else None, use_e3nn=self.use_e3nn)
            for i in range(len(dims) - 1)
        ]
        return nn.Sequential(*modules)

    def _resolve_head_out_features(self, head: HeadConfig) -> Union[int, o3.Irreps]:
        if not self.use_e3nn:
            return int(head.dim)

        irreps_out = getattr(head, "irreps_out", None)
        if irreps_out is None:
            return o3.Irreps(f"{int(head.dim)}x0e")
        if isinstance(irreps_out, str):
            irreps_out = o3.Irreps(irreps_out)
        if irreps_out.dim != int(head.dim):
            raise ValueError(
                f"Head '{head.key}' declares dim={head.dim} but irreps_out={irreps_out} has dim={irreps_out.dim}."
            )
        return irreps_out

    def _supports_projection(
        self,
        in_features: Union[int, o3.Irreps],
        out_features: Union[int, o3.Irreps],
    ) -> bool:
        if not self.use_e3nn or not isinstance(in_features, o3.Irreps) or not isinstance(out_features, o3.Irreps):
            return True
        input_irreps = {ir for _, ir in in_features}
        return all(ir in input_irreps for _, ir in out_features)

    def _make_head_projection(
        self,
        in_features: Union[int, o3.Irreps],
        head: HeadConfig,
    ) -> nn.Module:
        head_out = self._resolve_head_out_features(head)
        if not self._supports_projection(in_features, head_out):
            raise ValueError(
                f"Head '{head.key}' with output irreps {head_out} is incompatible with readout input {in_features}."
            )
        return Dense(in_features, head_out, activation=None, use_e3nn=self.use_e3nn)

    def _make_separate_readout(self) -> Tuple[nn.Module, nn.ModuleDict, Union[int, o3.Irreps]]:
        if len(self._hidden_neurons) == 0:
            shared = nn.Identity()
            shared_out = self.in_features
        else:
            shared_out = self._hidden_neurons[-1]
            shared = self._make_mlp(
                self.in_features,
                shared_out,
                hidden_neurons=self._hidden_neurons[:-1],
            )

        head_modules = nn.ModuleDict()
        for head in self.heads:
            head_modules[head.key] = self._make_head_projection(shared_out, head)
        return shared, head_modules, shared_out

    def _get_domain(self, data: properties.Type) -> Optional[str]:
        return None

class MACEAtomwiseNN(AtomwiseNN):
    """Atomwise feed-forward neural networks for MACE

    Args:
        num_layers: number of message-passing layers in MACE
        hidden_irreps: hidden_irreps in MACE
    """
    def _make_separate_readout(self) -> Tuple[nn.Module, nn.ModuleDict, Union[int, o3.Irreps]]:
        # MACE handles per-head projections in its own layerwise readouts. Keep a
        # shared scalar trunk alias for feature extraction / legacy access only.
        if len(self._hidden_neurons) == 0:
            shared = nn.Identity()
            shared_out = self.in_features
        else:
            shared_out = self._hidden_neurons[-1]
            shared = self._make_mlp(
                self.in_features,
                shared_out,
                hidden_neurons=self._hidden_neurons[:-1],
            )
        return shared, nn.ModuleDict(), shared_out

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

        self.in_features_list = []
        self.in_features_list.append(self.hidden_irreps[0].dim)
        if self.separate_heads:
            self.readouts_by_head = nn.ModuleDict()
            self.final_readouts = nn.ModuleDict()
            final_scalar_irreps = o3.Irreps(str(self.hidden_irreps[0]))
            for head in self.heads:
                per_layer = nn.ModuleList()
                head_out = self._resolve_head_out_features(head)
                supports_hidden = self._supports_projection(self.hidden_irreps, head_out)
                if not supports_hidden and num_interactions > 1:
                    raise ValueError(
                        f"Head '{head.key}' with output irreps {head_out} is incompatible with MACE hidden irreps {self.hidden_irreps}."
                    )
                for _ in range(num_interactions - 1):
                    self.in_features_list.insert(-1, self.hidden_irreps.dim)
                    per_layer.append(Dense(self.hidden_irreps, head_out, activation=None, use_e3nn=True))
                self.readouts_by_head[head.key] = per_layer
                if self._supports_projection(final_scalar_irreps, head_out):
                    self.final_readouts[head.key] = self._make_mlp(final_scalar_irreps, head_out)
                elif num_interactions == 1:
                    raise ValueError(
                        f"Head '{head.key}' with output irreps {head_out} cannot be predicted from the final scalar-only MACE layer."
                    )
            primary_head = self.heads[0].key
            self.readouts = nn.ModuleList(list(self.readouts_by_head[primary_head]))
            if primary_head in self.final_readouts:
                self.readouts.append(self.final_readouts[primary_head])
            self.in_features_list = [self.hidden_irreps.dim for _ in range(num_interactions - 1)] + [self.hidden_irreps[0].dim]
        else:
            self.readouts = nn.ModuleList()
            for _ in range(num_interactions - 1):
                self.in_features_list.insert(-1, self.hidden_irreps.dim)
                self.readouts.append(Dense(self.hidden_irreps, self.out_features, activation=None, use_e3nn=True))
            self.readouts.append(self.readout_mlp)
        invariant_dim = self._invariant_dim(self.hidden_irreps)
        self.invariant_features_list = [invariant_dim for _ in range(num_interactions)]

    def _compute(self, input: torch.Tensor, index: Optional[torch.Tensor] = None, domain: Optional[str] = None) -> properties.Type:
        # split node features to list then calculate contributions from different parts
        node_feat_list = torch.split(input, self.in_features_list, dim=-1)

        if not self.separate_heads:
            out_list = []
            for readout, node_feat in zip(self.readouts, node_feat_list):
                out_list.append(readout(node_feat))
            return torch.sum(torch.stack(out_list, dim=0), dim=0)

        outputs = []
        scalar_node_feat = node_feat_list[-1]
        for head in self.heads:
            contribs = [
                readout(node_feat)
                for readout, node_feat in zip(self.readouts_by_head[head.key], node_feat_list[:-1])
            ]
            if head.key in self.final_readouts:
                contribs.append(self.final_readouts[head.key](scalar_node_feat))
            if not contribs:
                raise ValueError(f"Head '{head.key}' has no compatible MACE readout contributions.")
            outputs.append(torch.sum(torch.stack(contribs, dim=0), dim=0))
        return torch.cat(outputs, dim=-1)

    @staticmethod
    def _invariant_dim(irreps: o3.Irreps) -> int:
        return int(sum(mul * ir.dim for mul, ir in irreps if ir.l == 0 and ir.p == 1))


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
        atom_domain: Optional[torch.Tensor] = None
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
                prefix = combined[key][: val.shape[0]]
                combined[key][: val.shape[0]] = prefix + val

    def _call_domain_module(self, dom: str, data: properties.Type) -> properties.Type:
        for key, module in self.domain_modules.items():
            if key == dom:
                return module(data)
        for _, module in self.domain_modules.items():
            return module(data)
        return data

    def forward(self, data: properties.Type) -> properties.Type:
        self._ensure_image_index(data)
        if len(self.domain_modules) == 1:
            for _, only_module in self.domain_modules.items():
                return only_module(data)

        node_feat = data[properties.node_feat]
        index = data[properties.image_idx]
        n_atoms = node_feat.shape[0]
        n_graphs = int(index.max().item()) + 1 if index.numel() > 0 else 0

        atom_domain = self._split_domain_labels(data, n_atoms)
        if atom_domain is None:
            dom = self._get_domain(data)
            return self._call_domain_module(dom, data)

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
            return self._call_domain_module(dom, data)

        remaining = ~matched
        if torch.any(remaining):
            atom_idx = remaining.nonzero(as_tuple=False).view(-1)
            fallback_dom = self.domains[0]
            for key, module in self.domain_modules.items():
                if key == fallback_dom:
                    output, _ = self._compute_outputs_subset(module, node_feat, index, atom_idx)
                    self._scatter_outputs(combined, output, module, atom_idx)
                    break

        data.update(combined)
        return data


class MultiDomainMACEAtomwiseNN(MultiDomainAtomwiseNN):
    """
    MultiDomain wrapper specialized for MACEAtomwiseNN.
    """

    def __init__(self, domains: Optional[List[Union[str, int]]] = None, *args, **kwargs):
        super().__init__(domains=domains, readout_cls=MACEAtomwiseNN, *args, **kwargs)
