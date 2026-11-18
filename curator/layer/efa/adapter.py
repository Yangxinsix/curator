"""Legacy MACE + EFA additive-energy adapter.

Keep the original parameter names and forward operations for existing models,
state dictionaries and serialized full-model checkpoints.
"""
import math

import torch
from torch import nn
from e3nn import o3

from curator.data import properties
from .core import EuclideanFastAttention
from .geometry import structure_cells


class EFAAdapter(nn.Module):
    """Add an EFA energy correction to an existing Curator representation.

    Bind before constructing the optimizer. Insert after physical-unit energy
    rescaling and before forces computed from positions. FFF uses spherical
    integration; TTT uses the authors' raw-lattice encoding. Partial periodicity
    is unsupported. Raw-lattice phases do not guarantee invariance to changing
    individual atom images. No charge labels/state are used.
    """

    repr_callback = True

    def __init__(self, feature_spec=None, hidden_features=32, num_directions=110,
                 max_frequency=2 * math.pi, max_length=32.0, energy_scale=0.01,
                 use_equivariant_queries=True):
        super().__init__()
        if not math.isfinite(energy_scale) or energy_scale <= 0:
            raise ValueError("energy_scale must be finite and positive")
        self.attention = EuclideanFastAttention(hidden_features, num_directions,
                                               max_frequency, max_length)
        self.hidden_features = hidden_features
        self.energy_scale = float(energy_scale)
        self.use_equivariant_queries = bool(use_equivariant_queries)
        self.register_buffer("_dtype_anchor", torch.empty(0))
        self.feature_spec = None
        self.query = self.key = self.value = self.readout = None
        self.model_outputs = [properties.energy, "efa_energy"]
        if feature_spec is not None:
            self.bind(feature_spec)

    def bind(self, feature_spec):
        if self.feature_spec is not None:
            if feature_spec != self.feature_spec:
                raise ValueError("EFAAdapter is already bound to a different feature specification")
            return self
        irreps = o3.Irreps(feature_spec.irreps)
        if o3.Irrep("0e") not in [ir for _, ir in irreps]:
            raise ValueError("EFA energy adapter needs invariant 0e backbone features")
        types = sorted({ir for _, ir in irreps
                        if self.use_equivariant_queries or ir == o3.Irrep("0e")})
        self.query_irreps = o3.Irreps([(self.hidden_features, ir) for ir in types])
        self.feature_spec = feature_spec
        self.query = o3.Linear(irreps, self.query_irreps, biases=True)
        self.key = o3.Linear(irreps, self.query_irreps, biases=True)
        self.value = o3.Linear(irreps, f"{self.hidden_features}x0e", biases=True)
        self.readout = nn.Sequential(nn.Linear(self.hidden_features, self.hidden_features),
                                     nn.SiLU(), nn.Linear(self.hidden_features, 1, bias=False))
        # Binding may happen after .to() but always before optimizer creation.
        self.query.to(device=self._dtype_anchor.device, dtype=self._dtype_anchor.dtype)
        self.key.to(device=self._dtype_anchor.device, dtype=self._dtype_anchor.dtype)
        self.value.to(device=self._dtype_anchor.device, dtype=self._dtype_anchor.dtype)
        self.readout.to(device=self._dtype_anchor.device, dtype=self._dtype_anchor.dtype)
        return self

    def register_repr_callback(self, model):
        self.bind(model.representation.response_feature_spec())
        reference = next(model.representation.parameters(), None)
        if reference is not None:
            self.to(device=reference.device, dtype=reference.dtype)

    def _angular_channels(self, features):
        return torch.cat([features[:, sl].reshape(len(features), self.hidden_features, ir.dim)
                          .transpose(1, 2)
                          for (_, ir), sl in zip(self.query_irreps, self.query_irreps.slices())], dim=1)

    def _project_features(self, features, data):
        """Project full backbone features before positional encoding."""
        return (self._angular_channels(self.query(features)),
                self._angular_channels(self.key(features)), self.value(features))

    def _readout_features(self, update, data, mask):
        """Map the global update to per-atom energy contributions."""
        return self.readout(update)

    def forward(self, data):
        if self.feature_spec is None:
            raise RuntimeError("bind EFAAdapter to response_feature_spec before training")
        positions = data[properties.positions]
        features = self.feature_spec.flatten(data)
        if len(features) != len(positions):
            raise ValueError("feature and position atom counts differ")
        counts = data[properties.n_atoms].reshape(-1)
        if bool((counts <= 0).any()) or int(counts.sum()) != len(positions):
            raise ValueError("n_atoms must describe nonempty structures")
        cells = structure_cells(data, len(counts))
        batch = data.get(properties.image_idx)
        if batch is None:
            if len(counts) != 1:
                raise ValueError("batched EFA requires image_idx")
            batch = torch.zeros(len(positions), device=positions.device, dtype=torch.long)
        if batch.shape != (len(positions),) or bool(((batch < 0) | (batch >= len(counts))).any()):
            raise ValueError("invalid EFA image_idx")
        query, key, value = self._project_features(features, data)
        corrections = []
        for graph, count in enumerate(counts):
            mask = batch == graph
            if int(mask.sum()) != int(count):
                raise ValueError("image_idx and n_atoms disagree")
            update = self.attention(query[mask], key[mask], value[mask], positions[mask],
                                    cell=cells[graph])
            corrections.append(self._readout_features(update, data, mask).sum() * self.energy_scale)
        correction = torch.stack(corrections)
        if data[properties.energy].numel() != len(counts):
            raise ValueError("base energy must contain one value per structure")
        data["efa_energy"] = correction
        data[properties.energy] = data[properties.energy] + correction.reshape_as(data[properties.energy])
        return data

