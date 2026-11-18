"""The trained q-EFA energy with charge supplied as an independent state."""

import copy

import torch
from e3nn import o3

from curator.data import properties
from .adapter import EFAAdapter
from .q_conditioned import QConditionedEFAAdapter, _QConditioning


class QConditionedEFAEnergy(_QConditioning, EFAAdapter):
    """EFA correction U(h, R, q) for ``VariationalEnergyOutput``.

    q has shape [N, 1] and units e. This functional owns the trained K/V gates
    and readout, while a separate proposal owns charge prediction. The existing
    adapter supplies geometry handling, attention and energy scaling unchanged.
    The variational framework adds the baseline and boundary energy exactly once.
    """

    def bind(self, feature_spec, *, components=1, state_kind="scalar", **unused):
        if components != 1 or state_kind != "scalar":
            raise ValueError("q-EFA energy requires one scalar charge per atom")
        if self.feature_spec is not None:
            return super().bind(feature_spec)
        super().bind(feature_spec)
        self._bind_conditioning(o3.Irreps(feature_spec.irreps))
        for module in (self.key_residual, self.value_residual, self.q_readout):
            module.to(device=self._dtype_anchor.device, dtype=self._dtype_anchor.dtype)
        return self

    @classmethod
    def from_adapter(cls, adapter):
        """Copy every energy parameter, excluding the old latent-charge head."""
        if type(adapter) is not QConditionedEFAAdapter or adapter.q_mode != "learned":
            raise TypeError("expected a learned QConditionedEFAAdapter")
        if adapter.feature_spec is None:
            raise ValueError("bind the q-EFA adapter before converting it")
        reference = next(adapter.parameters())
        energy = cls(
            hidden_features=adapter.hidden_features,
            num_directions=adapter.attention.num_directions,
            max_frequency=adapter.attention.max_frequency,
            max_length=adapter.attention.max_length,
            energy_scale=adapter.energy_scale,
            use_equivariant_queries=adapter.use_equivariant_queries,
        ).to(device=reference.device, dtype=reference.dtype)
        energy.bind(copy.deepcopy(adapter.feature_spec))
        for name in ("attention", "query", "key", "value", "readout",
                     "key_residual", "value_residual", "q_readout"):
            setattr(energy, name, copy.deepcopy(getattr(adapter, name)))
        energy.train(adapter.training)
        return energy

    def _project_features(self, features, data):
        return self._conditioned_projections(features, data["efa_charge"])

    def _readout_features(self, update, data, mask):
        return self._conditioned_readout(update, data["efa_charge"][mask])

    def forward(self, context, state):
        from curator.model.features import IrrepsFeatureSpec, ScalarVectorFeatureSpec

        if self.feature_spec is None:
            raise RuntimeError("bind q-EFA energy before evaluation")
        features = context["features"]
        if state.shape != (len(features), 1):
            raise ValueError("q-EFA state must have shape [N, 1]")
        data = {
            properties.positions: context["positions"],
            properties.n_atoms: torch.tensor([len(features)], device=features.device),
            properties.pbc: context["pbc"], properties.cell: context["cell"],
            properties.energy: features.new_zeros(1), "efa_charge": state,
        }
        # Reconstruct only the declared feature view, retaining the full tensor
        # and its graph. Both supported specifications are lossless packings.
        if isinstance(self.feature_spec, IrrepsFeatureSpec):
            data[self.feature_spec.key] = features
        elif isinstance(self.feature_spec, ScalarVectorFeatureSpec):
            spec = self.feature_spec
            data[spec.scalar_key] = features[:, :spec.scalar_channels]
            if spec.vector_channels:
                data[spec.vector_key] = features[:, spec.scalar_channels:].reshape(
                    len(features), spec.vector_channels, 3).transpose(1, 2)
        else:
            raise TypeError("unsupported q-EFA feature specification")
        return super().forward(data)["efa_energy"].squeeze(0)

    def get_extra_state(self):
        return {"schema": "explicit_q_efa_energy_v1", "state_unit": "e",
                "energy_unit": "eV", "energy_scale": self.energy_scale}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("explicit q-EFA checkpoint contract differs")


__all__ = ["QConditionedEFAEnergy"]
