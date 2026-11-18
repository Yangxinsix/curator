"""Full-feature latent-q residual conditioning of the legacy EFA energy."""

import copy
import math

import torch
from torch import nn
from e3nn import o3

from .adapter import EFAAdapter


class _QConditioning:
    """Shared charge-conditioned maps for latent and explicit-state energies."""

    def _bind_conditioning(self, irreps):
        self.key_residual = o3.Linear(irreps, self.query_irreps, biases=True)
        self.value_residual = o3.Linear(irreps, f"{self.hidden_features}x0e", biases=True)
        self.q_readout = nn.Linear(1, self.hidden_features, bias=False)
        for module in (self.key_residual, self.value_residual):
            if module.bias.numel():
                nn.init.normal_(module.bias, std=0.01)

    def _conditioned_projections(self, features, q):
        query = self._angular_channels(self.query(features))
        key = self._angular_channels(self.key(features) + q * self.key_residual(features))
        value = self.value(features) + q * self.value_residual(features)
        return query, key, value

    def _conditioned_readout(self, update, q):
        preactivation = self.readout[0](update) + self.q_readout(q)
        return self.readout[2](self.readout[1](preactivation))


class QConditionedEFAAdapter(_QConditioning, EFAAdapter):
    """Keep legacy EFA and add q-gated K/V bases and a short q readout path.

    Q = Q0(h), K = K0(h) + q B_K(h), V = V0(h) + q B_V(h).
    The existing readout becomes w2 SiLU(W1 G + b + wq q). All projections
    independently read the full backbone feature tensor. There is no shared
    compression, proposal, charge constraint, or charge-label supervision.

    q is a dimensionless latent scalar, not a calibrated atomic charge. Its
    head sees all 0e channels and smooth RMS norms of the other irreps. The
    Tanh output bounds q in (-1, 1), but does not impose a total-charge bound.
    Coordinate derivatives through h(R) and q(R) are never detached.

    New modules start nonzero. For fine-tuning, use ``from_legacy`` and scale
    their initial parameters using measured energy AND force perturbations.
    Scaling initial parameters does not leave a small fixed multiplier on
    their later gradients. ``q_mode='zero'`` bypasses all added paths exactly;
    it does not alter requires_grad flags, so posthoc mode changes are safe.

    Boundary support and finite spherical-quadrature accuracy are inherited
    from EFAAdapter. Bind before constructing an optimizer.
    """

    def __init__(self, feature_spec=None, q_mode="learned", q_hidden_features=64,
                 norm_epsilon=1e-8, hidden_features=32, num_directions=110,
                 max_frequency=2 * math.pi, max_length=32.0, energy_scale=0.01,
                 use_equivariant_queries=True):
        if q_mode not in ("learned", "zero"):
            raise ValueError("q_mode must be 'learned' or 'zero'")
        if not isinstance(q_hidden_features, int) or q_hidden_features < 1:
            raise ValueError("q_hidden_features must be a positive integer")
        if not math.isfinite(norm_epsilon) or norm_epsilon <= 0:
            raise ValueError("norm_epsilon must be finite and positive")
        super().__init__(hidden_features=hidden_features, num_directions=num_directions,
                         max_frequency=max_frequency, max_length=max_length,
                         energy_scale=energy_scale,
                         use_equivariant_queries=use_equivariant_queries)
        self.q_mode = q_mode
        self.q_hidden_features = q_hidden_features
        self.norm_epsilon = float(norm_epsilon)
        self.q_head = self.key_residual = self.value_residual = self.q_readout = None
        self.model_outputs.append("efa_latent_q")
        if feature_spec is not None:
            self.bind(feature_spec)

    def bind(self, feature_spec):
        if self.feature_spec is not None:
            return super().bind(feature_spec)
        super().bind(feature_spec)
        irreps = o3.Irreps(feature_spec.irreps)
        self.q_head = nn.Sequential(
            nn.Linear(irreps.num_irreps, self.q_hidden_features),
            nn.SiLU(), nn.Linear(self.q_hidden_features, 1), nn.Tanh(),
        )
        # e3nn initializes scalar biases to zero. Keep every new parameter
        # tensor nonzero so no path depends on a zero-initialization warmup.
        self._bind_conditioning(irreps)
        for module in self.new_modules():
            module.to(device=self._dtype_anchor.device, dtype=self._dtype_anchor.dtype)
        return self

    def new_modules(self):
        """Added modules, useful for explicit optimizer/freeze configuration."""
        return (self.q_head, self.key_residual, self.value_residual, self.q_readout)

    @classmethod
    def from_legacy(cls, legacy, **options):
        """Copy a bound legacy head without sharing or changing its parameters."""
        if not isinstance(legacy, EFAAdapter) or isinstance(legacy, cls):
            raise TypeError("from_legacy expects an unconditioned EFAAdapter")
        if legacy.feature_spec is None:
            raise ValueError("bind the legacy EFA before converting it")
        allowed = {"q_mode", "q_hidden_features", "norm_epsilon"}
        unexpected = set(options) - allowed
        if unexpected:
            raise TypeError(f"cannot override legacy settings: {sorted(unexpected)}")
        reference = next(legacy.parameters())
        adapter = cls(
            hidden_features=legacy.hidden_features,
            num_directions=legacy.attention.num_directions,
            max_frequency=legacy.attention.max_frequency,
            max_length=legacy.attention.max_length,
            energy_scale=legacy.energy_scale,
            use_equivariant_queries=legacy.use_equivariant_queries, **options,
        ).to(device=reference.device, dtype=reference.dtype)
        adapter.bind(copy.deepcopy(legacy.feature_spec))
        for name in ("attention", "query", "key", "value", "readout"):
            setattr(adapter, name, copy.deepcopy(getattr(legacy, name)))
        adapter.train(legacy.training)
        return adapter

    def invariant_features(self, features):
        """Return one invariant per irrep channel with smooth derivatives."""
        if self.feature_spec is None:
            raise RuntimeError("bind QConditionedEFAAdapter before using it")
        irreps = o3.Irreps(self.feature_spec.irreps)
        if features.ndim != 2 or features.shape[1] != irreps.dim:
            raise ValueError("features do not match the bound irreps")
        blocks = []
        for (multiplicity, irrep), part in zip(irreps, irreps.slices()):
            block = features[:, part].reshape(len(features), multiplicity, irrep.dim)
            if irrep == o3.Irrep("0e"):
                blocks.append(block.squeeze(-1))
            else:
                blocks.append((block.square().mean(-1) + self.norm_epsilon).sqrt())
        return torch.cat(blocks, dim=-1)

    def predict_charge(self, features):
        """Return differentiable latent q in atom order, shape [N, 1]."""
        if self.q_mode not in ("learned", "zero"):
            raise ValueError("q_mode must be 'learned' or 'zero'")
        invariants = self.invariant_features(features)
        if self.q_mode == "zero":
            return features.new_zeros((len(features), 1))
        return self.q_head(invariants)

    def _project_features(self, features, data):
        q = self.predict_charge(features)
        data["efa_latent_q"] = q
        if self.q_mode == "zero":
            return super()._project_features(features, data)
        return self._conditioned_projections(features, q)

    def _readout_features(self, update, data, mask):
        if self.q_mode == "zero":
            return super()._readout_features(update, data, mask)
        return self._conditioned_readout(update, data["efa_latent_q"][mask])

    @torch.no_grad()
    def scale_new_projections_(self, factor, paths=("key", "value", "readout")):
        """Scale initial added-map parameters in place; never scale the q head."""
        if self.feature_spec is None:
            raise RuntimeError("bind QConditionedEFAAdapter before calibration")
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("initialization factor must be finite and positive")
        modules = {"key": self.key_residual, "value": self.value_residual,
                   "readout": self.q_readout}
        if isinstance(paths, str):
            paths = (paths,)
        paths = tuple(paths)
        if not paths or len(set(paths)) != len(paths) or any(p not in modules for p in paths):
            raise ValueError("paths must be distinct members of key, value, readout")
        for path in paths:
            for parameter in modules[path].parameters():
                parameter.mul_(factor)
        return self

    def get_extra_state(self):
        return {"schema": "q_conditioned_efa_adapter_v1", "q_mode": self.q_mode,
                "q_hidden_features": self.q_hidden_features,
                "norm_epsilon": self.norm_epsilon,
                "q_semantics": "bounded_unconstrained_latent_scalar"}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("q-conditioned EFA checkpoint contract differs")


__all__ = ["QConditionedEFAAdapter"]
