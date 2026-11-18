"""Separate the charge driving force from the curvature of convex q-EFA."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .convex import ConvexQConditionedEFAEnergy


class TwoScaleQConditionedEFAEnergy(ConvexQConditionedEFAEnergy):
    """Keep geometry EFA and independently scale linear and convex q terms.

    For geometry-only offset b and affine charge response t=A(R)q, define
    L=sum(c*sigmoid(b)*t) and D=sum(c*(softplus(b+t)-softplus(b)-sigmoid(b)*t)).
    The charge energy is beta*L + alpha*D + hardness*||q||²/2. Positive alpha
    and c preserve H_q >= hardness*I for any signed beta. Both scales default
    to energy_scale, recovering the original convex q-EFA energy.

    Geometry and charge share one attention call; only Values depend on q.
    The independent geometry readout and all backbone features are retained.
    Calibrate scales before constructing an optimizer, using training data
    and equilibrium energy/force errors, rather than stationarity RMS alone.
    """

    def __init__(self, *args, driving_scale=None, curvature_scale=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.driving_scale = nn.Parameter(self._dtype_anchor.new_tensor(0.0))
        self.raw_curvature_scale = nn.Parameter(self._dtype_anchor.new_tensor(0.0))
        self.set_scales(
            driving_scale=self.energy_scale if driving_scale is None else driving_scale,
            curvature_scale=self.energy_scale if curvature_scale is None else curvature_scale,
        )

    @property
    def curvature_scale(self):
        return F.softplus(self.raw_curvature_scale)

    @torch.no_grad()
    def set_scales(self, *, driving_scale=None, curvature_scale=None):
        """Set physical scale values in place without replacing parameters."""
        if driving_scale is not None and not math.isfinite(driving_scale):
            raise ValueError("driving_scale must be finite")
        if curvature_scale is not None and (
            not math.isfinite(curvature_scale) or curvature_scale <= 0
        ):
            raise ValueError("curvature_scale must be finite and positive")
        if driving_scale is not None:
            self.driving_scale.fill_(driving_scale)
        if curvature_scale is not None:
            self.raw_curvature_scale.fill_(self._inverse_softplus(curvature_scale))
        return self

    @classmethod
    def from_adapter(cls, adapter, **charge_options):
        energy = super().from_adapter(adapter, **charge_options)
        # The source can use a different dtype than torch's constructor default.
        return energy.set_scales(
            driving_scale=(adapter.energy_scale if charge_options.get("driving_scale") is None
                           else charge_options["driving_scale"]),
            curvature_scale=(adapter.energy_scale if charge_options.get("curvature_scale") is None
                             else charge_options["curvature_scale"]),
        )

    @classmethod
    def from_convex(cls, energy, **scale_options):
        """Copy every weight of a bound original convex energy into this form."""
        if type(energy) is not ConvexQConditionedEFAEnergy:
            raise TypeError("expected an original ConvexQConditionedEFAEnergy")
        if energy.feature_spec is None:
            raise ValueError("bind the convex EFA energy before converting it")
        reference = next(energy.parameters())
        converted = cls(
            hidden_features=energy.hidden_features,
            num_directions=energy.attention.num_directions,
            max_frequency=energy.attention.max_frequency,
            max_length=energy.attention.max_length,
            energy_scale=energy.energy_scale,
            use_equivariant_queries=energy.use_equivariant_queries,
            charge_features=energy.charge_features,
            curvature_floor=energy.curvature_floor,
            hardness_init=energy.curvature_floor + 1.0,
            **scale_options,
        ).to(device=reference.device, dtype=reference.dtype)
        converted.set_scales(
            driving_scale=(energy.energy_scale if scale_options.get("driving_scale") is None
                           else scale_options["driving_scale"]),
            curvature_scale=(energy.energy_scale if scale_options.get("curvature_scale") is None
                             else scale_options["curvature_scale"]),
        )
        converted.bind(copy.deepcopy(energy.feature_spec))
        state = energy.state_dict()
        state["_extra_state"] = converted.get_extra_state()
        state["driving_scale"] = converted.driving_scale.detach()
        state["raw_curvature_scale"] = converted.raw_curvature_scale.detach()
        converted.load_state_dict(state)
        converted.train(energy.training)
        return converted

    def response_terms(self, context, state):
        """Differentiable terms for calibration, using one attention evaluation.

        ``linear`` and ``curvature`` include positive output weights but no
        beta/alpha scales. ``geometry_energy`` already includes energy_scale.
        ``offset`` and ``response`` expose b and t for derivative diagnostics.
        Nothing is detached, normalized or projected onto a charge constraint.
        """
        geometry, offset, response = self._attention_terms(context, state)
        linear_terms = torch.sigmoid(offset) * response
        nonlinear_terms = F.softplus(offset + response) - F.softplus(offset)
        weights = self.positive_output_weights
        return {
            "geometry_energy": self.energy_scale * self.readout(geometry).sum(),
            "linear": (weights * linear_terms).sum(),
            "curvature": (weights * (nonlinear_terms - linear_terms)).sum(),
            "offset": offset,
            "response": response,
        }

    def forward(self, context, state):
        terms = self.response_terms(context, state)
        return (
            terms["geometry_energy"]
            + self.driving_scale * terms["linear"]
            + self.curvature_scale * terms["curvature"]
            + 0.5 * self.hardness * state.square().sum()
        )

    def get_extra_state(self):
        return {
            **super().get_extra_state(),
            "schema": "two_scale_q_efa_energy_v1",
            "scale_parameterization": {"driving": "signed", "curvature": "softplus"},
        }


__all__ = ["TwoScaleQConditionedEFAEnergy"]
