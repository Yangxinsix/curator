"""Shared scalar charge energy with a geometry-conditioned initial charge readout."""
from torch import nn
from ._neural_charge_energy import NeuralChargeEnergy


class NeuralChargeStateEnergy(NeuralChargeEnergy):
    """Keep the existing energy exactly; add an independent initial-state readout.

    ``context['scalar']`` contains invariant backbone response features. For the
    current two-layer MATPES backbone it contains 256 scalar channels collected
    from the representation's response feature specification, not charge labels.
    The surrounding variational head projects this initial state and applies the
    proposal correction. This class neither applies boundary constraints nor
    defines a new coupling matrix. Its Hessian diagonal is inherited unchanged.
    The readout starts at zero and belongs to the physical energy parameters:
    energy/force losses train it; the outer head detaches it for residual-only
    training. The raw initial state depends on geometry, not boundary controls.
    """
    def __init__(self, scalar_channels=None, hidden_features=64, curvature_floor=20.0):
        super().__init__(None, hidden_features, curvature_floor)
        self.q0_readout = None
        if scalar_channels is not None:
            self.bind(scalar_channels)

    def bind(self, scalar_channels, vector_channels=0, components=1, state_kind='scalar', **kwargs):
        super().bind(scalar_channels, vector_channels=vector_channels,
                     components=components, state_kind=state_kind, **kwargs)
        if self.q0_readout is None:
            # A shared bias only shifts all charges equally and is removed by
            # a fixed-total-charge projection, so omit this redundant parameter.
            self.q0_readout = nn.Linear(self.scalar_channels, 1, bias=False)
            nn.init.zeros_(self.q0_readout.weight)
        return self

    def initial_state(self, context):
        if self.q0_readout is None:
            raise RuntimeError('bind energy before predicting the initial state')
        return self.q0_readout(context['scalar'])

    def get_extra_state(self):
        return {**super().get_extra_state(),
                'initial_state_schema': 'neural_charge_state_v1',
                'initial_state_readout': 'invariant_response_scalar_linear_no_bias',
                'initial_state_unit': 'e'}
