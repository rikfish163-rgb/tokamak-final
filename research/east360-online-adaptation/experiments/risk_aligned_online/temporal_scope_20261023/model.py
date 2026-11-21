"""Expand OFFLINE EAST preparation, never the deployed fast update scope."""
import copy

from ..increment_forecast_20261023.learning import IncrementFeedback


class BothBlockPreparation(IncrementFeedback):
    """Temporary training class; the unchanged linear risk head stays frozen."""
    def encoder_names(self):
        first=super().encoder_names()
        return first+[k for k,_ in self.named_parameters() if k.startswith('base.encoder.layers.1.')]


def to_deployment(prepared):
    if not isinstance(prepared,BothBlockPreparation):raise TypeError('explicit offline two-block preparation required')
    deployed=IncrementFeedback(copy.deepcopy(prepared.base),'increment')
    deployed.load_state_dict(prepared.state_dict());deployed.set_stage('frozen')
    assert not any(k.startswith('base.encoder.layers.1.') for k in deployed.encoder_names())
    return deployed
