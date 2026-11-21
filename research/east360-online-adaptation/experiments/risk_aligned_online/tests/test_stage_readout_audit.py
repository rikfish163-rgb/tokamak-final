import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.stage_readout_audit_20261028 import (
    factorial_predictions, sample_endpoints, summarize_stage,
)


class ToyAdapter:
    def __init__(self):
        self.state = {'slow': torch.tensor(7.), 'fast_output': torch.tensor(9.)}

    def adapted_state(self):
        return {k: v.clone() for k, v in self.state.items()}

    def restore(self, state):
        self.state = {k: v.clone() for k, v in state.items()}

    def risk(self, h, valid, delta):
        return h + self.state['slow'] + self.state['fast_output'] * delta[:, 0]


def test_factorial_swap_holds_fast_state_fixed_and_restores_model():
    model = ToyAdapter()
    source = {'slow': torch.tensor(0.), 'fast_output': torch.tensor(1.)}
    cold = {'slow': torch.tensor(2.), 'fast_output': torch.tensor(-3.)}
    delta = torch.tensor([[.1], [.2]])
    result = factorial_predictions(model, torch.zeros(2), None, delta, source, cold)
    assert torch.equal(model.state['slow'], torch.tensor(7.))
    assert torch.equal(model.state['fast_output'], torch.tensor(9.))
    assert torch.equal(delta, torch.tensor([[.1], [.2]]))
    np.testing.assert_allclose(result['source']['on'], [.1, .2])
    np.testing.assert_allclose(result['cold_slow_only']['on'], [2.1, 2.2])
    np.testing.assert_allclose(result['cold_output_only']['on'], [-.3, -.6])
    np.testing.assert_allclose(result['cold']['on'], [1.7, 1.4])
    np.testing.assert_array_equal(result['source']['off'], result['cold_output_only']['off'])


def test_failed_counterfactual_restores_original_model():
    model = ToyAdapter()
    model.risk = lambda *args: torch.tensor([float('nan')])
    state = model.adapted_state()
    with pytest.raises(FloatingPointError):
        factorial_predictions(model, torch.zeros(1), None, torch.ones(1, 1), state, state)
    assert model.state['slow'] == 7.
    assert model.state['fast_output'] == 9.


def test_sample_uses_entire_valid_record_not_alarm_or_outcome_class():
    ends = sample_endpoints(np.arange(1000))
    assert len(ends) == 32
    assert ends[0] == 127 and ends[-1] == 999
    assert np.all(np.diff(ends) > 0)
    np.testing.assert_array_equal(sample_endpoints(np.arange(140)), np.arange(127, 140))
    with pytest.raises(ValueError):
        sample_endpoints(np.arange(127))


def test_stage_summary_uses_shot_equal_weight_and_undefined_zero_effect():
    result = summarize_stage([
        {'mean_probability_change': .2, 'rms_probability_change': .2,
         'mean_logit_change': .4, 'positive_point_fraction': 1.},
        {'mean_probability_change': -.1, 'rms_probability_change': .1,
         'mean_logit_change': -.2, 'positive_point_fraction': 0.},
    ])
    assert result['shots'] == 2
    assert result['mean_probability_change']['mean'] == pytest.approx(.05)
    assert result['positive_mean_shots'] == 1
    assert result['negative_mean_shots'] == 1
