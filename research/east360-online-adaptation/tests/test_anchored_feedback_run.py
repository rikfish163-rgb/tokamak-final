import pytest
import torch

from experiments.risk_aligned_online.anchored_feedback_20261030.run import (
    feedback_controls, candidate_gate,
)


def test_controls_preserve_origin_and_donor_pool_values():
    context = torch.arange(3*358, dtype=torch.float32).reshape(3, 358)
    donor = context.flip(0).clone()
    mean = torch.zeros(358)
    arms = feedback_controls(context, mean, donor)
    for value in arms.values():
        assert torch.equal(value[:, :128], context[:, :128])
    assert torch.equal(arms['donor'][:, 128:], donor[:, 128:])
    assert torch.equal(arms['reverse'][:, 128:], context[:, 128:].reshape(3, 5, 46).flip(1).flatten(1))
    assert not arms['mean_feedback'][:, 128:].any()
    arms['donor'][0, 0] = -99
    assert context[0, 0] == 0


def test_controls_bad_shape_rejected():
    with pytest.raises(ValueError):
        feedback_controls(torch.zeros(2, 358), torch.zeros(358), torch.zeros(1, 358))


def test_gate_does_not_reward_tautological_mean_zero():
    groups = {arm: {g: dict(mean_delta_bce=-.01 if arm == 'real' else 0.,
                           improved=2 if g == 'safe' else 10)
                    for g in ('safe', 'danger')}
              for arm in ('real', 'reverse', 'mean_feedback', 'donor')}
    assert candidate_gate(groups, .8, .9)['passed']
    groups['donor']['danger']['mean_delta_bce'] = -.02
    assert not candidate_gate(groups, .8, .9)['passed']
    groups['donor']['danger']['mean_delta_bce'] = 0.
    groups['mean_feedback']['danger']['mean_delta_bce'] = -99
    assert candidate_gate(groups, .8, .9)['passed']


def test_gate_rejects_no_utility_or_no_direction():
    groups = {arm: {g: dict(mean_delta_bce=0., improved=0)
                    for g in ('safe', 'danger')}
              for arm in ('real', 'reverse', 'mean_feedback', 'donor')}
    assert not candidate_gate(groups, None, 0.)['passed']
