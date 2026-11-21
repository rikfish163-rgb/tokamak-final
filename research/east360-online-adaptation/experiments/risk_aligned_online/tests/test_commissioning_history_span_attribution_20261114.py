import numpy as np

from experiments.risk_aligned_online import commissioning_history_span_attribution_20261114 as audit


def test_no_whole_view_can_remove_the_alarm_while_a_pointwise_switch_can():
    scores = np.array([[.2,.6,.2,.2],[.2,.2,.6,.2],[.2,.4,.7,.2]])
    static = audit.margin_capacity(scores, np.ones(4,bool), .5)
    assert static['min_static'] and not static['forced_pointwise']
    assert static['can_raise']
    scores[:,2] = .7    # a single shared unsafe crossing cannot be removed
    other = audit.margin_capacity(scores, np.ones(4,bool), .5)
    assert other['forced_pointwise']


def test_cutoff_is_strict_and_a_static_alarm_sometimes_falls_after_it():
    clock = np.array([.965,.970,.975])
    valid = audit.timely(clock,1.)
    np.testing.assert_array_equal(valid,[True,False,False])
    capacity = audit.margin_capacity(np.array([[.4,.9,.9],[.3,.8,.7]]),valid,.5)
    assert not capacity['can_raise']
