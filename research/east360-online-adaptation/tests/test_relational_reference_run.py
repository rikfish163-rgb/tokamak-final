import numpy as np

from experiments.risk_aligned_online.relational_attention_20261030.run_reference_scale import acceptance


def example():
    groups = {a: {g: dict(shots=n, supported_shots=n, improved=n,
                          mean_delta_bce=-.02 if a == 'real' else -.01)
                  for g, n in [('safe', 3), ('danger', 17)]}
              for a in ('real', 'reverse', 'unconditioned', 'mean_target', 'mean_direction')}
    pred = dict(direction_nonzero=np.ones(3, bool), real_reverse_cosine=np.array([.5, .6, .7]),
                real=np.array([.2, .3, .4]), reverse=np.array([.1, .1, .1]))
    return groups, pred


def test_all_declared_conditions_required():
    groups, pred = example()
    assert acceptance(groups, pred)['full_stream_gate']
    groups['real']['safe']['mean_delta_bce'] = .001
    assert not acceptance(groups, pred)['full_stream_gate']


def test_specificity_is_not_benefit_and_no_vectors_not_success():
    groups, pred = example()
    groups['real']['danger']['improved'] = 8
    assert not acceptance(groups, pred)['full_stream_gate']
    pred['direction_nonzero'][:] = False
    assert acceptance(groups, pred)['real_reverse_median_cosine'] is None
    assert not acceptance(groups, pred)['full_stream_gate']


def test_tie_with_control_is_not_a_pass():
    groups, pred = example()
    groups['mean_target']['safe']['mean_delta_bce'] = -.02
    assert not acceptance(groups, pred)['full_stream_gate']
