import copy

import pytest

from experiments.risk_aligned_online.anchored_stream_20261031.core import (
    operational_status, validate_receipts,
)


def test_operational_decision_requires_actual_full_flow_improvement():
    rows = {'cold': {'full360': dict(tpr=.7, fpr=.18, J=.52)}}
    for variant in ('origin', 'mature'):
        for arm in ('real', 'reverse', 'donor', 'mean_feedback'):
            rows[variant+'__'+arm] = {'full360': dict(tpr=.72, fpr=.18, J=.54)}
    result = operational_status(rows)
    assert result['origin']['operational_improvement']
    assert not result['origin']['real_exceeds_controls']
    assert result['origin']['original_component_screen'] == 'FAILED'
    rows['origin__real']['full360'] = dict(tpr=.9, fpr=.3, J=.6)
    assert not operational_status(rows)['origin']['operational_improvement']
    rows['origin__real']['full360'] = dict(tpr=.69, fpr=.1, J=.59)
    assert not operational_status(rows)['origin']['operational_improvement']
    rows['origin__real']['full360'] = dict(tpr=.72, fpr=.17, J=.55)
    assert operational_status(rows)['origin']['real_exceeds_controls']


def receipt():
    names = ('cold', 'identical', *(v+'__'+a for v in ('origin','mature')
                                   for a in ('real','reverse','mean_feedback','donor')))
    norms = {n: (0. if n in ('cold','identical') or n.endswith('mean_feedback') else .1)
             for n in names}
    return dict(origin=127,maturity=152,first_affected=153,apply_count=2,
                state_norm=norms,max_bound=.1,feedback_supported=True)


def test_all_original_points_scored_before_update_can_act():
    validate_receipts([receipt()],155)
    validate_receipts([],152)
    item = receipt()
    item['first_affected'] = 152
    with pytest.raises(ValueError):
        validate_receipts([item],155)


def test_zero_reference_and_absent_feedback_cannot_act():
    original = receipt()
    for name in ('cold','identical','origin__mean_feedback'):
        item = copy.deepcopy(original)
        item['state_norm'][name] = .01
        with pytest.raises(ValueError):
            validate_receipts([item],155)
    item = receipt()
    item['feedback_supported'] = False
    with pytest.raises(ValueError):
        validate_receipts([item],155)
    item['state_norm'] = {k: 0. for k in item['state_norm']}
    item['max_bound'] = 0.
    validate_receipts([item],155)
