import copy

import pytest

from experiments.risk_aligned_online.operational_transport_20261030.stream import NAMES
from experiments.risk_aligned_online.operational_transport_20261030.verify import check_receipt


def receipt():
    norms = {name: 0.02 for name in NAMES}
    norms['cold'] = norms['identical'] = 0.
    zero = {name: 0. for name in NAMES}
    r = dict(observed=550, signal_gradient_norm=1., reverse_gradient_norm=1.,
             risk_gradient_norm=1., state_norm=norms, max_bound=.02,
             cosine_signal=zero.copy(), cosine_risk=zero.copy(),
             cosine_operational=zero.copy())
    r['cosine_signal']['signal_axis'] = -1.
    r['cosine_risk']['operational'] = -.4
    r['cosine_risk']['risk_axis'] = -1.
    r['cosine_operational']['operational'] = 1.
    return r


def test_good_receipt_and_exact_risk_orthogonality():
    check_receipt(receipt())
    r = receipt()
    r['state_norm']['risk_axis'] = 0.
    r['cosine_risk']['risk_axis'] = r['cosine_risk']['operational'] = 0.
    check_receipt(r)


@pytest.mark.parametrize('field,value', [('observed', -1), ('signal_gradient_norm', float('nan')),
                                        ('max_bound', .049)])
def test_invalid_receipt_is_rejected(field, value):
    r = receipt()
    r[field] = value
    with pytest.raises(AssertionError):
        check_receipt(r)


@pytest.mark.parametrize('change', ['bound', 'axis', 'wrong_direction', 'missing_signal', 'cold'])
def test_control_violations_are_rejected(change):
    r = copy.deepcopy(receipt())
    if change == 'bound':
        r['state_norm']['point'] = r['max_bound'] = .051
    elif change == 'axis':
        r['state_norm']['risk_axis'] = .01
    elif change == 'wrong_direction':
        r['cosine_signal']['signal_axis'] = 1.
    elif change == 'missing_signal':
        r['observed'] = 0
    else:
        r['state_norm']['cold'] = .01
    with pytest.raises(AssertionError):
        check_receipt(r)
