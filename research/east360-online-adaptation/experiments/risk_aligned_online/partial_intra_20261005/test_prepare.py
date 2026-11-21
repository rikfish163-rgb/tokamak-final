from copy import deepcopy

from .prepare import admission


def report(whole, later, tp=15, fp=2, maintenance=.16):
    return dict(whole=dict(balanced_loss=whole, metrics30=dict(tp=tp, fp=fp)),
                later=dict(mean_loss=later), maintenance=maintenance)


def test_admission_requires_risk_gain_clock_gain_and_signal_only_control():
    refs = [report(.22, .15), report(.23, .14)]
    good = report(.20, .12)
    result = admission(good, refs, .16, report(.21, .13), report(.205, .13))
    assert result['admitted']
    assert not admission(good, refs, .16, good, report(.205, .13))['admitted']
    assert not admission(good, refs, .16, report(.21, .13), good)['admitted']
    bad = deepcopy(good); bad['whole']['metrics30']['fp'] = 3
    assert not admission(bad, refs, .16, report(.21, .13), report(.205, .13))['admitted']


def test_lower_signal_loss_cannot_compensate_for_bad_whole_risk():
    bad = report(.221, .10, maintenance=.01)
    result = admission(bad, [report(.22, .15)], .16, report(.24, .14), report(.25, .14))
    assert not result['source_gate'] and not result['admitted']
