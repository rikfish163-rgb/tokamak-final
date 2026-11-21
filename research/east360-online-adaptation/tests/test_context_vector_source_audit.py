import numpy as np
import pytest

from experiments.risk_aligned_online.context_vector_20261030.audit_source_run import recount


def example():
    records = [dict(shot=1, group='safe', label=0.)] * 3 + [
        dict(shot=2, group='safe', label=0.), dict(shot=3, group='danger', label=1.)]
    pred = dict(baseline=np.zeros(5))
    for arm in ('real', 'reverse', 'mean_feedback', 'constant', 'risk_axis'):
        pred[arm+'_logit'] = np.array([1., 1., 1., -1., 1.])
        pred[arm+'_norm'] = np.ones(5)*.1
        pred[arm+'_cos_real'] = np.ones(5)
    return records, pred


def test_independent_audit_weights_shots_not_windows():
    records, pred = example()
    result = recount(records, pred)
    assert result['groups']['real']['safe']['shots'] == 2
    assert result['groups']['real']['safe']['windows'] == 4
    expected = (np.logaddexp(0., 1.) + np.logaddexp(0., -1.))/2 - np.log(2.)
    assert result['groups']['real']['safe']['mean_delta_bce'] == pytest.approx(expected)
    assert result['source_screen_passed'] is False


def test_bad_prediction_cannot_be_summarized_as_success():
    records, pred = example()
    pred['real_logit'][0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        recount(records, pred)
    pred['real_logit'] = np.zeros(4)
    with pytest.raises(ValueError, match='shape'):
        recount(records, pred)


def test_empty_early_group_is_visible_not_fake_zero():
    records, pred = example()
    result = recount(records, pred)
    assert result['groups']['real']['early']['shots'] == 0
    assert result['groups']['real']['early']['mean_delta_bce'] is None
