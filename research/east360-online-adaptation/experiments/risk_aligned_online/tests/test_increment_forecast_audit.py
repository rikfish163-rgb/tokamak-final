import pytest

from experiments.risk_aligned_online.increment_forecast_20261023.audit import gate


def test_restoring_static_is_not_intra_gain_and_control_must_be_beaten():
    static = dict(tp=3, fp=1, n_disrupt=4, n_safe=16)
    assert not gate(static, static, dict(static, tp=2))['pass']
    better = dict(static, tp=4)
    assert gate(better, static, dict(static, tp=2))['pass']
    assert not gate(better, static, better)['pass']


def test_fp_increase_allowed_inside_cap_when_tpr_and_j_improve():
    static = dict(tp=2, fp=1, n_disrupt=4, n_safe=16)
    assert gate(dict(static, tp=3, fp=2), static, static)['pass']
    assert not gate(dict(static, tp=4, fp=4), static, static)['pass']
    with pytest.raises(ValueError):
        gate(dict(static, n_safe=15), static, static)
