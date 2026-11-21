import numpy as np
import pytest

from experiments.risk_aligned_online.attention_protected_20261029.margin_diagnostic import (
    describe_shot,
)


def call(b, a, *, flag=True, down=.04, start=0):
    b=np.asarray(b,float);a=np.asarray(a,float)
    return describe_shot(np.arange(len(b))*.01,b,a,a,.5,flag,down,start)


def test_timely_deadline_is_strict_and_post_disruption_points_excluded():
    # t=.01 is exactly30ms before disruption; the last sample is after disruption.
    r=call([.1,.7,.1,.1,.1,.9],[.2,.8,.1,.1,.1,.95])
    assert r['cold']['alarm'] and r['cold']['lead_ms']==30
    assert not r['cold']['timely'] and not r['candidate']['timely']
    assert not r['envelope_correctable'] and r['operational_peak_delta']==pytest.approx(.1)


def test_envelope_is_optimistic_not_an_achieved_rescue():
    r=call([.4,.1],[.2,.1])
    assert not r['candidate']['timely'] and r['envelope_correctable']
    assert r['cold_wrong'] and r['operational_peak_delta']==pytest.approx(-.2)


def test_safe_shot_needs_all_alarm_points_removed_and_prefix_cannot_be_erased():
    r=call([.7,.6],[.2,.55],flag=False)
    assert not r['envelope_correctable']
    r=call([.7,.6],[.7,.1],flag=False,start=1)
    assert r['immutable_alarm'] and not r['envelope_correctable']
    r=call([.7,.6],[.2,.1],flag=False)
    assert r['envelope_correctable'] and not r['candidate']['alarm']


def test_deadline_without_mutable_points_is_not_opportunity():
    r=call([.2,.1,.1],[.2,.2,.1],start=1)
    assert r['mutable_timely_points']==0 and not r['envelope_correctable']


def test_zero_effect_has_no_cosine_and_contract_errors_fail():
    r=call([.1,.2],[.1,.2])
    assert r['real_reverse_cosine'] is None and r['max_abs_change']==0
    with pytest.raises(ValueError,match='prefix'):
        call([.2,.1],[.3,.1],start=1)
    with pytest.raises(ValueError):call([.2,np.nan],[.3,.1])
    with pytest.raises(ValueError):call([.2,.1],[.3])


def test_safe_shots_have_no_disruption_time_not_an_imputed_time():
    r=call([.7,.1],[.7,.1],flag=False,down=None)
    assert r['cold']['alarm'] and r['cold']['lead_ms'] is None
    with pytest.raises(ValueError):call([.7,.1],[.7,.1],down=None)


def test_envelope_agrees_with_exhaustive_pointwise_sign_choices():
    from itertools import product
    rng=np.random.default_rng(197)
    for flag in (False,True):
        for _ in range(30):
            b=rng.random(6);a=rng.random(6);a[:2]=b[:2]
            r=call(b,a,flag=flag,down=.07 if flag else None,start=2)
            outcomes=[]
            for signs in product((-1,1),repeat=6):
                curve=np.clip(b+np.asarray(signs)*abs(a-b),0,1)
                # For down=.07, only indices0..3 have strictly >30ms lead.
                outcomes.append(bool(np.any(curve[:4]>=.5)) if flag else bool(np.all(curve<.5)))
            assert r['envelope_correctable']==(r['cold_wrong'] and any(outcomes))
