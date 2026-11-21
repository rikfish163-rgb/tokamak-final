import numpy as np

from experiments.risk_aligned_online import commissioning_local_response_20261114 as model


def test_return_status_matches_real_prefix_and_censoring_is_not_a_return():
    n = 300
    x = np.full((n, 1), .2)
    x[150:166] = 5.
    x[204:] = 5.
    valid = np.ones_like(x, dtype=bool)
    fixed = dict(value=x, observed_now=valid, center=np.zeros_like(x),
                 spread=np.ones_like(x), reference_known=valid)
    windows = model.events(fixed, [0])
    assert windows['onset'].tolist() == [152, 206]
    assert windows['landmark'].tolist() == [177, 231]
    assert windows['response'].tolist() == [-1, 1]
    assert (windows['known_at'] == windows['landmark']+1).all()
    shortened = model.events({k:v[:233] for k,v in fixed.items()}, [0])
    for k in windows:
        np.testing.assert_array_equal(windows[k], shortened[k])
    ending_too_early = model.events({k:v[:232] for k,v in fixed.items()}, [0])
    assert ending_too_early['response'].tolist() == [-1]
    interrupted = {k:v.copy() for k,v in fixed.items()}
    interrupted['observed_now'][220] = False
    assert 206 not in model.events(interrupted, [0])['onset']


def test_holdout_is_never_shared_with_the_event_definition():
    cols = model.DRIVERS['magnetic']
    n = 240
    x = np.zeros((n, len(model.DIAGNOSTICS)))
    mask = np.ones_like(x,dtype=bool)
    x[145:,cols] = 1.
    r1 = model.recovery.rolling_reference(x,mask)
    a = model.events(r1,cols)
    assert a['response'].tolist() == [1]
    x[160:,model.HELD['radiation']] = 10.
    r2 = model.recovery.rolling_reference(x,mask)
    b = model.events(r2,cols)
    for name in a:
        np.testing.assert_array_equal(a[name],b[name])
    assert not set(model.HELD['radiation']) & set(cols)


def test_fixed_release_can_observe_older_responses_without_using_later_starts():
    n = 400
    x = np.full((n,1), .2)
    x[150:166] = 5.
    x[204:] = 5.
    observed = np.ones_like(x,dtype=bool)
    ref = dict(value=x,observed_now=observed,center=np.zeros_like(x),
               spread=np.ones_like(x),reference_known=observed)
    # q=159 results in a retrospective landmark at 184 that can use
    # the confirmed return of the first episode and not the next onset.
    old = model.events(ref,[0],slots=True)
    assert old['landmark'][:3].tolist() == [184,248,280]
    assert old['response'][:3].tolist() == [-1,1,1]
    assert old['onset'][:3].tolist() == [152,206,206]
    cut = {k:v[:218] for k,v in ref.items()}
    short = model.events(cut,[0],slots=True)
    for k in old:
        np.testing.assert_array_equal(old[k][old['known_at']<218],short[k])


def test_next_diagnostic_window_needs_full_observations_and_is_expressed_at_issue():
    n = 270
    wave = np.zeros((n,2))
    wave[216:] = 8.
    m = np.ones_like(wave,dtype=bool)
    t, a = np.array([215,244,245]), np.array([150,160,161])
    packet = model.held_packet(wave,m,t,a)
    np.testing.assert_array_equal(packet['mature'], [True,False,False])
    assert np.all(packet['future'][0]>packet['recent'][0])
    m[225,0] = False
    gap = model.held_packet(wave,m,t,a)
    assert not gap['mature'][0]
    np.testing.assert_array_equal(gap['recent'],packet['recent'])
