import numpy as np

from experiments.risk_aligned_online import commissioning_recovery_clock_20261111 as recovery


def track(x, *, reference=None, seen=None):
    x = np.asarray(x, dtype=float)[:, None]
    n = len(x)
    c = np.zeros_like(x) if reference is None else np.asarray(reference, dtype=float)[:, None]
    observed = np.ones(n, dtype=bool) if seen is None else np.asarray(seen, dtype=bool)
    return recovery.episode_trace(x, observed, c, np.ones_like(x), np.ones(n, dtype=bool))


def test_onset_and_return_only_act_after_confirmation():
    view = track([0, 4, 4, 4, 4, 1, 1, 1, 0])
    snapshot = recovery.sample_at_release(view, np.arange(1, 9))
    assert snapshot["state"].tolist() == [0, 0, 0, 1, 1, 1, 1, 0]
    assert snapshot["return_age"].tolist() == [-1, -1, -1, -1, -1, -1, -1, 0]
    assert view["episodes"].tolist() == [[3, 7, recovery.RETURNED]]


def test_reference_cannot_chase_a_persistent_excursion():
    view = track([0, 4, 4, 4, 4, 4, 4], reference=[0, 0, 0, 4, 4, 4, 4])
    assert view["episodes"].tolist() == [[3, 6, recovery.END_OF_RECORD]]
    assert view["state"][-1] == 1
    assert view["distance"][-1] == 4


def test_missing_observation_interrupts_without_qualifying_as_return():
    view = track([4, 4, 4, 1, 1, 1, 1], seen=[1, 1, 1, 1, 1, 0, 1])
    assert view["episodes"].tolist() == [[2, 4, recovery.OBSERVATION_GAP]]
    assert view["state"].tolist() == [0, 0, 1, 1, 1, -1, 0]
    assert (view["return_age"] == -1).all()


def test_actual_prefix_keeps_same_features_without_seeing_record_end():
    x = [0, 4, 4, 4, 4, 1, 1, 1, 0, 0]
    whole = track(x)
    for stop in (3, 5, 8):
        prefix = track(x[:stop])
        r = np.arange(1, stop+1)
        a, b = (recovery.sample_at_release(q, r) for q in (whole, prefix))
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])


def test_reference_window_is_earlier_than_the_current_five_samples():
    n = recovery.REFERENCE_POINTS + recovery.REFERENCE_LAG + 12
    base = np.zeros((n, 1))
    t = recovery.REFERENCE_POINTS + recovery.REFERENCE_LAG
    changed = base.copy()
    changed[t-4:t+1] = 7
    a, b = (recovery.rolling_reference(q, np.ones_like(q, dtype=bool)) for q in (base, changed))
    np.testing.assert_array_equal(a["center"][:t+1], b["center"][:t+1])
    assert b["value"][t, 0] == 7
    assert b["reference_known"][t, 0]
