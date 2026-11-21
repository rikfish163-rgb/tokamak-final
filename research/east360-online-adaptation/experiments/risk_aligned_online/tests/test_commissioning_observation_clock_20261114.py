import numpy as np
import pytest
import torch

from experiments.risk_aligned_online import commissioning_observation_clock_20261114 as clock


def shot():
    n = 240
    a = np.arange(n * 23, dtype=np.float32).reshape(n, 23)
    mask = np.ones_like(a, dtype=bool)
    mask[15, clock.RADIATION[0]] = False
    return dict(values=a, available=mask)


def test_delayed_radiation_stays_at_physical_position_not_at_receipt_position():
    s = shot()
    t = 25
    x, m = clock.windows(s, np.array([t]), 16, clock.DELAY)
    c = clock.RADIATION[0]
    np.testing.assert_array_equal(np.flatnonzero(m[0, :, c]), [0, 1, 2, 3, 4, 6, 7])
    assert x[0, 7, c] == s['values'][t-8, c]
    assert not x[0, 8:, c].any()
    assert x[0, -1, 0] == s['values'][t, 0]


def test_arrival_view_does_not_reveal_mutated_future_or_pending_samples():
    s = shot()
    u = {k: v.copy() for k, v in s.items()}
    t = 175
    u['values'][t+1:] = 1e6
    u['values'][t-7:t+1, clock.RADIATION] = 1e7
    u['available'][t-7:t+1, clock.RADIATION] = False
    for x, v in zip(clock.windows(s, np.array([t]), 128, clock.DELAY),
                    clock.windows(u, np.array([t]), 128, clock.DELAY), strict=True):
        np.testing.assert_array_equal(x, v)
    x, m = clock.windows(s, np.array([2]), 16, clock.DELAY)
    assert not m[..., clock.RADIATION].any()
    assert m[0, -3:, 0].all() and not m[0, :-3, 0].any()


def test_late_packet_crossing_next_issue_is_not_in_its_reference():
    q = np.arange(127, 127 + 9*32, 32)
    force = torch.arange(9, dtype=torch.float32)[:, None, None]
    eligible = np.ones(len(q), dtype=bool)
    released = q+26+8
    expected, current, earlier, ready = clock.innovation(q, released, force, eligible)
    assert not ready[:5].any()
    assert ready[5]
    torch.testing.assert_close(expected[5], torch.tensor([[6./5.]]))
    torch.testing.assert_close(current[5], torch.tensor([[3.8]]))
    # At q[5], the most recent fully available eligible innovation is q[3].
    torch.testing.assert_close(earlier[5], torch.tensor([[8./3.]]))
    exact = clock.innovation(q, q+26, force, eligible)
    original = clock.group.past_force_parts(q, force, eligible)
    for a, b in zip(exact, original, strict=True):
        if torch.is_tensor(a):
            torch.testing.assert_close(a, b, rtol=0., atol=0.)
        else:
            np.testing.assert_array_equal(a, b)


def test_observation_view_matches_existing_shape_and_motion_at_zero_age():
    s = shot()
    ends = np.array([0, 63, 127, 180, 239])
    z, m = s['values'], s['available']
    cols = list(clock.lag.CHANNELS)
    c = list(clock.CONTEXT)
    design = clock.lag.direct.shape.inputs(z[:, cols], m[:, cols], z[:, c], m[:, c], ends)
    current = clock.input_design(z[:, cols], m[:, cols], z[:, c], m[:, c], ends, 0)
    for a, b in zip(design, current, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(clock.motion(s, ends, 0), clock.contrast(z, m)[ends])


def test_motion_does_not_resurrect_unavailable_border_or_shift_a_timestamp():
    s = shot()
    t = 160
    idx = np.array([t])
    z, m = clock.windows(s, idx, 64, clock.DELAY)
    expected = clock.contrast(z[0], m[0])[-1]
    np.testing.assert_array_equal(clock.motion(s, idx, clock.DELAY)[0], expected)


def test_delayed_input_preserves_the_production_forecaster_packet_contract():
    s = shot()
    net = torch.nn.Module()
    net.stem = torch.nn.Linear(186, 24)
    params = (np.zeros(23,np.float32), np.ones(23,np.float32))
    tr = clock._observed_track(s, net, params, clock.DELAY)
    readout = np.zeros((clock.lag.H*len(clock.lag.CHANNELS), tr['design'].shape[1]+1), np.float32)
    q,key,forecast,observed = clock.group.teacher._mature(tr, (readout[:,:-1],readout[:,-1]),
                                                       np.ones(10,np.float32))
    np.testing.assert_array_equal(q, [127,159,191])
    assert key.shape==(3,4)
    assert forecast.shape==observed.shape==(3,25,10)
    np.testing.assert_array_equal(forecast, np.zeros_like(observed))


def test_prefix_numeric_tolerance_never_permits_a_different_alarm():
    with pytest.raises(AssertionError):
        clock.prefix_check(np.array([.49999]), np.array([.50001]), [.5])
    assert clock.prefix_check(np.array([.49997]), np.array([.49999]), [.5]) < .00003


def test_unreceived_tail_mutation_does_not_change_the_delayed_feature_design():
    s = shot()
    u = {k: v.copy() for k, v in s.items()}
    t = 175
    u['values'][t+1:] = -1e6
    u['values'][t-7:t+1, clock.RADIATION] = 1e7
    u['available'][t-7:t+1, clock.RADIATION] = False
    cols, context = list(clock.lag.CHANNELS), list(clock.CONTEXT)
    views = []
    for v in (s,u):
        a,m=v['values'],v['available']
        views.append(clock.input_design(a[:,cols],m[:,cols],a[:,context],m[:,context],
                                       np.array([t]),clock.DELAY))
    for a,b in zip(*views,strict=True):
        np.testing.assert_array_equal(a,b)
