import numpy as np

from experiments.risk_aligned_online.commissioning_operating_memory_20261111 import replay


def episode(context, corrections):
    q = 127 + 32*np.arange(len(context))
    y = np.broadcast_to(np.asarray(corrections, np.float32)[:, None, None], (len(q), 25, 10)).copy()
    pred = np.zeros_like(y)
    ctx = np.column_stack((context, np.zeros((len(q), 2))))
    return pred, y, np.ones_like(y, bool), q, ctx, np.ones(len(q), bool), int(q[-1]+32)


def test_regime_memory_forecasts_from_similar_past_not_latest_regime():
    pred, target, m, q, ctx, ok, n = episode([0, 0, 2, 2, 0], [.8, .8, -.8, -.8, .8])
    states, audit = replay(pred, target, m, q, ctx, ok, n)
    assert audit['forecast_sq']['matched'] < audit['forecast_sq']['recent']
    # Only first two past windows match when returning to the original context.
    assert states['matched'][q[-1]+26, 0] < states['recent'][q[-1]+26, 0]
    assert states['matched'][q[-1]+26, 0] < states['wrong_context'][q[-1]+26, 0]


def test_unarrived_truth_is_not_available_for_either_forecast_or_state():
    pred, target, m, q, ctx, ok, n = episode([0, 1, 0, 1], [.7, -.7, .7, -.7])
    full, _ = replay(pred, target, m, q, ctx, ok, n)
    horizon = int(q[-1]+15)
    altered = target.copy()
    altered[-1] = 3
    prefix, _ = replay(pred, altered, m, q, ctx, ok, horizon)
    for k, s in full.items():
        np.testing.assert_array_equal(s[:horizon], prefix[k])
        assert not s[:153].any()
    assert not any(x.any() for x in replay(pred[:1], target[:1], m[:1], q[:1], ctx[:1], ok[:1], 150)[0].values())


def test_missing_targets_do_not_enter_history_and_fresh_shot_has_no_state():
    pred, target, m, q, ctx, ok, n = episode([0, 0, 0], [.6, .6, .6])
    m[0, :, 1:] = False  # not even two families
    first, audit = replay(pred, target, m, q, ctx, ok, n)
    target[0] = 1000
    second, _ = replay(pred, target, m, q, ctx, ok, n)
    assert audit['released'] == 2
    for k in first:
        assert not first[k][:185].any()
        np.testing.assert_array_equal(first[k], second[k])
