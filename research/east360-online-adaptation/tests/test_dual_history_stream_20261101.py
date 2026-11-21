import numpy as np
import pytest

from experiments.risk_aligned_online.dual_history_ttt_20261101.stream import (
    feedback_origin,
    targets,
)


def test_feedback_clock_is_at_chunk_end_after_full_horizon():
    assert feedback_origin(159) == 134
    assert feedback_origin(287) == 262
    assert all(feedback_origin(e) is None for e in range(159))
    assert feedback_origin(160) is None
    assert feedback_origin(286) is None


def test_targets_same_support_reverse_and_hold_and_only_arrived():
    values = np.arange(310 * 23, dtype=np.float32).reshape(310, 23)
    available = np.ones_like(values, dtype=bool)
    available[140, 2] = False
    available[134, 5] = False
    signal = dict(values=values, available=available)
    truth, mask = targets(signal, 134, 159)
    assert set(truth) == {"real", "reverse", "hold"}
    assert mask.shape == (1, 25, 23)
    assert not mask[:, :, 10].any() and not mask[:, :, 5].any()
    assert not mask[0, 5, 2] and not mask[0, 19, 2]
    assert np.array_equal(mask, mask[:, ::-1])
    assert np.array_equal(truth["real"][0], values[135:160])
    assert np.array_equal(truth["reverse"], truth["real"][:, ::-1])
    assert np.array_equal(truth["hold"][0], np.repeat(values[134:135], 25, axis=0))
    prefix = {k: v[:160].copy() for k, v in signal.items()}
    changed = {k: v.copy() for k, v in signal.items()}
    changed["values"][160:] = -9999
    for other in (prefix, changed):
        t, m = targets(other, 134, 159)
        assert np.array_equal(m, mask)
        assert all(np.array_equal(t[a], truth[a]) for a in truth)


def test_target_clock_cannot_reveal_future():
    signal = dict(values=np.ones((200, 23)), available=np.ones((200, 23), bool))
    with pytest.raises(ValueError):
        targets(signal, 134, 158)
    with pytest.raises(ValueError):
        targets(signal, 134, 160)


def test_prediction_and_first_feedback_do_not_depend_on_later_signals():
    import torch

    from experiments.risk_aligned_online.context_memory_20261028.model import (
        ContextAdapter,
    )
    from experiments.risk_aligned_online.dual_history_ttt_20261101.run import predict
    from experiments.risk_aligned_online.simple23 import Config, Simple23

    torch.set_num_threads(2)
    torch.manual_seed(431)
    torch.backends.mha.set_fastpath_enabled(False)
    model = (
        ContextAdapter(
            Simple23(
                Config(fast_site="tokens", fast_rule="normalized"),
                np.zeros(23),
                np.ones(23),
            )
        )
        .eval()
        .requires_grad_(False)
    )
    rng = np.random.default_rng(44)
    signal = dict(
        values=rng.normal(size=(320, 23)).astype(np.float32),
        available=np.ones((320, 23), bool),
    )
    full, receipt = predict(model, signal)
    prefix = {k: v[:160].copy() for k, v in signal.items()}
    partial, partial_receipt = predict(model, prefix)
    changed = {k: v.copy() for k, v in signal.items()}
    changed["values"][160:] += 1000
    altered, altered_receipt = predict(model, changed)
    for arm in full:
        assert np.array_equal(full[arm][:160], partial[arm])
        assert np.array_equal(full[arm][:160], altered[arm][:160])
    for arm in ("real", "reverse", "hold"):
        first = receipt["updates"][arm][0]
        assert first == partial_receipt["updates"][arm][0]
        assert first == altered_receipt["updates"][arm][0]
        assert first["origin"] == 134 and first["target_last"] == 159
        assert first["first_affected"] == 160
        assert np.array_equal(full[arm][:160], full["cold"][:160])
    repeated, repeated_receipt = predict(model, signal)
    assert all(np.array_equal(repeated[a], full[a]) for a in full)
    assert repeated_receipt == receipt  # State is not carried to the next shot.
