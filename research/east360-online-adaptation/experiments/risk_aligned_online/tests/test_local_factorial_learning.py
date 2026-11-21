import numpy as np
import torch

from experiments.risk_aligned_online.local_factorial_20261029.learning import (
    ARMS,
    MODES,
    prepare,
    feedback,
    predict,
)
from experiments.risk_aligned_online.local_readout_20261029.learning import History
from experiments.risk_aligned_online.tests.test_local_readout import make_head
from experiments.risk_aligned_online.tests.test_qk_matched_head import fixture


def test_factorial_separates_fit_target_from_recent_target():
    net, s = fixture()
    prior = make_head()
    sig = {k: s[k][:256] for k in ("values", "available")}
    data = prepare(net, sig, None, prior, History())
    assert len(ARMS) == 11 and len(MODES) == 10
    assert MODES["true_reverse"] == ("true", "reverse")
    assert MODES["reverse_true"] == ("reverse", "true")
    changed = {k: v.copy() for k, v in sig.items()}
    changed["values"][206:] += 12
    other = prepare(net, changed, None, prior, History())
    for mode in ("prior", "true", "reverse", "hold"):
        torch.testing.assert_close(
            data["heads"][mode].coef, other["heads"][mode].coef, rtol=0, atol=0
        )
    assert not torch.equal(data["wanted"], other["wanted"])
    assert data["receipt"]["calibration_last_target"] == 205
    assert data["receipt"]["gradient_first_target"] == 206


def test_accepted_feedback_decreases_loss_and_preserves_model():
    net, s = fixture()
    prior = make_head()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    sig = {k: s[k][:256] for k in ("values", "available")}
    states, receipt, heads = feedback(net, sig, None, prior, History())
    assert set(states) == set(MODES) == set(receipt["arms"])
    assert heads["true_true"] is heads["true_reverse"]
    for name, state in states.items():
        r = receipt["arms"][name]
        assert 0 <= r["trials"] <= 8 and state.norm() <= 0.050001
        if r["reason"] == "accepted":
            assert r["loss_after"] < r["loss_before"]
            assert r["directional_dot"] < 0
            assert r["loss_after"] <= r["loss_before"] + 1e-4 * r["directional_dot"]
        else:
            assert torch.count_nonzero(state) == 0
    assert all(torch.equal(v, before[k]) for k, v in net.state_dict().items())
    missing = {k: v.copy() for k, v in sig.items()}
    missing["available"][:] = False
    zero, r, _ = feedback(net, missing, None, prior, History())
    assert all(torch.count_nonzero(v) == 0 for v in zero.values())
    assert all(
        x["reason"] == "no_observed_target" and x["observed"] == 0
        for x in r["arms"].values()
    )


def test_prediction_resets_and_future_changes_cannot_rewrite_past():
    net, s = fixture()
    prior = make_head()
    sig = {k: s[k][:400] for k in ("values", "available")}
    a, info, aux = predict(net, sig, None, prior)
    again, info_again, _ = predict(net, sig, None, prior)
    changed = {k: v.copy() for k, v in sig.items()}
    changed["values"][320:] += 10
    other, _, _ = predict(net, changed, None, prior)
    for arm in ARMS:
        np.testing.assert_array_equal(a[arm], again[arm])
        np.testing.assert_array_equal(a[arm][:320], other[arm][:320])
        np.testing.assert_array_equal(a[arm][:256], a["cold"][:256])
    assert info == info_again
    assert len(info["feedback"]) == 2
    assert all(row["clock"] + 25 <= 400 for row in aux)
