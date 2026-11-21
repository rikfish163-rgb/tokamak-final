import copy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.qk_forecast_stream_20261029.learning import (
    feedback_step,
    predict,
    fit_head,
)
from experiments.risk_aligned_online.qk_matched_head_20261029.learning import (
    adapt,
    assemble,
)
from experiments.risk_aligned_online.tests.test_qk_matched_head import fixture


def test_feedback_matches_source_training_update_and_rejects_outcomes():
    net, s = fixture()
    e = assemble([s], [(0, 300)], "cpu")
    prefix = {k: s[k][:256] for k in ("values", "available")}
    state, r = feedback_step(net, prefix, None, "true")
    expected = adapt(net, e["feedback"])[0][0]
    torch.testing.assert_close(state, expected, atol=1e-7, rtol=2e-5)
    assert r["count"] == 256 and r["last_target"] == 255
    assert r["norm"] == pytest.approx(0.05, abs=1e-6)
    with pytest.raises(ValueError):
        feedback_step(net, dict(prefix, label=1), None, "true")
    missing = {k: v.copy() for k, v in prefix.items()}
    missing["available"][:] = False
    zero, r = feedback_step(net, missing, None, "true")
    assert not torch.count_nonzero(zero) and r["observed"] == 0


def test_actual_stream_is_causal_resets_and_keeps_risk_network_fixed():
    net, s = fixture()
    nets = {k: copy.deepcopy(net) for k in ("original", "source", "cold")}
    sig = {k: s[k][:416] for k in ("values", "available")}
    slow = torch.eye(64) * 0.01
    before = {k: v.clone() for k, v in net.state_dict().items()}
    first, info = predict(nets, sig, slow)
    again, info2 = predict(nets, sig, slow)
    changed = {k: v.copy() for k, v in sig.items()}
    changed["values"][288:] += 20
    other, _ = predict(nets, changed, slow)
    for arm in first:
        np.testing.assert_array_equal(first[arm], again[arm])
        np.testing.assert_array_equal(first[arm][:288], other[arm][:288])
        np.testing.assert_array_equal(first[arm][:256], first["cold"][:256])
    for arm in ("source_forecast", "cold_forecast"):
        np.testing.assert_array_equal(first[arm], first["original_qk"])
    assert info == info2 and all(r["updates"] == 2 for r in info.values())
    assert all(
        torch.equal(v, before[k]) for k, v in nets["original"].state_dict().items()
    )


def test_ended_signal_head_fit_cannot_change_static_classifier():
    net, s = fixture()
    sig = {k: s[k] for k in ("values", "available")}
    records = [dict(shot=5, device="EAST", **sig)]
    before = {k: v.clone() for k, v in net.state_dict().items()}
    slow = torch.eye(64) * 0.01
    r = fit_head(net, records, slow, steps=2)
    assert len(r["history"]) == 2 and len(r["receipts"]) == 8
    assert all(
        x["shot"] == 5 and x["last_target"] < len(sig["values"]) for x in r["receipts"]
    )
    now = net.state_dict()
    assert any(
        not torch.equal(v, now[k]) for k, v in before.items() if "forecast_head." in k
    )
    assert all(
        torch.equal(v, now[k]) for k, v in before.items() if "forecast_head." not in k
    )
    assert not any(p.requires_grad for p in net.parameters())
    with pytest.raises(ValueError):
        fit_head(net, [dict(records[0], outcome=1)], slow, steps=2)
