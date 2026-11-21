import numpy as np
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.shadow_commit_20261104.timeline import trace, issue


def fixture(n=240):
    torch.manual_seed(3)
    b = (
        Simple23(Config(fast_site="tokens"), torch.zeros(23), torch.ones(23))
        .eval()
        .requires_grad_(False)
    )
    x = np.random.default_rng(3).normal(size=(n, 23)).astype("float32")
    return b, dict(values=x, available=np.ones_like(x, bool)), torch.zeros(168, 70)


def test_first_feedback_only_learns_shadow_and_never_changes_risk():
    b, s, w = fixture()
    delta, history, receipt = trace(b, s, w, "real", policy="delay", device="cpu")
    assert not delta[:167].any() and delta[167:].abs().sum() > 0
    assert history[0]["arrival"] == 135 and not history[0]["parameter_changed"]
    assert history[1]["candidate_version"] == 1
    assert receipt["issued_at"].tolist() == [127, 159, 191, 223]
    assert receipt["released_at"].tolist() == [135, 167, 199, 231]


def test_appending_future_never_changes_issued_forecasts_or_old_risk():
    b, s, w = fixture()
    full, _, r = trace(b, s, w, "real", policy="guard", device="cpu")
    prefix, _, q = trace(
        b, {k: v[:200] for k, v in s.items()}, w, "real", policy="guard", device="cpu"
    )
    assert torch.equal(prefix, full[:200])
    assert np.array_equal(q["candidate_prediction"], r["candidate_prediction"][:3])
    assert np.array_equal(q["active_prediction"], r["active_prediction"][:3])


def test_pending_forecast_has_no_invented_target_and_state_resets():
    b, s, w = fixture(n=164)
    a, h, r = trace(b, s, w, "real", policy="delay", device="cpu")
    aa, hh, rr = trace(b, s, w, "real", policy="delay", device="cpu")
    assert torch.equal(a, aa) and h == hh and not w.any()
    assert r["issued_at"].tolist() == [127, 159]
    assert r["released_at"].tolist() == [135, -1]
    assert not r["target_mask"][-1].any()
    assert np.array_equal(r["candidate_prediction"], rr["candidate_prediction"])


def test_guard_only_commits_on_separately_observed_better_forecast():
    b, s, w = fixture()
    _, h, _ = trace(b, s, w, "real", policy="guard", device="cpu")
    assert not h[0]["accepted"]
    for row in h:
        assert row["origin"] + 8 == row["arrival"]
        assert row["accepted"] == (
            row["support"] > 0 and row["candidate_mse"] < row["active_mse"]
        )


def test_issued_snapshot_is_not_an_alias_of_later_weights():
    w = torch.ones(1, 168, 70)
    a = torch.zeros_like(w)
    key = torch.ones(1, 70)
    pending = issue(w, a, key, origin=127, candidate_version=2, active_version=0)
    w.zero_()
    a.add_(3)
    assert torch.equal(pending.candidate_weight, torch.ones_like(w))
    assert torch.equal(pending.candidate_prediction, torch.full((1, 168), 70.0))
    assert not pending.active_prediction.any()


def test_absent_target_support_never_commits_or_invents_performance():
    b, s, w = fixture()
    s["available"][:] = False
    d, h, r = trace(b, s, w, "real", policy="delay", device="cpu")
    assert not d.any() and not r["target_mask"].any()
    assert all(
        not x["accepted"] and not x["parameter_changed"] and x["support"] == 0
        for x in h
    )
