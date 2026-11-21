from experiments.risk_aligned_online.conditional_channel_policy_20261102.trial import (
    summarize,
    gates,
)


def test_policy_summary_is_shot_macro_and_a_fake_context_win_cannot_be_renamed():
    rows = [
        dict(shot=1, group="safe", delta=-2.0, proxy_slope=-1.0),
        dict(shot=1, group="safe", delta=-4.0, proxy_slope=-3.0),
        dict(shot=2, group="safe", delta=1.0, proxy_slope=2.0),
        dict(shot=3, group="danger30_125", delta=-1.0, proxy_slope=-2.0),
        dict(shot=3, group="disruptive_early", delta=0.0, proxy_slope=-1.0),
    ]
    s = summarize(rows)
    assert s["safe"]["mean_delta"] == -1.0
    assert s["safe"]["shots"] == 2
    same = {
        "conditional_true_real": s,
        "global_real": s,
        "conditional_shuffled_real": s,
        "conditional_true_reverse": s,
        "conditional_true_hold": s,
    }
    assert not gates(same)["passed"]
