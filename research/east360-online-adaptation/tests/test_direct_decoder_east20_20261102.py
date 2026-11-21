import copy

from experiments.risk_aligned_online.direct_decoder_meta_20261102.east20 import gates


def test_local_gate_requires_all_risk_groups_and_real_feedback():
    v = dict(
        disruptive=dict(mean_delta_bce=-0.1),
        safe=dict(mean_delta_bce=-0.1),
        danger_window=dict(mean_delta_bce=-0.1),
        balanced=-0.1,
    )
    s = {
        a: copy.deepcopy(v)
        for a in (
            "forecast_only_real",
            "original_real",
            "forecast_only_reverse",
            "forecast_only_hold",
        )
    }
    assert not gates(s)["passed"]
    s["forecast_only_real"]["balanced"] = -0.2
    assert gates(s)["passed"]
    s["forecast_only_real"]["danger_window"]["mean_delta_bce"] = 0.01
    assert not gates(s)["passed"]
