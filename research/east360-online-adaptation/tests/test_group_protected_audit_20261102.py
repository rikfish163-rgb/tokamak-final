import copy

from experiments.risk_aligned_online.group_protected_meta_20261102.audit import (
    compatible,
    gate,
)


def test_compatible_summary_uses_shots_and_ignores_source_negative_early():
    rows = [
        dict(shot=1, group="danger", lead_ms=31.0, delta_bce=-1.0),
        dict(shot=1, group="danger", lead_ms=40.0, delta_bce=-1.0),
        dict(shot=2, group="danger", lead_ms=125.0, delta_bce=3.0),
        dict(shot=3, group="early", lead_ms=60.0, delta_bce=-100.0),
    ]
    assert compatible(rows) == dict(n_shots=2, n_points=3, mean_delta_bce=1.0)


def test_independent_gate_preserves_all_declared_requirements():
    v = dict(
        safe=dict(mean_delta_bce=-0.1),
        danger=dict(mean_delta_bce=-0.1),
        compatible_danger=dict(mean_delta_bce=-0.1),
        balanced=-0.1,
    )
    s = {
        t: {
            a: copy.deepcopy(v)
            for a in (
                "worst_risk_real",
                "mean_risk_real",
                "worst_risk_reverse",
                "worst_risk_hold",
            )
        }
        for t in ("native", "no_MNM_CIII")
    }
    old = {"no_MNM_CIII": {"true_real": copy.deepcopy(v)}}
    assert not gate(s, old)["passed"]
    s["no_MNM_CIII"]["worst_risk_real"]["balanced"] = -0.2
    assert gate(s, old)["passed"]
    s["native"]["worst_risk_real"]["danger"]["mean_delta_bce"] = 2e-6
    assert not gate(s, old)["passed"]
