from experiments.risk_aligned_online.source_anticipation_20261104.evaluate import (
    aggregate,
    gate,
)


def test_eval_keeps_future_event_and_original_point_targets_separate():
    rows = [
        dict(shot=1, stratum="safe", future_delta=-0.01, point_delta=-0.01),
        dict(
            shot=2,
            stratum="disruptive_30_to_125ms",
            future_delta=-0.03,
            point_delta=0.04,
        ),
        dict(
            shot=2,
            stratum="disruptive_earlier_than_125ms",
            future_delta=0.0,
            point_delta=0.0,
        ),
        dict(shot=3, stratum="disruptive_0_to_30ms", future_delta=0.0, point_delta=0.0),
    ]
    out = aggregate(rows)
    assert out["balanced"] == -0.02
    assert out["disruptive_30_to_125ms"]["point_delta"] == 0.04


def test_source_gate_requires_real_feedback_and_both_time_controls():
    def item(delta):
        return {
            "balanced": delta,
            **{
                g: {"future_delta": delta}
                for g in (
                    "safe",
                    "disruptive_30_to_125ms",
                    "disruptive_earlier_than_125ms",
                    "disruptive_0_to_30ms",
                )
            },
        }

    table = {
        view: {
            head: {mode: item(-0.001) for mode in ("real", "reverse", "hold")}
            for head in (
                "native_reference",
                "point_contract",
                "anticipation125",
                "time_shuffled125",
            )
        }
        for view in ("native", "no_MNM_CIII")
    }
    assert gate(table)["passed"] is False
    table["no_MNM_CIII"]["anticipation125"]["real"] = item(-0.002)
    assert gate(table)["passed"] is True
    table["no_MNM_CIII"]["anticipation125"]["real"]["disruptive_earlier_than_125ms"][
        "future_delta"
    ] = 0.01
    assert gate(table)["passed"] is False
