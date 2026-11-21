import pytest

from experiments.risk_aligned_online.native_token_probe_20261104 import scoring


def test_primary_is_native_absolute_and_controls_are_own_configuration():
    def arm(delta):
        return {
            **{
                g: {"mean_delta_bce": delta}
                for g in ("safe", "disruptive", "danger_window")
            },
            "balanced": delta,
        }

    groups = {
        cfg: {m: arm(-0.01) for m in ("static", "real", "reverse", "hold")}
        for cfg in scoring.CONFIGS
    }
    groups["source_decoder_centered"]["real"] = arm(-0.02)
    result = scoring.gates(groups)
    assert result["primary_configuration"] == "native_absolute"
    assert result["primary_gate"]["passed"] is False
    assert result["configuration_gates"]["source_decoder_centered"]["passed"] is True
    groups["native_absolute"]["real"] = arm(-0.02)
    assert scoring.gates(groups)["primary_gate"]["passed"] is True


def test_static_reference_needs_every_coordinate_and_exact_logit_match():
    refs = [
        {
            "stem": "original",
            "mode": "static",
            "shot": 1,
            "query": 200,
            "after_logit": 0.5,
        }
    ]
    rows = [{"shot": 1, "query": 200, "before_logit": 0.5, "after_logit": 0.6}]
    assert scoring.check_static_reference(rows, refs)["max_abs_logit_difference"] == 0
    rows[0]["before_logit"] += 1e-7
    with pytest.raises(AssertionError):
        scoring.check_static_reference(rows, refs)
