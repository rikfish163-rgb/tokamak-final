import pytest
import torch

from experiments.risk_aligned_online.east_decoder_calibration_20261102.direction_audit import (
    direction_statistics,
    protection_terms,
    summarize,
)


def test_descent_direction_can_harm_risk_and_parameter_anchor_is_not_hidden_budget():
    phi = torch.ones(1, 16, 4, dtype=torch.float64) * 0.01
    pack = dict(phi=phi, valid=torch.ones(1, 17, dtype=torch.bool))
    diagnostic = torch.ones(4, 64, dtype=torch.float64)
    direction, result = direction_statistics(pack, diagnostic, -diagnostic)
    assert result["diagnostic_slope"] < 0 < result["risk_slope"]
    assert result["step_hidden_rms"] <= 0.05 + 1e-12
    assert result["risk_cosine"] > 0.99
    terms = protection_terms(pack, direction)
    assert terms["parameter_mean_square"] > terms["hidden_mean_square"]
    with pytest.raises(ValueError):
        direction_statistics(pack, diagnostic[:, :2], diagnostic)


def test_zero_gradient_is_not_invented_as_an_alignment():
    pack = dict(phi=torch.ones(1, 16, 4), valid=torch.ones(1, 17, dtype=torch.bool))
    zero = torch.zeros(4, 64)
    direction, result = direction_statistics(pack, zero, torch.ones_like(zero))
    assert torch.count_nonzero(direction) == 0
    assert result["risk_cosine"] is None
    assert result["risk_slope"] == result["diagnostic_slope"] == 0


def test_diagnostic_summary_weights_shots_not_number_of_queries():
    def row(shot, slope, delta, accepted, without_anchor):
        return dict(
            shot=shot,
            group="safe",
            risk_slope=slope,
            steps=[
                dict(
                    fraction=1.0,
                    delta_risk=delta,
                    accepted=accepted,
                    diagnostic_improved=without_anchor,
                )
            ],
        )

    rows = [
        row(1, -1, -0.2, True, True),
        row(1, -1, -0.4, True, True),
        row(2, 1, 0.1, False, True),
    ]
    result = summarize(rows)
    assert result["shots"] == 2 and result["episodes"] == 3
    assert result["shot_macro_beneficial_direction_fraction"] == 0.5
    assert result["steps"]["1.0"]["shot_macro_delta_risk"] == pytest.approx(-0.1)
    assert result["steps"]["1.0"]["parameter_anchor_blocks"] == 1
