import pytest

from experiments.risk_aligned_online.east_decoder_calibration_20261102.parameter_verify import (
    check_arm,
    gate,
)


def test_receipt_rejection_and_fixed_trial_order_are_checked():
    r = dict(
        maturity=159,
        origin=134,
        sealed_through=159,
        first_affected=160,
        forecast_sha256="a" * 64,
        mask_sha256="b" * 64,
        target_sha256="c" * 64,
        observed=23,
        accepted=True,
        reason="improved",
        loss_before=1.0,
        loss_proposal=0.9,
        gradient_norm=0.0,
        step_norm=0.0,
        gradient256=[0.0] * 256,
        state256_after=[0.0] * 256,
        chosen_fraction=0.25,
        tested_steps=[
            dict(fraction=1.0, diagnostic_loss=1.0, protection_loss=0.0, objective=1.0),
            dict(
                fraction=0.25, diagnostic_loss=0.9, protection_loss=0.0, objective=0.9
            ),
        ],
    )
    assert check_arm([r], 160, False)["accepted"] == 1
    r["tested_steps"][0]["objective"] = 0.95
    r["tested_steps"][0]["diagnostic_loss"] = 0.95
    with pytest.raises(ValueError):
        check_arm([r], 160, False)


def test_matched_equal_feedback_scores_do_not_support_unique_real_gain():
    scores = dict(
        static=dict(tp=125, fp=35),
        real=dict(tp=126, fp=34),
        reverse=dict(tp=126, fp=34),
        hold=dict(tp=126, fp=34),
    )
    assert gate(scores)["operational"] and not gate(scores)["passed"]
