import numpy as np
import torch

from experiments.risk_aligned_online.critical_shot_meta_20261102.run import (
    export_traces,
)


def test_trace_export_keeps_arrays_and_explicit_absent_gradient_separate():
    zero = torch.zeros(4, 64)
    event = dict(
        origin=134,
        maturity=159,
        first_affected=160,
        before=zero,
        after=zero.clone(),
        forecast_sha256="f",
        mask_sha256="m",
        target_sha256="t",
        observed=0,
        receipt=dict(
            accepted=False, reason="no_target", step_norm=0.0, gradient_norm=0.0
        ),
    )
    arrays, metadata = export_traces({"head": [event], "empty": []}, {})
    assert arrays["head_before"].shape == (1, 4, 64)
    assert arrays["empty_gradient"].shape == (0, 256)
    assert metadata["updates"]["head"][0]["gradient_computed"] is False
    np.testing.assert_array_equal(arrays["head_gradient"], 0)
    assert "before" not in metadata["updates"]["head"][0]


def test_computed_gradient_is_not_silently_replaced():
    zero = torch.zeros(4, 64)
    event = dict(
        origin=134,
        maturity=159,
        first_affected=160,
        before=zero,
        after=zero + 0.01,
        forecast_sha256="f",
        mask_sha256="m",
        target_sha256="t",
        observed=575,
        receipt=dict(
            accepted=True,
            reason="improved",
            gradient256=[2.0] * 256,
            state256_after=[0.01] * 256,
            step_norm=0.16,
            gradient_norm=32.0,
        ),
    )
    arrays, metadata = export_traces({"head": [event]}, {})
    assert metadata["updates"]["head"][0]["gradient_computed"] is True
    np.testing.assert_array_equal(arrays["head_gradient"], 2)
    assert "gradient256" not in metadata["updates"]["head"][0]
