import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.window_history_diagnostic_20261101.core import (
    history_view,
    first_alarm_index,
    timing_bin,
)


def fixture_window():
    x = torch.arange(2 * 128 * 23, dtype=torch.float32).reshape(2, 128, 23)
    m = torch.ones_like(x, dtype=torch.bool)
    m[:, :120] = False
    m[0, 122, 3] = False
    m[0, 127, 4] = False
    m[:, :, 6] = False
    x[~m] = -17
    return x, m


def test_reverse_only_real_past_preserves_latest_and_padding():
    x, m = fixture_window()
    out, mask = history_view(x, m, [7, 400], "reverse_past")
    assert torch.equal(out[0, :120], x[0, :120])
    assert torch.equal(mask[0, :120], m[0, :120])
    assert torch.equal(out[:, -1], x[:, -1])
    assert torch.equal(mask[:, -1], m[:, -1])
    assert torch.equal(out[0, 120:127], x[0, 120:127].flip(0))
    assert torch.equal(mask[0, 120:127], m[0, 120:127].flip(0))
    assert torch.equal(out[1, :127], x[1, :127].flip(0))


@pytest.mark.parametrize("mode", ["last_observed", "mean_observed"])
def test_constant_views_keep_mask_and_unavailable_values(mode):
    x, m = fixture_window()
    out, mask = history_view(x, m, [7, 400], mode)
    assert torch.equal(mask, m)
    assert torch.equal(out[~m], x[~m])
    for b in range(2):
        for c in range(23):
            observed = x[b, m[b, :, c], c]
            if len(observed):
                expected = (
                    observed[-1]
                    if mode == "last_observed"
                    else observed.double().mean().float()
                )
                assert torch.all(out[b, m[b, :, c], c] == expected)


def test_identity_has_no_mutation_or_shared_output_storage():
    x, m = fixture_window()
    original = x.clone()
    out, mask = history_view(x, m, [7, 400], "original")
    assert torch.equal(out, x) and torch.equal(mask, m)
    out.zero_()
    assert torch.equal(x, original)


def test_one_observation_and_all_missing_are_defined():
    x = torch.zeros(2, 128, 23)
    m = torch.zeros_like(x, dtype=torch.bool)
    x[0, -1, 0], m[0, -1, 0] = 3, True
    for mode in ("original", "reverse_past", "last_observed", "mean_observed"):
        out, mask = history_view(x, m, [0, 0], mode)
        assert torch.equal(out, x) and torch.equal(mask, m)


def test_invalid_endpoint_or_mode_rejected():
    x, m = fixture_window()
    with pytest.raises(ValueError):
        history_view(x, m, [7], "original")
    with pytest.raises(ValueError):
        history_view(x, m, [-1, 7], "original")
    with pytest.raises(ValueError):
        history_view(x, m, [7, 400], "unknown")


def test_first_alarm_and_bins_include_exact_threshold():
    assert first_alarm_index(np.array([0.1, 0.5, 0.9]), 0.5) == 1
    assert first_alarm_index(np.array([0.1, 0.2]), 0.5) is None
    assert [timing_bin(i) for i in (0, 15, 16, 127, 128)] == [
        "before16",
        "before16",
        "16to127",
        "16to127",
        "from128",
    ]
