import numpy as np
import pytest

from experiments.risk_aligned_online.commissioning_calibration_20261108 import (
    held_safe_cutoffs,
)


def test_cutoffs_use_held_safe_shot_peaks_and_handle_ties_in_float32():
    peaks = np.array([.20, .21, .22, .23, .25, .25, .29, .29, .31, .33, .45, .45],
                     np.float32)
    cutoffs = held_safe_cutoffs(peaks)
    assert cutoffs["0.1"] == cutoffs["0.15"]
    for key, threshold in cutoffs.items():
        assert (peaks >= threshold).mean() <= float(key)
        assert threshold == np.nextafter(np.float32(.45), np.float32(np.inf))


def test_cutoffs_change_with_budget_when_safe_peaks_are_distinct():
    peaks = np.linspace(.3, .6, 12, dtype=np.float32)
    thresholds = held_safe_cutoffs(peaks)
    assert np.count_nonzero(peaks >= thresholds["0.2"]) == 1
    assert np.count_nonzero(peaks >= thresholds["0.1"]) == 0
    with pytest.raises(ValueError, match="12"):
        held_safe_cutoffs(peaks[:8])
