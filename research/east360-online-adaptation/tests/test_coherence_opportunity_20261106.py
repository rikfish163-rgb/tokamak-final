import numpy as np
import pytest

from coherence_opportunity_20261106 import opportunity, ready_blocks


def test_first_crossing_opportunity_does_not_credit_older_or_later_writes():
    t = np.arange(500) / 1000
    p = np.zeros_like(t)
    p[[100, 220, 410]] = 0.85
    b = np.zeros((16, 4), dtype=bool)
    b[[2, 4, 6], 1] = True
    ready = np.ones(len(b), dtype=bool)
    d = opportunity(t, p, False, None, b, ready, 0.725)
    assert d["status"] == "fp" and d["first_index"] == 100
    assert not d["first_active"]["coherent"]
    assert d["ever_active"]["coherent"]
    b[3, 1] = True
    assert opportunity(t, p, False, None, b, ready, 0.725)["first_active"]["coherent"]


def test_timing_excludes_strict_30_and_keeps_above_three_seconds_separate():
    t = np.arange(4032) / 1000
    p = np.zeros_like(t)
    p[4000] = 0.9
    b = np.zeros((126, 4), dtype=bool)
    b[[0, 125], 1] = True
    r = np.ones(len(b), dtype=bool)
    y = opportunity(t, p, True, 4.030, b, r, 0.725)
    assert y["status"] == "tardy" and y["lead_ms"] == 30
    assert y["strict_any"]["coherent"] and not y["reasonable_any"]["coherent"]
    b[124, 1] = True
    assert opportunity(t, p, True, 4.030, b, r, 0.725)["reasonable_any"]["coherent"]


def test_ready_requires_nine_earlier_complete_supported_blocks():
    support = np.ones(21, dtype=bool)
    support[10] = False
    r = ready_blocks(support, 21 * 32 + 3)
    np.testing.assert_array_equal(np.flatnonzero(r), [9, 10, 20, 21])
    with pytest.raises(ValueError, match="complete"):
        ready_blocks(support, 128)
