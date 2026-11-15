import numpy as np

from experiments.risk_aligned_online.source_missing_diagnostics_20261103 import (
    diagnostic_view,
    score_shot,
)


def test_views_mask_only_declared_diagnostics_and_preserve_raw_values():
    x = np.ones((3, 23), np.float32)
    x[1, 1] = -10
    a = np.ones_like(x, dtype=bool)
    s = dict(values=x, available=a)
    original = diagnostic_view(s, "qualified_native")
    contract = diagnostic_view(s, "contract_native")
    drop = diagnostic_view(s, "no_MNM_CIII")
    assert original["available"].all() and a.all()
    assert contract["available"][0, 1] and not contract["available"][1, 1]
    assert not drop["available"][:, [1, 2]].any()
    for view in (original, contract, drop):
        np.testing.assert_array_equal(view["values"], x)
    single = diagnostic_view(s, "no_MNM")
    assert not single["available"][:, 2].any()
    assert single["available"][0, 1]


def test_jtext_timing_uses_hdf_down_and_strict10():
    row = score_shot(11, np.array([0.99]), np.array([0.8]), True, 1.0, 0.7)
    assert row["alarm"] and row["lead_ms"] == 10 and not row["timely"]
    row = score_shot(11, np.array([0.989, 0.99]), np.array([0.8, 0.9]), True, 1.0, 0.7)
    assert row["lead_ms"] == 11 and row["timely"]
    safe = score_shot(11, np.array([0.989]), np.array([0.8]), False, None, 0.7)
    assert safe["alarm"] and safe["lead_ms"] is None and not safe["timely"]
