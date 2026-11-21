import numpy as np
import pytest

from verify_paired_diagnostic_update_20261106 import verify_prediction


def sample():
    n = 70
    z = np.zeros((n, 3, 6), dtype=np.float32)
    z[64:, :, 5] = 0.1
    pred = dict(
        time=np.arange(n) / 1000,
        logits=z,
        maturity=np.array([63]),
        visible=np.array([True]),
        common=np.array([True]),
        active=np.arange(n) >= 64,
        norms=np.full((1, 3, 4), 0.2, dtype=np.float32),
        cosine=np.zeros((1, 3, 3)),
        offsets=np.full((1, 3), 0.1, dtype=np.float32),
        auxiliary=np.zeros((1, 3, 2)),
    )
    return pred, np.ones((n, 23), dtype=bool)


def test_archive_uses_next_observed_point_and_a_constant_score_control():
    p, mask = sample()
    verify_prediction(p, p["time"], mask, enabled=True)
    p["active"][63] = True
    with pytest.raises(ValueError, match="boundary"):
        verify_prediction(p, p["time"], mask, enabled=True)
    p["active"][63] = False
    p["logits"][67, 0, 5] = 0
    with pytest.raises(ValueError, match="anchor"):
        verify_prediction(p, p["time"], mask, enabled=True)


def test_mask_recomputation_catches_cross_channel_missingness_and_cold_writes():
    p, mask = sample()
    with pytest.raises(ValueError, match="support"):
        verify_prediction(p, p["time"], mask, enabled=False)
    mask[:, :14] = False
    mask[0:32, 19:] = False
    with pytest.raises(ValueError, match="support"):
        verify_prediction(p, p["time"], mask, enabled=True)
