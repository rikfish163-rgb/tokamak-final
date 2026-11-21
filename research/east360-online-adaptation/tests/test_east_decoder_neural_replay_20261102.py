import numpy as np
import pytest

from experiments.risk_aligned_online.east_decoder_calibration_20261102.neural_replay import (
    state_at,
)


def test_saved_update_cannot_affect_maturity_or_an_earlier_prediction():
    records = [dict(maturity=159, first_affected=160, state256_after=[0.25] * 256)]
    assert np.array_equal(state_at(records, 159), np.zeros((4, 64), np.float32))
    assert np.array_equal(state_at(records, 160), np.full((4, 64), 0.25, np.float32))
    with pytest.raises(ValueError):
        state_at(records, -1)
    with pytest.raises(ValueError):
        state_at(
            [dict(maturity=159, first_affected=159, state256_after=[0.0] * 256)], 160
        )
