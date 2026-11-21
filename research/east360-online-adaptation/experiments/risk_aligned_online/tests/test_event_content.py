import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.event_prefix_probe import causal_summary
from experiments.risk_aligned_online.event_content_20261023.probe import select_features, independent_f1


def test_exact_information_blocks_and_no_mutation():
    raw = np.arange(230, dtype=np.float32).reshape(2, 115)
    before = raw.copy()
    for mode, blocks in {'mask': [4], 'mean': [1, 4], 'moments': [1, 2, 4],
                         'change': [3, 4], 'variation': [2, 3, 4], 'full': list(range(5))}.items():
        selected = select_features(raw, mode)
        for block in range(5):
            sl = slice(block * 23, (block + 1) * 23)
            np.testing.assert_array_equal(selected[:, sl], raw[:, sl] if block in blocks else 0)
    np.testing.assert_array_equal(raw, before)


def test_time_reversal_preserves_moments_but_reverses_change():
    x = torch.arange(8, dtype=torch.float32)[None, :, None].expand(2, 8, 23)
    mask = torch.ones_like(x, dtype=torch.bool)
    a = causal_summary(x, mask, torch.zeros(23), torch.ones(23)).numpy()
    b = causal_summary(x.flip(1), mask, torch.zeros(23), torch.ones(23)).numpy()
    np.testing.assert_allclose(select_features(a, 'moments'), select_features(b, 'moments'), atol=1e-6)
    np.testing.assert_allclose(a[:, 69:92], -b[:, 69:92], atol=1e-6)


def test_unknown_mode_shape_and_nonfinite_rejected():
    with pytest.raises(ValueError): select_features(np.zeros((3, 114)), 'mean')
    with pytest.raises(ValueError): select_features(np.zeros((3, 115)), 'unknown')
    with pytest.raises(ValueError): select_features(np.full((3, 115), np.nan), 'full')


def test_event_equal_f1_is_not_endpoint_equal():
    y = np.array([0, 0, 0, 0, 1, 1]); pred = np.array([0, 0, 0, 0, 0, 1])
    events = np.array([10, 10, 10, 10, 11, 12])
    matrix, f1 = independent_f1(y, pred, events)
    np.testing.assert_array_equal(matrix[:2, :2], [[1, 0], [1, 1]])
    assert f1 == pytest.approx((2 / 3 + 2 / 3) / 6)
