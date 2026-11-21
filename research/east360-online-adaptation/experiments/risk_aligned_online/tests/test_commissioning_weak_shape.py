"""Synthetic contracts for causal inputs and matched future shape controls."""

import numpy as np
import pytest

from experiments.risk_aligned_online import commissioning_weak_shape_data_20261111 as weak


def series(length=200):
    z = np.zeros((length, 10), dtype=np.float64)
    mask = np.ones_like(z, dtype=bool)
    context = np.zeros((length, 2), dtype=np.float64)
    return z, mask, context, np.ones_like(context, dtype=bool)


def test_causal_inputs_feature_order_shape_and_pair_coordinates():
    z, mask, context, cmask = series()
    # Eight observed bins with known levels, plus a distinct Ip/Bt history.
    for b in range(8):
        z[b * 16:(b + 1) * 16] = b * np.arange(1, 11)
    context[:112] = [2, 3]
    context[112:128] = [5, 7]
    x, pair = weak.inputs(z, mask, context, cmask, np.array([127]))
    assert x.shape == (1, 186)
    assert x.dtype == pair.dtype == np.float32
    assert pair.shape == (1, 7)
    np.testing.assert_allclose(x[0, :10], 7 * np.arange(1, 11))
    shape = x[0, 10:90].reshape(8, 10)
    np.testing.assert_allclose(shape, np.broadcast_to(np.arange(8)[:, None] - 3.5, (8, 10)))
    np.testing.assert_allclose(x[0, 90:100], np.sqrt(5.25) * np.arange(1, 11))
    np.testing.assert_array_equal(x[0, 100:180], 1)
    np.testing.assert_array_equal(x[0, 180:], [5, 7, 3, 1, 1, 1])
    # The latest bin differs by 1.5 slopes from the previous two bins.
    slopes = 1.5 * np.arange(1, 11)
    groups = ((0, 1, 2), (3, 4), (6, 7, 8, 9), (5,))
    expected = [5, 7, 3] + [np.sqrt(np.mean(slopes[list(g)]**2)) for g in groups]
    np.testing.assert_allclose(pair[0], expected, rtol=1e-6)


def test_prefix_ignores_future_and_record_length():
    z, mask, context, cmask = series()
    rng = np.random.default_rng(7)
    z[:] = rng.normal(size=z.shape)
    context[:] = rng.normal(size=context.shape)
    before = weak.inputs(z, mask, context, cmask, [63, 127])
    z[128:] = 1e9
    context[128:] = -1e9
    after = weak.inputs(z[:160], mask[:160], context[:160], cmask[:160], [63, 127])
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    # Perturb the future separately for each origin, not just after both.
    for q in (63, 127):
        original = weak.inputs(z, mask, context, cmask, [q])
        altered_z, altered_context = z.copy(), context.copy()
        altered_z[q+1:] = 12345
        altered_context[q+1:] = -12345
        modified = weak.inputs(altered_z, mask, altered_context, cmask, [q])
        for a, b in zip(original, modified):
            np.testing.assert_array_equal(a, b)


def test_early_prefix_requires_eight_observations_and_pads_missing_bins():
    z, mask, context, cmask = series()
    mask[:] = False
    z[:8, 0] = 3
    mask[:8, 0] = True
    x, pair = weak.inputs(z, mask, context, cmask, [15])
    assert x[0, 0] == 3
    np.testing.assert_array_equal(x[0, 10:100], 0)
    flags = x[0, 100:180].reshape(8, 10)
    assert flags.sum() == 1 and flags[-1, 0] == 1
    np.testing.assert_array_equal(pair[0, 3:], 0)
    mask[0, 0] = False
    x, _ = weak.inputs(z, mask, context, cmask, [15])
    assert x[0, 0] == 0 and not x[0, 100:180].any()


def test_missing_and_nonfinite_values_cannot_leak_into_inputs_or_matching():
    z, mask, context, cmask = series()
    z[:, :] = np.arange(200)[:, None] / 10
    mask[112:121, 0] = False  # Only 7 observed values in the latest bin.
    cmask[112:121, 0] = False
    z[mask == 0] = np.nan
    context[cmask == 0] = np.inf
    first = weak.inputs(z, mask, context, cmask, [127])
    z[mask == 0] = -1e15
    context[cmask == 0] = 1e15
    second = weak.inputs(z, mask, context, cmask, [127])
    for a, b in zip(first, second):
        assert np.isfinite(a).all()
        np.testing.assert_array_equal(a, b)
    assert first[0][0, 0] == 0
    assert first[0][0, 100 + 7 * 10] == 0
    assert first[0][0, 180] == 0
    assert first[0][0, 182] == 0
    # Even a non-finite value marked observed must be treated as missing.
    z[121, 0] = np.nan
    out = weak.inputs(z, mask, context, cmask, [127])
    assert np.isfinite(out[0]).all() and np.isfinite(out[1]).all()


def test_feedback_known_shape_and_shift_preserve_norm_and_marginals():
    z, mask, _, _ = series()
    q = 80
    for b, value in enumerate([0, 1, 4, 2, 3]):
        z[q + 1 + 5 * b:q + 6 + 5 * b, 0] = value
        z[q + 1 + 5 * b:q + 6 + 5 * b, 3] = 2 * value
    mask[:, :] = False
    mask[q + 1:q + 26, [0, 3]] = True
    real, shifted, eligible = weak.feedback(z, mask, [q])
    assert eligible.tolist() == [True]
    assert real.shape == shifted.shape == (1, 50)
    assert real.dtype == shifted.dtype == np.float32
    r, s = real.reshape(5, 10), shifted.reshape(5, 10)
    centered = np.array([-2, -1, 2, 0, 1], dtype=float)
    np.testing.assert_allclose(r[:, 0], centered / np.sqrt(20), atol=1e-7)
    np.testing.assert_allclose(r[:, 3], centered / np.sqrt(20), atol=1e-7)
    np.testing.assert_allclose(s, np.roll(r, 2, axis=0), atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(real, axis=1), 1, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(shifted, axis=1), 1, atol=1e-7)
    np.testing.assert_allclose(np.sort(r, axis=0), np.sort(s, axis=0), atol=1e-7)


def test_feedback_exact_horizon_missingness_and_eligibility():
    z, mask, _, _ = series()
    q = 80
    for b in range(5):
        z[q + 1 + 5*b:q + 6 + 5*b, :] = b
    baseline = weak.feedback(z, mask, [q])
    z[:q+1] = -999
    z[q+26:] = 999
    same = weak.feedback(z, mask, [q])
    for a, b in zip(baseline, same):
        np.testing.assert_array_equal(a, b)
    for boundary in (q + 1, q + 25):
        changed = z.copy()
        changed[boundary, 0] += 10
        assert not np.array_equal(baseline[0], weak.feedback(changed, mask, [q])[0])
    # Exactly three samples per bin remain valid for one channel; two do not.
    mask[q+1:q+3, 0] = False
    mask[q+1:q+4, 1] = False
    z[~mask] = np.nan
    r, s, ok = weak.feedback(z, mask, [q])
    assert ok[0] and np.isfinite(r).all() and np.isfinite(s).all()
    assert np.any(r.reshape(5, 10)[:, 0])
    np.testing.assert_array_equal(r.reshape(5, 10)[:, 1], 0)
    z[~mask] = 1e12
    r2, s2, ok2 = weak.feedback(z, mask, [q])
    np.testing.assert_array_equal(r, r2)
    np.testing.assert_array_equal(s, s2)
    np.testing.assert_array_equal(ok, ok2)
    # A single observed family or a constant shape cannot qualify.
    mask[:] = False
    mask[q+1:q+26, :3] = True
    for signal in (z, np.zeros_like(z)):
        a, b, allowed = weak.feedback(signal, mask, [q])
        assert not allowed[0]
        assert not a.any() and not b.any()
    mask[q+1:q+26, 3] = True
    a, b, allowed = weak.feedback(np.zeros_like(z), mask, [q])
    assert not allowed[0] and not a.any() and not b.any()
    with pytest.raises(ValueError):
        weak.feedback(z[:q+25], mask[:q+25], [q])


def test_empty_batches_and_input_validation():
    z, mask, context, cmask = series()
    x, pair = weak.inputs(z, mask, context, cmask, [])
    assert x.shape == (0, 186) and pair.shape == (0, 7)
    real, shifted, ok = weak.feedback(z, mask, [])
    assert real.shape == shifted.shape == (0, 50) and ok.shape == (0,)
    with pytest.raises(ValueError):
        weak.inputs(z[:, :9], mask[:, :9], context, cmask, [127])
