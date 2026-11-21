"""Synthetic causality, matching, and readout tests for persistent motion."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import joint_lora_persistent_motion_probe_20261104 as probe


def test_four_bins_short_and_long_have_same_per_sample_trend():
    values = np.arange(32, dtype=float)[:, None]
    mask = np.ones_like(values, dtype=bool)
    result = probe.motion_features(values, mask, np.array([0, 1, 2, 3]))
    assert result["common"].tolist() == [True]
    assert result["short"][0] == pytest.approx(1.0)
    assert result["long"][0] == pytest.approx(1.0)
    assert result["permuted"][0] == pytest.approx(result["long"][0])
    np.testing.assert_allclose(result["long_bins"][:, 0], [3.5, 11.5, 19.5, 27.5])


def test_masks_are_joint_and_missing_values_never_become_observed():
    values = np.tile(np.arange(32, dtype=float)[:, None], (1, 2))
    mask = np.ones_like(values, dtype=bool)
    mask[:5, 0] = False  # first 8-point bin has fewer than half observed
    values[:5, 0] = 1e20
    mask[-1, 1] = False  # maturity point is not observed
    result = probe.motion_features(values, mask, np.array([2, 0, 3, 1]))
    assert not result["common"].any()
    for name in ("short", "long", "permuted"):
        assert np.array_equal(result[name], [0, 0])


def test_order_control_preserves_long_bin_medians_and_support():
    values = np.column_stack((np.arange(32), np.arange(32) ** 2)).astype(float)
    mask = np.ones_like(values, dtype=bool)
    perm = np.array([2, 0, 3, 1])
    result = probe.motion_features(values, mask, perm)
    np.testing.assert_array_equal(result["permuted_bins"], result["long_bins"][perm])
    np.testing.assert_array_equal(np.sort(result["permuted_bins"], axis=0), np.sort(result["long_bins"], axis=0))
    assert result["common"].all()
    with pytest.raises(ValueError, match="permutation"):
        probe.motion_features(values, mask, np.array([0, 0, 1, 2]))


def test_history_is_causal_and_not_affected_by_future_mutation():
    values = np.arange(100 * 3, dtype=float).reshape(100, 3)
    mask = np.ones_like(values, dtype=bool)
    before = probe.history_motion(values, mask, 63, np.array([1, 3, 0, 2]))
    values[64:] = -1e10
    mask[64:] = False
    after = probe.history_motion(values, mask, 63, np.array([1, 3, 0, 2]))
    for name in ("short", "long", "permuted", "common"):
        np.testing.assert_array_equal(before[name], after[name])
    with pytest.raises(ValueError, match="history"):
        probe.history_motion(values, mask, 30, np.array([0, 1, 2, 3]))


def test_permutation_is_deterministic_by_shot_and_maturity():
    p = probe.bin_permutation(74087, 135)
    np.testing.assert_array_equal(p, probe.bin_permutation(74087, 135))
    assert sorted(p.tolist()) == [0, 1, 2, 3]


def test_shot_balanced_weights_and_auc_with_ties():
    rows = [dict(group="safe", shot=1), dict(group="safe", shot=1),
            dict(group="safe", shot=2), dict(group="earlier", shot=3),
            dict(group="positive", shot=3)]
    weights = probe.balanced_weights(rows)
    np.testing.assert_allclose(weights, [1/12, 1/12, 1/6, 1/3, 1/3])
    assert probe.weighted_auc(np.array([0, 0, 1]), np.array([0., 1., 1.]), np.ones(3)) == pytest.approx(.75)


def test_fixed_readout_uses_native_offset_and_zero_features_do_not_change_fit():
    y = np.array([0, 1, 0, 1], dtype=float)
    offset = np.array([-1., 1., -.5, .5])
    weights = np.full(4, .25)
    base = np.array([[0.], [1.], [.1], [.9]])
    m0 = probe.fit_readout(base, offset, y, weights)
    m1 = probe.fit_readout(np.column_stack((base, np.zeros(4))), offset, y, weights)
    np.testing.assert_allclose(probe.predict_readout(m0, base, offset),
                               probe.predict_readout(m1, np.column_stack((base, np.zeros(4))), offset), atol=1e-8)
    assert m1["beta"][-1] == 0.0


def test_v5_verification_checks_truth_clock_and_stat():
    meta = dict(shot=123, disruptive=True, down=1.0, samples=3, start=0.0,
                rate=1000.0, end=.002, size=20, mtime_ns=99)
    stat = SimpleNamespace(st_size=20, st_mtime_ns=99)
    probe.verify_v5(123, np.array([0., .001, .002]), True, 1.0, stat, meta)
    with pytest.raises(ValueError, match="v5"):
        probe.verify_v5(123, np.array([0., .001, .002]), False, None, stat, meta)
    with pytest.raises(ValueError, match="v5"):
        probe.verify_v5(123, np.array([0., .001, .0021]), True, 1.0, stat, meta)


def test_plan_locks_common_maturity_offset_and_no_test_fit():
    plan = probe.fixed_plan(list(range(1, 21)), list(range(21, 101)), {})
    assert plan["train_indices"] == [1, 20] and plan["test_indices"] == [21, 100]
    assert plan["window_samples"] == [8, 32] and plan["bins"] == 4
    assert plan["future_offset_samples"] == 24 and not plan["test_labels_fit"]
    assert plan["lambda"] == probe.LAMBDA and not plan["independent_validation"]


def test_real_extraction_path_uses_only_causal_128_and_shared_masks():
    class Native(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("center", torch.zeros(23))
            self.register_buffer("scale", torch.ones(23))

        def forward(self, values, available):
            return {"logit": torch.zeros(len(values))}

    values = np.arange(200 * 23, dtype=np.float32).reshape(200, 23) / 100
    available = np.ones_like(values, dtype=bool)
    available[:, 2] = False
    signal = dict(values=values, available=available)
    case = dict(origin=127, arrival=135, future=159, label=0, group="safe", lead_ms=None)
    before = probe.extract_case(Native(), signal, case, 74087, np.full(200, .5))
    values[136:] = -1e9
    available[136:] = False
    after = probe.extract_case(Native(), signal, case, 74087, np.full(200, .5))
    assert before["maturity"] == 135 and before["future"] == 159
    assert len(before["endpoint"]) == len(probe.CHANNEL_NAMES) == 21
    for key in ("endpoint", "endpoint_mask", "short", "long", "permuted", "common"):
        np.testing.assert_array_equal(before[key], after[key])
    assert "MNM" not in probe.CHANNEL_NAMES and "ne_nG" in probe.CHANNEL_NAMES


def test_equal_arm_logits_give_equal_auc_and_shot_paired_ties():
    rows = [dict(shot=1, group="safe", label=0), dict(shot=2, group="earlier", label=0),
            dict(shot=2, group="positive", label=1)]
    logits = {arm: np.array([-.4, -.2, .4]) for arm in probe.ARMS}
    result = probe.summarize(rows, logits)
    assert result["arms"]["native"]["auc"]["balanced_all"] == 1.0
    assert result["groups"]["safe"]["short_long_vs_short"]["ties"] == 1
