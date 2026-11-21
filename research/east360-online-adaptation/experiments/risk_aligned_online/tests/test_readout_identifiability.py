import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.readout_identifiability_20261029.learning import (
    ridge,
    predict,
    examples,
    fit_probes,
    predictions,
    scores,
)
from experiments.risk_aligned_online.tests.test_qk_matched_head import fixture


def test_weighted_ridge_agrees_with_direct_normal_equations_and_constant_column():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(20, 4))
    x[:, 3] = 2
    y = rng.normal(size=(20, 3))
    ids = np.repeat([1, 2, 3], [2, 8, 10])
    fit = ridge(x, y, ids, 0.01, device="cpu")
    w = np.array([1 / np.sum(ids == i) for i in ids])
    w /= w.sum()
    mu = np.sum(w[:, None] * x, axis=0)
    scale = np.sqrt(np.sum(w[:, None] * (x - mu) ** 2, axis=0))
    scale[3] = 1
    xx = (x - mu) / scale
    ym = np.sum(w[:, None] * y, axis=0)
    coef = np.linalg.solve(
        xx.T @ (w[:, None] * xx) + 0.01 * np.eye(4), xx.T @ (w[:, None] * (y - ym))
    )
    np.testing.assert_allclose(predict(fit, x), xx @ coef + ym, rtol=1e-10, atol=1e-10)
    assert fit["equation_relative_residual"] < 1e-10
    assert fit["solver"] == "primal" and fit["scale"][3] == 1
    other = ridge(x[:3], y[:3], ids[:3], 0.01, device="cpu")
    assert other["solver"] == "dual"
    with pytest.raises(ValueError):
        ridge(x, y, ids, 0, device="cpu")


def test_features_match_actual_forward_and_future_is_not_an_input():
    net, s = fixture()
    signal = {k: s[k] for k in ("values", "available")}
    data = examples(net, signal, [256, 384], None)
    with torch.no_grad():
        h = torch.as_tensor(data["all_tokens"])
        p = net.base.forecast_head(h).reshape(2, 25, 23).numpy()
    np.testing.assert_allclose(p, data["original_forecast"], atol=1e-6)
    assert data["raw8"].shape == (2, 368) and data["last_token"].shape == (2, 64)
    assert not data["mask"][:, 10].any()
    changed = {k: v.copy() for k, v in signal.items()}
    changed["values"][256:] += 100
    # Hold batch shape fixed: GEMM accumulation can change across batch sizes.
    original = examples(net, signal, [256], None)
    second = examples(net, changed, [256], None)
    for name in ("all_tokens", "last_token", "raw8"):
        np.testing.assert_array_equal(original[name], second[name])
    assert not np.array_equal(original["future"], second["future"])
    with pytest.raises(ValueError):
        examples(net, dict(signal, outcome=1), [256], None)


def test_unobserved_channel_is_not_fitted_or_silently_scored():
    rng = np.random.default_rng(8)
    data = dict(
        ids=np.array([1, 1, 2, 2]),
        mask=np.zeros((4, 23), bool),
        current=rng.normal(size=(4, 23)),
        future=rng.normal(size=(4, 25, 23)),
        **{
            view: rng.normal(size=(4, 3))
            for view in ("all_tokens", "last_token", "raw8")
        },
    )
    data["mask"][:, 0] = True
    fits = fit_probes(data, device="cpu")
    assert all(fits[v][0] is not None and fits[v][1] is None for v in fits)
    pred = predictions(fits, data)
    assert all(np.isnan(p[:, :, 1:]).all() for p in pred.values())
    result = scores(pred, data, data["mask"][0])
    assert all(r["shots"] == 2 and r["channel_counts"][1] == 0 for r in result.values())
    data["mask"][:, 1] = True
    with pytest.raises(ValueError, match="prediction missing"):
        scores(pred, data, np.ones(23, bool))


def test_scores_weight_shots_not_total_windows():
    data = dict(
        ids=np.array([1, 1, 1, 2]),
        mask=np.ones((4, 23), bool),
        current=np.zeros((4, 23)),
        future=np.zeros((4, 25, 23)),
    )
    pred = np.zeros((4, 25, 23))
    pred[3] = 2
    result = scores({"forecast": pred}, data, np.ones(23, bool))["forecast"]
    assert result["mse"] == 2 and result["smooth_l1"] == 0.75
    assert result["shots"] == 2 and result["windows"] == 4


def test_score_accumulation_uses_same_precision_for_float32_and_ridge_outputs():
    data = dict(
        ids=np.ones(1000, int),
        mask=np.ones((1000, 23), bool),
        current=np.full((1000, 23), 0.1, np.float32),
        future=np.full((1000, 25, 23), 0.1, np.float32),
    )
    pred = np.full((1000, 25, 23), 0.2, np.float32)
    r = scores({"forecast": pred}, data, np.ones(23, bool))["forecast"]
    expected = ((float(np.float32(0.2)) - float(np.float32(0.1))) ** 2) * 25000 / 2
    np.testing.assert_allclose(r["channel_sums"]["smooth_l1"], expected, rtol=1e-12)
