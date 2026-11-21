import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.local_readout_20261029.learning import (
    Head,
    History,
    solve,
    forward_features,
    feedback,
    predict,
)
from experiments.risk_aligned_online.tests.test_qk_matched_head import fixture


def make_head():
    torch.manual_seed(31)
    active = torch.ones(23, dtype=torch.bool)
    active[[2, 10]] = False
    return Head(
        torch.zeros(23, 64, dtype=torch.float64),
        torch.ones(23, 64, dtype=torch.float64),
        torch.randn(23, 65, 25, dtype=torch.float64) * 0.01,
        active,
    )


def test_prior_ridge_matches_direct_normal_equations_and_missing_channel():
    h = make_head()
    rng = np.random.default_rng(21)
    history = History()
    for i in range(6):
        history.add(
            torch.tensor(rng.normal(size=64)),
            torch.tensor(rng.normal(size=(25, 23))),
            h.active,
            127 + 25 * i,
        )
    fitted, counts = solve(h, history, "true")
    x = torch.stack([r[0] for r in history.rows]).numpy()
    x = np.column_stack([x, np.ones(len(x))])
    y = torch.stack([r[1] for r in history.rows]).numpy()[:, :, 0]
    p = h.coef[0].numpy()
    expected = np.linalg.solve(x.T @ x + np.eye(65), x.T @ y + p)
    np.testing.assert_allclose(fitted.coef[0].numpy(), expected, atol=1e-11, rtol=1e-10)
    assert counts[0] == 6 and counts[2] == 0
    torch.testing.assert_close(fitted.coef[2], h.coef[2], rtol=0, atol=0)
    with pytest.raises(ValueError):
        history.add(history.rows[-1][0], history.rows[-1][1], h.active, 127)


def test_history_truncation_and_current_target_cannot_fit_decoder_early():
    h = make_head()
    history = History()
    for i in range(20):
        history.add(torch.zeros(64), torch.zeros(25, 23), h.active, 127 + 25 * i)
    assert len(history.rows) == 16 and history.rows[0][3] == 227
    net, s = fixture()
    signal = {k: s[k][:256] for k in ("values", "available")}
    delta, receipt, heads = feedback(net, signal, None, h, History())
    assert receipt["calibration_last_target"] < receipt["gradient_first_target"]
    assert receipt["gradient_last_target"] == 255
    assert receipt["fit_counts_max"] == 2 and receipt["history_rows_after"] == 4
    for name, d in delta.items():
        assert torch.isfinite(d).all() and d.norm() <= 0.050001
    changed = {k: v.copy() for k, v in signal.items()}
    changed["values"][206:256] += 10
    _, _, other = feedback(net, changed, None, h, History())
    for name in heads:
        torch.testing.assert_close(heads[name].coef, other[name].coef, rtol=0, atol=0)


def test_auxiliary_head_does_not_change_risk_and_future_does_not_rewrite_prefix():
    net, s = fixture()
    h = make_head()
    signal = {k: s[k][:400] for k in ("values", "available")}
    with torch.no_grad():
        x = torch.as_tensor(signal["values"][:128])[None]
        a = torch.as_tensor(signal["available"][:128])[None]
        z, f = forward_features(net, x, a, x.new_zeros(1, 8192), None)
        first = z.clone()
        h.predict(f)
        h.coef *= 3
        z, _ = forward_features(net, x, a, x.new_zeros(1, 8192), None)
    torch.testing.assert_close(z, first, rtol=0, atol=0)
    curves, info, aux = predict(net, signal, None, h)
    changed = {k: v.copy() for k, v in signal.items()}
    changed["values"][320:] += 12
    curves2, _, _ = predict(net, changed, None, h)
    for name in curves:
        np.testing.assert_array_equal(curves[name][:320], curves2[name][:320])
        np.testing.assert_array_equal(curves[name][:256], curves["cold"][:256])
    assert info["local_readout"]["opportunities"] == 2
    assert all(r["clock"] + 25 <= 400 for r in aux)
