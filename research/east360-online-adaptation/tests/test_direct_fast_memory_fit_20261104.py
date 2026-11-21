import pytest
import torch

from experiments.risk_aligned_online.direct_fast_memory_20261104.fit import fit_initial


def test_masked_ridge_fits_real_observed_targets_without_imputing_missing():
    torch.manual_seed(3)
    x = torch.randn(200, 70)
    truth = x @ torch.randn(70, 168) * 0.02
    mask = torch.ones_like(truth, dtype=torch.bool)
    mask[:, 2] = False
    truth[:, 2] = float("nan")
    w, info = fit_initial(x, truth, mask)
    assert w.shape == (168, 70) and not w[2].any()
    assert info["observed_outputs"] == 167
    assert info["fit_mse"] < info["hold_mse"]
    assert w.isfinite().all()


def test_ridge_rejects_nonfinite_observed_target():
    x, y = torch.zeros(2, 70), torch.zeros(2, 168)
    y[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        fit_initial(x, y, torch.ones_like(y, dtype=torch.bool))
