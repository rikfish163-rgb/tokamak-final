import numpy as np
import pytest
import torch

import fit_layer0_ordered_motion_20261106 as fit
from layer0_ordered_motion_20261106 import OrderedHead


def test_exact_origin_sampling_keeps_historical_and_maturity_boundaries():
    assert not len(fit.origins_for(202, "train"))
    np.testing.assert_array_equal(fit.origins_for(203, "val"), [177])
    np.testing.assert_array_equal(fit.origins_for(300, "cold"), [206, 238, 270])
    with pytest.raises(ValueError, match="role"):
        fit.origins_for(300, "test")


def test_only_train_rows_carry_future_even_when_raw_has_full_record():
    v = np.broadcast_to(np.arange(300, dtype=np.float32)[:, None], (300, 23)).copy()
    a = np.ones_like(v, dtype=bool)
    t = np.array([177, 202])
    for role in ("val", "cold"):
        q = fit.make_rows(v, a, 9, t, role)
        assert set(q) == {"x", "a", "shot", "origin"}
        np.testing.assert_array_equal(q["x"][:, 0, -1, 0], t)
    train = fit.make_rows(v, a, 9, t, "train")
    np.testing.assert_array_equal(train["y"][:, 0, 0], t + 1)
    np.testing.assert_array_equal(train["y"][:, -1, 0], t + 25)


def test_forecast_cannot_see_unarrived_samples(monkeypatch):
    class Base:
        center = torch.zeros(23)
        scale = torch.ones(23)

    def features(base, x, a, action=None):
        v = x[:, -1, 0] / 100
        return v[:, None].expand(-1, 128), None

    monkeypatch.setattr(fit.ordered.motion, "features", features)
    head = OrderedHead(torch.ones(128)).requires_grad_(False)
    with torch.no_grad():
        head.context.weight.fill_(0.001)
    v = np.broadcast_to(np.arange(300, dtype=np.float32)[:, None], (300, 23)).copy()
    a = np.ones_like(v, dtype=bool)
    t = np.array([177, 202])
    all_ = fit.infer_rows(Base(), head, fit.make_rows(v, a, 9, t, "cold"))
    past = fit.infer_rows(
        Base(), head, fit.make_rows(v[:178], a[:178], 9, t[:1], "cold")
    )
    for key in all_:
        np.testing.assert_allclose(all_[key][:1], past[key], atol=3e-6)
    assert all_["predictions"].shape == (2, 6, 10, 23)
    y = all_["predictions"]
    assert not y[:, 0].any() and not np.array_equal(y[:, 3], y[:, 4])
    np.testing.assert_allclose((y[:, 3] + y[:, 4]) / 2, y[:, 2], atol=3e-6)
