import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.normal_reference_alignment_20261031.core import (
    fit_coordinates,
    make_transport,
    transport,
)
from experiments.risk_aligned_online.normal_reference_alignment_20261031.normalized_forward import (
    coordinate_signal,
    forward_coordinates,
)


def fixture():
    torch.manual_seed(17)
    base = Simple23(Config(fast_site="tokens"), torch.zeros(23), torch.ones(23)).eval()
    x = torch.randn(2, 128, 23)
    m = torch.rand_like(x) > 0.1
    return base, x, m


def test_identical_source_coordinates_exact_forward_and_cleanup():
    base, x, m = fixture()
    original = base(x, m)
    level = torch.asinh(torch.where(m, x, base.center) - base.center)
    state = {k: v.clone() for k, v in base.state_dict().items()}
    result = forward_coordinates(base, x, m, level, torch.ones(23, dtype=torch.bool))
    assert torch.equal(result["logit"], original["logit"])
    assert torch.equal(result["forecast"], original["forecast"])
    assert not base.embedding._forward_pre_hooks
    assert all(torch.equal(v, base.state_dict()[k]) for k, v in state.items())


def test_large_coordinates_do_not_need_unrepresentable_raw_values():
    base, x, m = fixture()
    level = torch.full_like(x, 103.0)
    result = forward_coordinates(base, x, m, level, torch.ones(23, dtype=torch.bool))
    assert torch.isfinite(result["logit"]).all()
    assert not base.embedding._forward_pre_hooks


def test_masked_coordinates_and_inactive_channels_ignored():
    base, x, m = fixture()
    active = torch.zeros(23, dtype=torch.bool)
    active[1] = True
    level = torch.full_like(x, 7.0)
    out = forward_coordinates(base, x, m, level, active)
    level[:, :, 3] = 700.0
    level[~m] = 900.0
    changed = forward_coordinates(base, x, m, level, active)
    assert torch.equal(out["logit"], changed["logit"])


def test_hook_removed_on_model_exception():
    base, x, m = fixture()

    def fail(*args):
        raise RuntimeError("injected encoder failure")

    handle = base.encoder.register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match="injected"):
        forward_coordinates(
            base, x, m, torch.zeros_like(x), torch.ones(23, dtype=torch.bool)
        )
    assert not base.embedding._forward_pre_hooks
    handle.remove()


@pytest.mark.parametrize("mode", ["diagonal", "correlation"])
def test_coordinate_map_matches_v1_where_raw_representable(mode):
    rng = np.random.default_rng(18)
    rows = [
        dict(
            shot=i,
            values=rng.normal(size=(128, 23)).astype("float32"),
            available=np.ones((128, 23), bool),
        )
        for i in range(3)
    ]
    target = [
        dict(
            shot=r["shot"],
            values=(r["values"] * 2 + 0.3).astype("float32"),
            available=r["available"],
        )
        for r in rows
    ]
    fit = make_transport(
        fit_coordinates(rows, np.zeros(23), np.ones(23)),
        fit_coordinates(target, np.zeros(23), np.ones(23)),
        mode,
    )
    signal = {k: target[0][k] for k in ("values", "available")}
    z = coordinate_signal(signal, fit, np.zeros(23), np.ones(23))
    raw = transport(signal, fit, np.zeros(23), np.ones(23))
    assert np.allclose(
        z["values"][:, fit["active"]],
        np.arcsinh(raw["values"])[:, fit["active"]],
        atol=3e-7,
        rtol=1e-6,
    )
    prefix = {k: v[:19] for k, v in signal.items()}
    assert np.array_equal(
        coordinate_signal(prefix, fit, np.zeros(23), np.ones(23))["values"],
        z["values"][:19],
    )


def test_reject_nonfinite_mapped_coordinate():
    base, x, m = fixture()
    level = torch.full_like(x, float("inf"))
    with pytest.raises(ValueError):
        forward_coordinates(base, x, m, level, torch.ones(23, dtype=torch.bool))


def test_no_observed_active_coordinate_has_nullable_diagnostic():
    from experiments.risk_aligned_online.normal_reference_alignment_20261031.run_coordinates import (
        predict,
    )

    base, _, _ = fixture()
    rng = np.random.default_rng(21)
    records = [
        dict(
            shot=i,
            values=rng.normal(size=(24, 23)).astype("float32"),
            available=np.ones((24, 23), bool),
        )
        for i in range(3)
    ]
    stats = fit_coordinates(records, np.zeros(23), np.ones(23))
    fit = make_transport(stats, stats, "diagonal")
    signal = dict(
        values=np.zeros((2, 23), "float32"), available=np.zeros((2, 23), bool)
    )
    scores, info = predict(base, signal, fit)
    assert np.isfinite(scores).all()
    assert info["observed_active_points"] == 0
    assert info["max_abs_mapped_coordinate"] is None
