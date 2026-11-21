import numpy as np
import torch

import run_adjacent_patch_intra_20261106 as intra
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_internal_policy_respects_release_and_true_score_control():
    with torch.random.fork_rng():
        torch.manual_seed(9123)
        base = (
            Simple23(
                Config(fast_site="tokens", fast_rule="normalized"),
                torch.zeros(23),
                torch.ones(23),
            )
            .eval()
            .requires_grad_(False)
        )
    n = 141
    rng = np.random.default_rng(33)
    raw = dict(
        values=rng.normal(size=(n, 23)).astype(np.float32),
        available=np.ones((n, 23), dtype=bool),
        times=np.arange(n, dtype=np.float64) / 1000,
    )
    projection = intra.op.basis(base)
    pred = intra.predict(base, projection, raw)
    np.testing.assert_array_equal(pred["clock"], [63, 95, 127])
    np.testing.assert_array_equal(
        pred["curve"][:64],
        np.repeat(pred["curve"][:64, :1], len(intra.ARMS), axis=1),
    )
    assert pred["active"].any()
    assert pred["guard"].max() <= intra.pair.LIMIT
    assert np.ptp(pred["edit_rms"], axis=1).max() < 1e-6
    for k, t in enumerate(pred["clock"]):
        q = slice(t + 1, min(t + 33, n))
        np.testing.assert_allclose(
            pred["curve"][q, 5],
            pred["curve"][q, 0] + pred["offset"][k, 0],
            atol=1e-7,
        )
    assert intra.check_prefix(base, projection, raw, pred)["curve_error"] < 8e-5
    assert intra.check_row_independence(base, projection, raw) < 2e-6
