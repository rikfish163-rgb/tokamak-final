import numpy as np
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from timed_token_norm_20261105 import support_weights, timed_logit, set_direction


def test_recent_and_older_masks_use_absolute_patch_intervals_and_match_energy():
    valid = torch.ones(1, 128, 23, dtype=torch.bool)
    results = {
        a: support_weights(valid, [159], [135], a)
        for a in ("recent", "older", "global", "cls")
    }
    assert results["recent"].nonzero()[:, 1].tolist() == [9, 10, 11, 12]
    assert results["older"].nonzero()[:, 1].tolist() == [1, 2, 3, 4]
    assert all(abs(float(v.square().sum()) - 4) < 1e-6 for v in results.values())
    assert results["global"][0, -1] == 0
    valid[:, 8:16] = False
    assert not support_weights(valid, [159], [135], "recent").any()


def test_zero_state_is_exact_parent_and_nonzero_step_only_changes_enabled_tokens():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(8)
    base = (
        Simple23(
            Config(fast_site="tokens", fast_rule="normalized"),
            torch.zeros(23),
            torch.ones(23),
        )
        .eval()
        .requires_grad_(False)
    )
    x = torch.randn(1, 128, 23)
    a = torch.ones_like(x, dtype=torch.bool)
    weights = support_weights(a, [159], [135], "recent")
    delta = torch.nn.Parameter(torch.zeros(128))
    with torch.no_grad():
        zero, energy = timed_logit(base, x, a, weights, delta)
        assert torch.equal(zero, base(x, a)["logit"])
        assert float(energy) == 0
    set_direction(delta, np.arange(1, 129))
    change, energy = timed_logit(base, x, a, weights, delta)
    assert abs(float(delta.detach().norm()) - 0.05) < 1e-7
    assert float(energy.detach()) > 0
    assert not torch.equal(change, zero)
    assert not base.encoder.layers[-1].norm1._forward_hooks
