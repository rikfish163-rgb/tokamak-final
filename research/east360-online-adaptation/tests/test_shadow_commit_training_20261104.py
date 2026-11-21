import numpy as np
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.shadow_commit_20261104.training import (
    records_and_pools,
    fit_connection,
)
from experiments.risk_aligned_online.cold20_compare_20261104.data import make_schedule


def test_both_connections_share_exact_schedule_and_source_stays_frozen():
    torch.manual_seed(3)
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    base = (
        Simple23(Config(fast_site="tokens"), torch.zeros(23), torch.ones(23))
        .eval()
        .requires_grad_(False)
    )
    values = np.random.default_rng(3).normal(size=(300, 23)).astype("float32")
    shots = {
        s: dict(
            values=values,
            available=np.ones_like(values, bool),
            times=np.arange(300) / 1000,
        )
        for s in range(20)
    }
    truth = {s: (s % 2 == 0, 0.3 if s % 2 == 0 else None) for s in shots}
    records, pools = records_and_pools(shots, truth, list(shots))
    draws = make_schedule(pools, steps=2, batch_size=4)
    zero = {s: torch.zeros(300, 168) for s in shots}
    learned = {s: torch.randn(300, 168) for s in shots}
    before = {k: v.clone() for k, v in base.state_dict().items()}
    a, h = fit_connection(
        base, shots, records, draws, zero, threshold=0.5, device="cpu", name="zero"
    )
    b, hh = fit_connection(
        base,
        shots,
        records,
        draws,
        learned,
        threshold=0.5,
        device="cpu",
        name="learned",
    )
    assert len(h) == len(hh) == 2 and not a.weight.any() and b.weight.abs().sum() > 0
    assert all(torch.equal(v, before[k]) for k, v in base.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in base.parameters())
    assert not b.weight.requires_grad and b.weight.grad is None


def test_records_keep_hdf125_and_exclude_only_late_positive_sampling():
    values = np.zeros((300, 23), np.float32)
    shots = {
        s: dict(
            values=values,
            available=np.ones_like(values, bool),
            times=np.arange(300) / 1000,
        )
        for s in range(20)
    }
    truth = {s: (s % 2 == 0, 0.3 if s % 2 == 0 else None) for s in shots}
    records, _ = records_and_pools(shots, truth, list(shots))
    for row in records:
        lead = (0.3 - shots[row["shot"]]["times"][row["query"]]) * 1000
        if row["group"] == "positive":
            assert 30 < lead <= 125 + 1e-6
        if row["group"] == "earlier":
            assert lead > 125
