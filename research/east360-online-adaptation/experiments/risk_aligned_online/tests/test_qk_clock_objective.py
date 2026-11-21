import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.tests.test_qk_matched_head import fixture
from experiments.risk_aligned_online.qk_clock_objective_20261029.learning import (
    assemble,
    aggregate_objective,
    train_step,
)
from experiments.risk_aligned_online.qk_matched_head_20261029.learning import adapt


def test_fresh_feedback_excludes_query_and_future_and_differs_only_in_feedback():
    net, s = fixture()
    choices = [(0, 300)]
    fresh = assemble([s], choices, "cpu", "fresh")
    blocked = assemble([s], choices, "cpu", "blocked")
    assert (
        fresh["receipts"][0]["count"] == 300 and blocked["receipts"][0]["count"] == 256
    )
    assert fresh["receipts"][0]["last_target"] == 299
    torch.testing.assert_close(fresh["qx"], blocked["qx"], atol=0, rtol=0)
    changed = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in s.items()}
    changed["values"][300:] += 99
    changed["y"][:] = 1
    other = assemble([changed], choices, "cpu", "fresh")
    for k in fresh["feedback"]:
        torch.testing.assert_close(
            fresh["feedback"][k], other["feedback"][k], atol=0, rtol=0
        )
    torch.testing.assert_close(
        adapt(net, fresh["feedback"])[0],
        adapt(net, other["feedback"])[0],
        atol=0,
        rtol=0,
    )
    with pytest.raises(ValueError):
        assemble([s], choices, "cpu", "unknown")


def test_clocks_agree_at_block_boundary():
    _, s = fixture()
    a = assemble([s], [(0, 256)], "cpu", "fresh")
    b = assemble([s], [(0, 256)], "cpu", "blocked")
    assert a["receipts"] == b["receipts"]
    for k in a["feedback"]:
        torch.testing.assert_close(a["feedback"][k], b["feedback"][k], atol=0, rtol=0)


def test_worst_objective_does_not_average_away_group_harm():
    delta = torch.tensor([-1.0, -1.0, 0.1], requires_grad=True)
    signal = torch.zeros(3)
    alignment = torch.zeros(3)
    mean = aggregate_objective(signal, delta, alignment, "mean")
    worst = aggregate_objective(signal, delta, alignment, "worst")
    assert float(mean.detach()) < 0 and float(worst.detach()) == pytest.approx(0.05)
    (g,) = torch.autograd.grad(worst, delta)
    torch.testing.assert_close(g, torch.tensor([0.0, 0.0, 0.5]))
    with pytest.raises(ValueError):
        aggregate_objective(signal, delta, alignment, "arbitrary")


def test_worst_update_trains_only_auxiliary_head():
    net, s = fixture()
    e = assemble([s], [(0, 300)] * 4 + [(0, 350)] * 4 + [(0, 480)] * 4, "cpu", "fresh")
    before = {k: v.clone() for k, v in net.state_dict().items()}
    net.base.forecast_head.requires_grad_(True)
    opt = torch.optim.AdamW(
        net.base.forecast_head.parameters(), lr=1e-4, weight_decay=0
    )
    r = train_step(net, opt, e, "worst")
    assert np.isfinite(r["objective"]) and r["head_gradient_norm"] > 0
    state = net.state_dict()
    assert all(
        torch.equal(state[k], v) for k, v in before.items() if "forecast_head." not in k
    )
    assert any(
        not torch.equal(state[k], v) for k, v in before.items() if "forecast_head." in k
    )


def test_code_snapshot_includes_cli_entry_even_when_executed_as_main():
    from pathlib import Path
    from experiments.risk_aligned_online.qk_clock_objective_20261029.run import (
        source_files,
    )

    entry = Path("/tmp/example_research_entry.py")
    assert entry in source_files(entry)


def test_independent_recount_weights_shots_not_number_of_endpoints():
    from experiments.risk_aligned_online.qk_clock_objective_20261029.audit import (
        describe,
    )

    rows = []
    for group in ("safe", "early", "danger"):
        for sid, delta, copies in ((1, -1.0, 3), (2, 1.0, 1)):
            rows.extend(
                dict(
                    shot=sid,
                    group=group,
                    before=2.0,
                    after=2.0 + delta,
                    probability_change=0.1 * delta,
                )
                for _ in range(copies)
            )
    r = describe(rows)
    for group in ("safe", "early", "danger"):
        assert r[group]["mean"] == 0 and r[group]["shots"] == 2
        assert r[group]["improved"] == 1 and r[group]["worsened"] == 1
        assert r[group]["sd_between_shots"] == pytest.approx(2**0.5)
