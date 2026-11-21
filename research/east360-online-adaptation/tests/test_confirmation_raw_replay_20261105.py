import numpy as np
import pytest
import torch

from confirmation_raw_replay_20261105 import ARMS, CausalConfirm, infer_raw
from experiments.risk_aligned_online.simple23 import Config, Simple23
from point_confirmation_20261105 import gated


def test_confirmation_memory_respects_inclusive_boundary_and_never_backdates():
    base = np.full(70, 0.1, dtype=np.float32)
    active = np.full_like(base, 0.2)
    shadow = np.full_like(base, 0.3)
    active[[3, 35, 36, 64]] = 0.8
    shadow[[4, 65]] = 0.9
    base[8], shadow[8] = 0.95, 0.95  # a native alarm cannot support a new one
    g = CausalConfirm(0.7, 32)
    cuts = (0, 4, 10, 35, 36, 64, 70)
    actual = np.concatenate(
        [
            g.push(base[a:b], active[a:b], shadow[a:b])
            for a, b in zip(cuts[:-1], cuts[1:], strict=True)
        ]
    )
    np.testing.assert_array_equal(actual, gated(base, active, shadow, 0.7, 32))
    assert np.flatnonzero(actual >= 0.7).tolist() == [8, 35, 36]


def test_confirmation_rejects_unaligned_or_nonprobability_input():
    gate = CausalConfirm(0.7, 32)
    with pytest.raises(ValueError):
        gate.push([0.2], [0.2, 0.3], [0.2])
    with pytest.raises(ValueError):
        gate.push([0.2], [1.1], [0.2])


@pytest.fixture
def toy():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        model = (
            Simple23(
                Config(fast_site="tokens", fast_rule="normalized"),
                torch.zeros(23),
                torch.ones(23),
            )
            .eval()
            .requires_grad_(False)
        )
        point = (torch.ones(84), 0.05 * torch.randn(84, 256))
        values = torch.randn(171, 23).numpy()
        mask = np.ones_like(values, dtype=bool)
        try:
            yield model, point, {"values": values, "available": mask}
        finally:
            torch.backends.mha.set_fastpath_enabled(previous)


def test_all_forecasts_through_anchor_invariant_to_later_signal_and_missingness(toy):
    model, point, shot = toy
    p, _, _ = infer_raw(model, shot, point, 0.7, batch_size=47)
    changed = {k: v.copy() for k, v in shot.items()}
    changed["values"][167:] += 20
    changed["available"][167:] = False
    q, _, _ = infer_raw(model, changed, point, 0.7, batch_size=47)
    r, _, _ = infer_raw(
        model, {k: v[:167] for k, v in shot.items()}, point, 0.7, batch_size=1
    )
    for arm in ARMS:
        np.testing.assert_array_equal(p[arm][:167], q[arm][:167])
        np.testing.assert_allclose(p[arm][:167], r[arm], rtol=0, atol=4e-6)


def test_empty_history_and_short_shot_have_no_donor_effect(toy):
    model, point, shot = toy
    q, _, info = infer_raw(model, shot, point, 0.7, batch_size=47)
    assert info["n_arrivals"] == 2
    for arm in ("previous_clock", "donor_confirm_past32"):
        np.testing.assert_array_equal(q[arm], q["static"])
    r, direction, stats = infer_raw(
        model, {k: v[:135] for k, v in shot.items()}, point, 0.7
    )
    assert stats["n_arrivals"] == 0 and direction.shape == (0, 256)
    assert all(np.array_equal(v, r["static"]) for v in r.values())


def test_rank100_can_supply_a_donor_without_applying_current_updates(toy):
    model, point, shot = toy
    q, direction, info = infer_raw(
        model, shot, point, 0.7, enabled=False, collect_donor=True
    )
    assert direction.shape == (2, 256) and info["n_arrivals"] == 2
    assert sum(info["optimizer_writes"].values()) == 0
    assert all(np.array_equal(v, q["static"]) for v in q.values())
    _, _, used = infer_raw(model, shot, point, 0.7, previous=direction[:1])
    assert used["donor_held_arrivals"] == 1
    assert used["optimizer_writes"]["previous_clock"] == 2
    assert used["norm_mismatches"]["previous_clock"] == 0
