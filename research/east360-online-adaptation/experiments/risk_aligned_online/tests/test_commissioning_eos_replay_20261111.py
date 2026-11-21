from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online import commissioning_eos_replay_20261111 as replay


def row(rank, positive, key):
    return dict(rank=rank, shot=10000 + rank, disruptive=positive,
                match=np.asarray(key, dtype=float))


def test_both_policies_use_same_class_budget_and_only_past():
    now = row(21, True, [0., 0., 0., 0.])
    prior = [row(16, True, [0., 0., 0., 0.]), row(17, False, [.1, 0., 0., 0.]),
             row(18, True, [1., 1., 1., 1.]), row(20, False, [2., 1., 1., 1.])]
    recent, matched = replay.select_history(prior, now)
    assert [r['rank'] for r in recent] == [18, 20]
    assert [r['rank'] for r in matched] == [16, 17]
    with pytest.raises(ValueError, match='previous'):
        replay.select_history([*prior, row(22, True, [0., 0., 0., 0.])], now)


def test_candidate_direction_receives_only_timely_positive_points_and_guard_peaks():
    # Positive: 125 ms and 31 ms are included; exactly 30 ms is excluded.
    t = np.array([.874, .875, .900, .969, .970, .971, 1.0])
    z = np.array([5., .1, .6, .2, 7., 8., 9.])
    p, anchor = replay.support(t, z, True, 1., .3)
    assert set(p) == {1, 2, 3}
    assert anchor == 2
    q, first = replay.support(t, z, False, None, .3)
    assert 6 in q and 0 in q and first == 0


def test_logit_loss_rewards_earlier_positive_and_protects_safe_peak():
    positive = torch.tensor([True, False, False])
    boundary = .1
    assert replay.alarm_loss(torch.tensor([-.3, -.4, -.1]), positive, boundary) > replay.alarm_loss(
        torch.tensor([.2, -.4, -.3]), positive, boundary)
    peak = torch.tensor([.2, -.4, .2], requires_grad=True)
    replay.alarm_loss(peak, positive, boundary).backward()
    assert peak.grad[0] < 0 and torch.all(peak.grad[1:] > 0)


def test_fast_release_never_touches_origin_maturity_or_previous_output():
    q = np.array([127, 159])
    np.testing.assert_array_equal(
        replay.release_slots(q, np.array([127, 151, 152, 153, 184, 185])),
        [0, 0, 0, 1, 1, 2],
    )


def test_same_persistent_input_dose_for_real_and_shifted_in_both_directions():
    start = torch.tensor([[1., 1.], [3., 3.], [2., 3.]])
    real = start + torch.tensor([[.5, 0.], [0., 1.], [1., 0.]])
    sham = start + torch.tensor([[0., 1.], [0., 0.], [0., .5]])
    a, b, rms = replay.equalized_inputs(start, real, sham)
    torch.testing.assert_close((a-start).norm(dim=1), (b-start).norm(dim=1))
    torch.testing.assert_close(rms, torch.tensor([.5, 0., .5]) / 2**.5)
    torch.testing.assert_close(a[1], start[1])
    torch.testing.assert_close(b[1], start[1])
    assert a[0, 0] > start[0, 0] and b[0, 1] > start[0, 1]


def test_seal_has_all_arms_before_label_is_opened_and_no_clobber(tmp_path):
    p = Path(tmp_path) / '0001_10001.npz'
    arrays = dict(times=np.arange(4), inter=np.arange(4, dtype=np.float32))
    seen = []

    def read(device, shot):
        assert device == 'EAST' and shot == 10001
        with np.load(p, allow_pickle=False) as file:
            assert set(file.files) == set(arrays)
            np.testing.assert_array_equal(file['inter'], arrays['inter'])
        seen.append(shot)
        return True, 2.

    digest, outcome = replay.seal_and_reveal(p, arrays, 10001, read)
    assert len(digest) == 64 and outcome == (True, 2.)
    with pytest.raises(FileExistsError):
        replay.seal_and_reveal(p, arrays, 10001, read)
    assert seen == [10001]
