import numpy as np

import fit_output_trust_20261106 as fit


def test_joint_reward_cannot_profit_by_lowering_the_reached_anchor_more():
    n = np.zeros((2, 32), dtype=np.float32)
    n[0, 0] = fit.TAU + 0.01
    n[1, 1] = fit.TAU - 0.01
    z = n[:, None, None, None, :].copy()
    z[0, ..., 0] -= 0.2
    z[1, ..., 1] += 0.03
    pos, protect = np.zeros_like(n, dtype=bool), np.zeros_like(n, dtype=bool)
    pos[0, 0], protect[0, 0] = True, True
    inputs = dict(
        row_shot=np.array([100, 200]),
        positive=pos,
        protect=protect,
        present=np.ones_like(pos),
        frozen=n,
        future=z,
        anchor=np.array([-2.0, 0.0])[:, None, None, None],
    )
    records = [
        (dict(shot=100, disruptive=True), None, None),
        (dict(shot=200, disruptive=False), None, None),
    ]
    utility, guard, raw, joint, harm = fit.targets(inputs, records)
    np.testing.assert_array_equal(utility, [1, 0])
    np.testing.assert_array_equal(guard, [1, 0])
    np.testing.assert_allclose(raw[:, 0, 0, 0], [-0.2, -0.03], atol=2e-7)
    np.testing.assert_array_equal(joint, raw)
    assert harm[:, 0, 0, 0].tolist() == [1.0, 1.0]


def test_causal_stitch_writes_only_after_maturity_and_score_arm_is_distinct():
    n, s, k = 90, len(fit.SEEDS), len(fit.ARMS) - 1
    frozen = np.zeros(n, dtype=np.float32)
    z = np.ones((1, 2, s, 2, 32), dtype=np.float32)
    data = dict(
        maturity=np.array([63]),
        present=(np.arange(32) < 26)[None],
        available=np.ones(z.shape[:-1], dtype=bool),
        future=z,
        anchor=np.full(z.shape[:-1], 0.15, dtype=np.float32),
    )
    act = np.zeros((1, s, k), dtype=np.int64)
    replay = fit.trace_from_table(data, act, [0], frozen)
    assert not replay[:64].any()
    np.testing.assert_array_equal(replay[64:, :, 1], np.ones((26, s), dtype=np.float32))
    np.testing.assert_allclose(replay[64:, :, -1], 0.15)


def test_stale_control_requires_same_shot_consecutive_and_shared_diagnostics():
    x = np.ones((3, 3, 35))
    x[0, :, :22] = 0.4
    x[2, :, :22] = 0.8
    result, use = fit.previous_descriptor(
        x, np.array([1, 1, 2]), np.array([63, 95, 127])
    )
    assert not use[0].any() and use[1].all() and not use[2].any()
    np.testing.assert_array_equal(result[1, :, :22], x[0, :, :22])
    np.testing.assert_array_equal(result[2], x[2])
