import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from run_alarm_order_attribution_20261106 import anchors_for, joint_relabel


def test_anchor_schedule_no_label_arguments_and_never_uses_post_alarm_windows():
    p = np.full(400, 0.3, dtype=np.float32)
    p[[159, 250, 333]] = (0.69, 0.8, 0.9)
    a, f, near = anchors_for(p, 0.725)
    np.testing.assert_array_equal(a, [127, 159, 191, 223, 250])
    np.testing.assert_array_equal(f, [False, False, False, False, True])
    np.testing.assert_array_equal(near, [False, True, False, False, False])
    p[252:] = 0.0
    assert all(
        np.array_equal(x, y) for x, y in zip((a, f, near), anchors_for(p, 0.725))
    )
    p[2] = 0.8
    assert anchors_for(p, 0.725)[0].tolist() == [2]
    with pytest.raises(ValueError, match="probability"):
        anchors_for(np.array([float("nan")]), 0.725)


def test_joint_token_and_position_relabel_leaves_frozen_cls_risk_invariant():
    torch.manual_seed(131)
    base = Simple23(Config(fast_site="tokens"), torch.zeros(23), torch.ones(23))
    base.eval().requires_grad_(False)
    x = torch.randn(3, 128, 23)
    a = torch.ones_like(x, dtype=torch.bool)
    a[:, :, 2] = False
    q = torch.arange(16).expand(3, 16).clone()
    q[:, 11:15] = q[:, 11:15].flip(1)
    with torch.no_grad():
        original = base(x, a)["logit"]
    got = joint_relabel(base, x, a, q)
    torch.testing.assert_close(got, original, rtol=2e-5, atol=5e-6)
    assert not base.encoder._forward_pre_hooks


def test_relabel_rejects_support_changes_and_never_moves_the_terminal_patch():
    base = Simple23(Config(fast_site="tokens"), torch.zeros(23), torch.ones(23))
    base.eval().requires_grad_(False)
    x = torch.zeros((1, 128, 23))
    a = torch.ones_like(x, dtype=torch.bool)
    q = torch.arange(16)[None, :]
    bad = q.clone()
    bad[:, -2:] = bad[:, -2:].flip(1)
    with pytest.raises(ValueError, match="terminal"):
        joint_relabel(base, x, a, bad)
    a[0, 11 * 8, 4] = False
    bad = q.clone()
    bad[:, 11:15] = bad[:, 11:15].flip(1)
    with pytest.raises(ValueError, match="mask"):
        joint_relabel(base, x, a, bad)
