import numpy as np
import torch

from joint_lora_gate_screen_20261104 import Gate, adam_delta, folds, group_weights


def test_gate_starts_at_uniform_and_stays_bounded():
    x = torch.randn(5, 192, dtype=torch.float64)
    for conditional in (False, True):
        gate = Gate(conditional)
        assert torch.equal(gate(x), torch.ones(5, 21, dtype=x.dtype))
        assert sum(p.numel() for p in gate.parameters()) == (447 if conditional else 21)
        with torch.no_grad():
            for parameter in gate.parameters():
                parameter.normal_(0, 20)
        w = gate(x)
        assert torch.all((w >= 0.5) & (w <= 1.5))
        torch.testing.assert_close(w.mean(-1), torch.ones(5, dtype=x.dtype))


def test_adam_teacher_displacement_matches_actual_first_step_even_when_clipped():
    grad = torch.tensor(
        [[3.0, -1e-8, 0.0, -0.03], [0.01, 0.0, 0.7, -1.2]], dtype=torch.float64
    )
    expected = adam_delta(grad)
    for vector, row in zip(grad, expected, strict=True):
        p = torch.nn.Parameter(torch.zeros_like(vector))
        p.grad = vector.clone()
        torch.nn.utils.clip_grad_norm_([p], 1.0)
        torch.optim.Adam([p], lr=0.0001).step()
        torch.testing.assert_close(p.detach(), row, rtol=1e-10, atol=1e-14)


def test_folds_keep_shots_whole_and_balance_twenty_shots():
    groups = folds(list(range(20)), {i: i < 8 for i in range(20)})
    assert sorted(i for fold in groups for i in fold) == list(range(20))
    assert all(len(fold) == 4 and 1 <= sum(i < 8 for i in fold) <= 2 for fold in groups)


def test_group_balance_counts_equal_shots_not_equal_windows():
    rows = [
        dict(shot=1, group="safe"),
        dict(shot=1, group="safe"),
        dict(shot=2, group="safe"),
        dict(shot=3, group="earlier"),
        dict(shot=3, group="positive"),
    ]
    np.testing.assert_allclose(
        group_weights(rows).numpy(), [1 / 12, 1 / 12, 1 / 6, 1 / 3, 1 / 3]
    )
