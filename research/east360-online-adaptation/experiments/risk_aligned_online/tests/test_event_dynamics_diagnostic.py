import torch

from experiments.risk_aligned_online.event_dynamics_diagnostic import measure_step
from experiments.risk_aligned_online.tests.test_event_dynamics import make_model


def test_posthoc_measurement_replays_actual_step_without_mutating_model():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        m = make_model().requires_grad_(False)
        initial = {k: v.clone() for k, v in m.state_dict().items()}
        x, truth, qx = torch.randn(1, 128, 23), torch.randn(1, 25, 23), torch.randn(1, 128, 23)
        a = torch.ones_like(x, dtype=torch.bool)
        b = torch.ones_like(truth, dtype=torch.bool)
        row, candidate = measure_step(m, x, a, truth, b, qx, a, m.initial_fast(1), 1)
        assert row['offline_outcome_used']
        assert 0 < row['step_norm'] <= .050001
        assert row['gradient_norm'] > 0
        assert -1.00001 <= row['risk_descent_cosine'] <= 1.00001
        assert candidate.shape == (1, m.temporary_dim)
        for k, v in m.state_dict().items():
            torch.testing.assert_close(v, initial[k], rtol=0, atol=0)
        assert all(p.grad is None for p in m.parameters())
    finally:
        torch.set_num_threads(previous)
