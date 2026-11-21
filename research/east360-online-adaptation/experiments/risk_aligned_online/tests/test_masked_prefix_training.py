import pytest
import torch

from experiments.risk_aligned_online.masked_prefix_train import train_step, current_window
from experiments.risk_aligned_online.tests.test_masked_prefix_feedback import make_model


@pytest.mark.parametrize('mode', ['signal', 'risk'])
def test_only_auxiliary_decoder_changes_in_source_training(mode):
    torch.set_num_threads(2)
    m = make_model()
    m.decoder.requires_grad_(True)
    old = {k: v.clone() for k, v in m.base.state_dict().items()}
    weight = m.decoder.weight.clone()
    e = dict(x=torch.randn(2, 128, 23), a=torch.ones(2, 128, 23, dtype=torch.bool),
             truth=torch.randn(2, 25, 23), mask=torch.ones(2, 25, 23, dtype=torch.bool),
             qx=torch.randn(2, 128, 23), qa=torch.ones(2, 128, 23, dtype=torch.bool),
             y=torch.tensor([0., 1.]))
    optimizer = torch.optim.AdamW(m.decoder.parameters(), lr=1e-4, weight_decay=0.)
    report = train_step(m, optimizer, e, phase=1, mode=mode)
    assert report['head_grad_norm'] > 0 and report['risk_after'] >= 0
    assert not torch.equal(m.decoder.weight, weight)
    for k, v in m.base.state_dict().items(): torch.testing.assert_close(v, old[k], atol=0, rtol=0)
    assert all(p.grad is None for p in m.base.parameters())


def test_source_episode_window_stops_before_query():
    e = dict(x=torch.zeros(1, 128, 23), a=torch.ones(1, 128, 23, dtype=torch.bool),
             truth=torch.ones(1, 25, 23), mask=torch.ones(1, 25, 23, dtype=torch.bool),
             qx=torch.full((1, 128, 23), 100.))
    x, a = current_window(e)
    assert torch.equal(x[:, :-25], e['x'][:, 25:])
    assert torch.equal(x[:, -25:], e['truth']) and a.all()
