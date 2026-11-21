import torch
from experiments.risk_aligned_online.channel_oracle import measure
from experiments.risk_aligned_online.fast_direction import episode
from experiments.risk_aligned_online.tests.test_sequence_fast import fixture


def test_offline_selector_includes_noop_and_uniform_and_keeps_model():
    model, shot = fixture()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    row = measure(model, episode(shot, 0, model.cfg, 'cpu'))
    assert row['selected_loss'] <= row['before_loss'] + 1e-7
    assert row['selected_loss'] <= row['uniform_loss'] + 1e-7
    assert len(row['channel_losses']) == 23
    assert row['selected_index'] in range(25)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def test_no_observed_channel_cannot_provide_a_direction():
    model, shot = fixture()
    e = episode(shot, 0, model.cfg, 'cpu')
    e['mask'][:] = False
    row = measure(model, e)
    assert row['available_channels'] == 0
    assert row['selected_index'] == 0
    assert row['channel_cosines'] == [0.] * 23
