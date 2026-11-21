import pytest

from experiments.risk_aligned_online.east_context_operator_20261031.prepare import pair_schedule


def test_schedule_is_balanced_over_two_fixed_epochs_without_new_ids():
    examples = [dict(shot=i, flag=i not in (5, 12)) for i in range(20)]
    schedule = pair_schedule(examples)
    assert len(schedule) == 36
    for epoch in (1, 2):
        pairs = [p for p in schedule if p['epoch'] == epoch]
        assert [p['disruptive_index'] for p in pairs] == [i for i in range(20) if i not in (5, 12)]
        assert [p['safe_index'] for p in pairs] == [5, 12]*9
    assert {examples[p['safe_index']]['shot'] for p in schedule} == {5, 12}


@pytest.mark.parametrize('bad', [
    [], [dict(shot=i, flag=True) for i in range(20)],
    [dict(shot=i, flag=i < 17) for i in range(20)],
    [dict(shot=0, flag=i < 18) for i in range(20)],
])
def test_schedule_rejects_a_changed_cold_contract(bad):
    with pytest.raises(ValueError):
        pair_schedule(bad)


def test_real_fit_loop_uses_only_operator_and_preserves_off_predictions():
    import torch
    from experiments.risk_aligned_online.context_memory_20261028.model import ContextAdapter
    from experiments.risk_aligned_online.east_context_operator_20261031.prepare import (
        TRAINABLE, fit_operator,
    )
    from experiments.risk_aligned_online.simple23 import Config, Simple23
    torch.set_num_threads(1)
    torch.manual_seed(131)
    model = ContextAdapter(Simple23(Config(window=128, dim=64, fast_site='tokens',
                                          fast_rule='normalized'),
                                   torch.zeros(23), torch.ones(23)).eval()).eval()
    with torch.no_grad():
        model.memory.output.weight.normal_(0, .1)
        x = torch.randn(3, 128, 23)
        h, valid = model.tokens(x, torch.ones_like(x, dtype=torch.bool))
    examples = [dict(shot=i, flag=i < 18, h=h, valid=valid, means=torch.randn(4, 23),
                     complete=torch.ones(4, 23, dtype=torch.bool), ends=[16, 24, 31],
                     y=torch.full((3,), float(i < 18))) for i in range(20)]
    before = {n:v.clone() for n, v in model.state_dict().items()}
    progress = []
    result = fit_operator(model, examples, progress=progress.append)
    assert len(progress) == result['optimizer_steps'] == 36
    assert result['parameters'] == 7001 and result['zero_feedback_training_logits_bitwise']
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]) == (name not in TRAINABLE)
