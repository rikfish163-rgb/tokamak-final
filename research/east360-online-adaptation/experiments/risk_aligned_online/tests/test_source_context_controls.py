import torch

from experiments.risk_aligned_online.context_memory_20261028.model import ContextAdapter
from experiments.risk_aligned_online.context_memory_20261028.source_controls import evaluate_modes
from experiments.risk_aligned_online.tests.test_source_task_memory import setup


def test_untrained_zero_readout_has_identical_risk_across_feedback_controls():
    old,e=setup()
    model=ContextAdapter(old.base,forecast_weights=torch.ones(4,23)).eval()
    e.update(labels=torch.ones(4),local_mask=torch.ones(4,dtype=torch.bool),direction_mask=torch.ones(4,dtype=torch.bool))
    result=evaluate_modes(model,e,.7)
    assert set(result)=={'memory','off','reverse','hold'}
    for mode in result:
        assert result[mode]['bag']==result['off']['bag']
        assert result[mode]['local']==result['off']['local']
        assert result[mode]['mean_logit']==result['off']['mean_logit']
    assert all(p.grad is None for p in model.parameters())
