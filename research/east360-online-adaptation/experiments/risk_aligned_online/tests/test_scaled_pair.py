import pytest
import torch

from experiments.risk_aligned_online.context_memory_20261028.scaled_pair import errors


def test_pair_audit_uses_identical_metric_for_both_forecasts():
    x=torch.arange(7.)[:,None].repeat(1,23)
    a=torch.ones_like(x,dtype=torch.bool)
    w=torch.ones(4,23)
    pred=torch.zeros(7,4,23)
    row=errors(x,a,pred,w)
    assert row['raw']==row['weighted']==row['persistence']==7.5
    pred[:]=torch.arange(1.,5.).view(1,4,1)
    perfect=errors(x,a,pred,w)
    assert perfect['weighted']==0 and perfect['persistence']==7.5
    a[:,3]=False; x[:,3]=float('nan'); pred[:,:,3]=float('nan')
    assert errors(x,a,pred,w)['weighted']==0


def test_empty_pair_metric_is_not_reported_as_zero_error():
    with pytest.raises(ValueError):
        errors(torch.ones(7,23),torch.zeros(7,23,dtype=torch.bool),torch.zeros(7,4,23),torch.ones(4,23))
