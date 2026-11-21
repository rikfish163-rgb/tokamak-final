import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.tests.test_contextual_feedback import setup
from experiments.risk_aligned_online.contextual_feedback_20261023.learning import step,predict
from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder


def test_context_dependent_weights_and_mean_ablation_are_distinct():
    _,p,_=setup()
    with torch.no_grad():p.network[-1].weight.normal_(0,.1)
    a,b=torch.zeros(185),torch.ones(185)
    assert not torch.equal(p(a),p(b))
    torch.testing.assert_close(p(a,mean_context=True),p(b,mean_context=True),rtol=0,atol=0)


def test_missing_values_cannot_enter_context_or_weighted_gradient():
    m,p,s=setup()
    with torch.no_grad():p.network[-1].weight.normal_(0,.1)
    signal={k:s[k][:256].copy() for k in ('values','available')}
    signal['available'][150:,1]=False
    changed={k:v.copy() for k,v in signal.items()};changed['values'][150:,1]=np.nan
    fast=initial_encoder(m)
    a,ar=step(m,p,signal,256,fast,device='cpu')
    b,br=step(m,p,changed,256,fast,device='cpu')
    assert ar==br
    for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    with pytest.raises(ValueError):predict(m,p,p,dict(signal,y=np.ones(256)),device='cpu')
