import numpy as np
import pytest
import torch
from unittest.mock import patch

from experiments.risk_aligned_online.tests.test_contextual_feedback import setup
from experiments.risk_aligned_online.contextual_feedback_20261023.learning import step
from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
from experiments.risk_aligned_online.history_matched_rule_20261023.learning import replay, query_loss


def test_history_replay_matches_actual_forward_updates():
    m,p,s=setup();signal={k:s[k][:768] for k in ('values','available')}
    state=initial_encoder(m)
    for count in range(256,769,128):
        state,_=step(m,p,{k:v[:count] for k,v in signal.items()},count,state,device='cpu')
    result,receipt=replay(m,p,signal,768,device='cpu',tail_steps=2)
    assert receipt['all_counts']==[256,384,512,640,768]
    assert receipt['differentiated_counts']==[]
    for k in result:torch.testing.assert_close(result[k],state[k],rtol=0,atol=0)
    again,_=replay(m,p,signal,768,device='cpu',tail_steps=2)
    for k in result:torch.testing.assert_close(result[k],again[k],rtol=0,atol=0)


def test_two_step_meta_gradient_is_not_detached_and_matches_difference():
    m,p,s=setup();m=m.double();p=p.double()
    # Exactly two updates, with explicitly fixed context covariates: finite
    # differences verify the conditional two-step derivative. Deployment
    # context depends on state but intentionally has stopped derivatives.
    e=dict(count=384,end=400,label=0)
    with patch('experiments.risk_aligned_online.contextual_feedback_20261023.learning.make_context',return_value=np.zeros(185)):
        loss,_=query_loss(m,p,s,e,device='cpu',tail_steps=2)
    param=p.network[-1].bias;g=torch.autograd.grad(loss,param)[0]
    idx=int(g.abs().argmax());assert abs(float(g[idx]))>1e-10
    vals=[];eps=1e-4
    for sign in (1,-1):
        with torch.no_grad():param[idx]=sign*eps
        with patch('experiments.risk_aligned_online.contextual_feedback_20261023.learning.make_context',return_value=np.zeros(185)):
            vals.append(float(query_loss(m,p,s,e,device='cpu',tail_steps=2)[0].detach()))
    with torch.no_grad():param[idx]=0
    assert np.isclose(float(g[idx]),(vals[0]-vals[1])/(2*eps),rtol=.02,atol=1e-9)
    assert not any(q.requires_grad for q in m.parameters())


def test_replay_rejects_labels_future_suffix_and_off_clock_endpoints():
    m,p,s=setup();signal={k:s[k][:384] for k in ('values','available')}
    with pytest.raises(ValueError):replay(m,p,dict(signal,y=s['y'][:384]),384,device='cpu')
    with pytest.raises(ValueError):replay(m,p,{k:s[k][:400] for k in signal},384,device='cpu')
    with pytest.raises(ValueError):replay(m,p,{k:v[:383] for k,v in signal.items()},383,device='cpu')
    with pytest.raises(ValueError):query_loss(m,p,s,dict(count=384,end=385,label=1),device='cpu')


def test_first_step_matches_previous_objective():
    from experiments.risk_aligned_online.contextual_feedback_20261023.learning import policy_loss
    m,p,s=setup();e=dict(count=256,end=270,label=0)
    a,receipt=query_loss(m,p,s,e,device='cpu')
    b,_=policy_loss(m,p,[(s,e)],device='cpu')
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert receipt['differentiated_counts']==[256]
