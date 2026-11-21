import copy
import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.peak_balanced_feedback_20261023.learning import peak_manifest,balanced_head_gradient


def test_peak_sampling_covers_each_safe_shot_without_replacing_all_random_blocks():
    pools=[dict(shot=1,is_disrupt=False,blocks=[dict(count=256,ends=[256,257]),dict(count=384,ends=[384,385])]),
           dict(shot=2,is_disrupt=True,blocks=[dict(count=256,ends=[256])])]
    scores=np.zeros(400);scores[1]=.99;scores[385]=.8
    original=[dict(step=i,episodes=[dict(shot=1,is_disrupt=False,count=256,ends=[256,257]),
              dict(shot=2,is_disrupt=True,count=256,ends=[256])]) for i in range(4)]
    before=copy.deepcopy(original)
    revised,peaks=peak_manifest(original,pools,{1:scores})
    assert [x['episodes'][0]['count'] for x in revised]==[384,256,384,256]
    assert all(x['episodes'][1]==y['episodes'][1] for x,y in zip(original,revised))
    assert original==before and peaks[0]['count']==384
    with pytest.raises(ValueError):peak_manifest(original,pools,{1:scores,3:scores})
    bad=copy.deepcopy(original);bad[0]['episodes'][0]['shot']=3
    with pytest.raises(ValueError):peak_manifest(bad,pools,{1:scores})


def test_gradient_cap_preserves_signal_gradient_and_bounds_risk_component():
    p=torch.tensor([1.,2.],requires_grad=True)
    out=dict(forecast=p.sum(),risk_after=(p*torch.tensor([100.,0.])).sum(),differentiated_counts=[256,384])
    gradient,info=balanced_head_gradient(out,(p,))
    torch.testing.assert_close(gradient[0],torch.tensor([1.+2**.5,1.]))
    assert info['capped_risk_norm']<=info['forecast_norm']+1e-6
    assert p.grad is None and not gradient[0].requires_grad


def test_zero_risk_gradient_does_not_remove_forecast_learning():
    p=torch.tensor([1.],requires_grad=True)
    out=dict(forecast=p.sum(),risk_after=(p*0).sum(),differentiated_counts=[256])
    gradient,info=balanced_head_gradient(out,(p,))
    assert gradient[0].item()==1 and info['risk_scale']==1
    q=torch.tensor([1.],requires_grad=True)
    gradient,info=balanced_head_gradient(dict(forecast=(q*0).sum(),risk_after=q.sum(),differentiated_counts=[256]),(q,))
    assert gradient[0].item()==0 and info['risk_scale']==0


def test_capped_outer_gradient_changes_only_auxiliary_head_on_real_operator():
    from experiments.risk_aligned_online.tests.test_matched_forecast_meta import fixture
    from experiments.risk_aligned_online.target_feedback_meta_20261023.learning import block_episode
    model,signal=fixture();model.base.forecast_head.requires_grad_(True)
    before={k:v.clone() for k,v in model.state_dict().items()}
    out=block_episode(model,signal,256,list(range(256,280)),True,boundary=0.,
                      objective='maximum',device='cpu',create_graph=True)
    params=tuple(model.base.forecast_head.parameters());grads,info=balanced_head_gradient(out,params)
    opt=torch.optim.Adam(params,lr=1e-4)
    for p,g in zip(params,grads):p.grad=g
    opt.step()
    changed={k for k,v in model.state_dict().items() if not torch.equal(v,before[k])}
    assert changed=={'base.forecast_head.weight','base.forecast_head.bias'}
    assert info['capped_risk_norm']<=info['forecast_norm']+1e-6
