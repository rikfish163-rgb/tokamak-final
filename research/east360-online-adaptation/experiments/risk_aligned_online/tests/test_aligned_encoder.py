from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.aligned_encoder import (
    encoder_parameters,source_step,predict_shot,
)
from experiments.risk_aligned_online.tests.test_cross_channel_stable import model
from experiments.risk_aligned_online.tests.test_cross_channel_prefix import signals


@pytest.fixture(autouse=True)
def thread_limit():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_source_preparation_changes_copy_not_teacher_head_or_coordinates():
    teacher=model();student=deepcopy(teacher)
    original={k:v.clone() for k,v in teacher.state_dict().items()}
    params=encoder_parameters(student)
    for p in params.values():p.requires_grad_(True)
    student.decoder.requires_grad_(True)
    opt=torch.optim.Adam([*params.values(),*student.decoder.parameters()],lr=1e-4)
    x=torch.randn(3,128,23);a=torch.ones_like(x,dtype=torch.bool);y=torch.tensor([0.,1.,0.])
    row=source_step(student,teacher,opt,x,a,y,phase=0)
    assert np.isfinite(row['total']) and row['alignment_width']==64
    changed=[k for k,v in student.state_dict().items() if not torch.equal(v,original[k])]
    assert any(k.startswith('base.encoder.') for k in changed)
    assert any(k.startswith('decoder.') for k in changed)
    assert all(k.startswith(('base.embedding.','base.encoder.','base.norm.','decoder.')) for k in changed)
    for k,v in teacher.state_dict().items():torch.testing.assert_close(v,original[k],atol=0,rtol=0)


def test_source_optimizer_may_not_modify_risk_head():
    teacher=model();student=deepcopy(teacher)
    params=list(encoder_parameters(student).values())+list(student.decoder.parameters())
    opt=torch.optim.Adam([*params,student.base.risk_head.weight],lr=1e-4)
    x=torch.randn(2,128,23);a=torch.ones_like(x,dtype=torch.bool)
    with pytest.raises(ValueError,match='optimizer'):
        source_step(student,teacher,opt,x,a,torch.tensor([0.,1.]),phase=0)


def test_whole_encoder_intra_is_causal_and_ephemeral():
    m=model();s=signals(105);original={k:v.clone() for k,v in m.state_dict().items()}
    p,audit=predict_shot(m,s,intra=True)
    changed=dict(s,values=s['values'].copy());changed['values'][60:]+=12.
    q,_=predict_shot(m,changed,intra=True)
    r,_=predict_shot(m,s,intra=True)
    np.testing.assert_array_equal(p[:60],q[:60]);np.testing.assert_array_equal(p,r)
    assert audit['accepted_updates']>0 and audit['fast_discarded']
    assert all(r['effective_index']==r['update_index']+1 for r in audit['attempts'])
    assert audit['parameter_delta_norm']<=.500001
    for k,v in m.state_dict().items():torch.testing.assert_close(v,original[k],atol=0,rtol=0)
    with pytest.raises(ValueError,match='outcome'):
        predict_shot(m,dict(s,is_disrupt=False),intra=True)


def test_failure_restores_model_and_gradient_flags():
    m=model();s=signals(45);original={k:v.clone() for k,v in m.state_dict().items()}
    original_flags={k:p.requires_grad for k,p in m.named_parameters()}
    real=m.four_losses;calls=0
    def fail_after_accepted_step(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==3:raise RuntimeError('injected update failure')
        return real(*args,**kwargs)
    m.four_losses=fail_after_accepted_step
    with pytest.raises(RuntimeError,match='injected'):
        predict_shot(m,s,intra=True)
    for k,v in m.state_dict().items():torch.testing.assert_close(v,original[k],atol=0,rtol=0)
    assert original_flags=={k:p.requires_grad for k,p in m.named_parameters()}
