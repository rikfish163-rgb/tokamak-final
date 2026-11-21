import copy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.increment_forecast_20261023.learning import IncrementFeedback
from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
from experiments.risk_aligned_online.tests.test_task_geometry import small_model
from experiments.risk_aligned_online.attention_scope_20261023.learning import (
    ARMS, masks, attention_probe, update_all, predict,
)


def make_model():
    parent=small_model();m=IncrementFeedback(copy.deepcopy(parent.base),'increment')
    m.load_state_dict(parent.state_dict());m.set_stage('frozen')
    return m


def signal():
    return dict(values=np.random.default_rng(6).normal(size=(400,23)).astype('float32'),
                available=np.ones((400,23),bool))


def test_masks_modify_qk_or_v_but_never_bias_and_fail_on_wrong_shape():
    m=make_model();fast=initial_encoder(m);d=m.cfg.dim
    q,v=masks(fast,d)
    assert sum(int(t.sum()) for t in q.values())==2*d*d
    assert sum(int(t.sum()) for t in v.values())==d*d
    assert all(not (q[k]&v[k]).any() for k in q)
    assert not q['base.encoder.layers.0.self_attn.in_proj_bias'].any()
    with pytest.raises(ValueError):masks(fast,d+1)


def test_attention_probe_matches_pytorch_and_qk_keeps_values_identical():
    m=make_model();fast=initial_encoder(m);s=signal()
    x=torch.tensor(s['values'][None,:128]);a=torch.tensor(s['available'][None,:128])
    before=attention_probe(m,x,a,fast)
    with torch.no_grad():
        _,actual=m.base.encoder.layers[0].self_attn(before['normalized'],before['normalized'],before['normalized'],
            key_padding_mask=~before['valid'],need_weights=True,average_attn_weights=False)
    torch.testing.assert_close(before['attention'],actual,rtol=1e-5,atol=1e-7)
    q,_=masks(fast,m.cfg.dim)
    changed={k:(v+.01*q[k]).detach().requires_grad_(True) for k,v in fast.items()}
    after=attention_probe(m,x,a,changed)
    torch.testing.assert_close(before['value'],after['value'],rtol=0,atol=0)
    assert not torch.equal(before['attention'],after['attention'])


def test_updates_obey_scope_and_match_requested_norm_without_label_input():
    m=make_model();s={k:v[:256] for k,v in signal().items()}
    states={a:initial_encoder(m) for a in ARMS}
    after,receipts=update_all(m,s,256,states,device='cpu')
    q,v=masks(states['qk'],m.cfg.dim)
    for arm,mask in (('qk',q),('value_matched',v)):
        for k in states[arm]:
            torch.testing.assert_close(after[arm][k][~mask[k]],states[arm][k][~mask[k]],rtol=0,atol=0)
    length=receipts['qk']['requested_step_norm']
    assert length>0
    for a in ARMS:
        assert receipts[a]['requested_step_norm']==length
        assert receipts[a]['parameter_delta_norm']==pytest.approx(length,rel=2e-3,abs=2e-7)
        assert receipts[a]['last_observation']==255 and receipts[a]['first_affected_prediction']==256
    assert receipts['qk']['value_change_max']==0
    assert receipts['value_matched']['attention_change_max']==0
    with pytest.raises(ValueError):update_all(m,dict(s,y=1),256,states,device='cpu')


def test_empty_targets_no_update_and_stream_causality_reset():
    m=make_model();s=signal();original={k:v.clone() for k,v in m.state_dict().items()}
    empty={k:v[:256].copy() for k,v in s.items()};empty['available'][:]=False
    states={a:initial_encoder(m) for a in ARMS}
    same,u=update_all(m,empty,256,states,device='cpu')
    assert all(r['gradient_steps']==0 for r in u.values())
    for a in ARMS:
        for k in same[a]:torch.testing.assert_close(same[a][k],states[a][k],rtol=0,atol=0)
    first,updates=predict(m,s,device='cpu');again,again_updates=predict(m,s,device='cpu')
    assert updates==again_updates
    for a in first:np.testing.assert_array_equal(first[a],again[a])
    s['values'][256:]=1000
    changed,cu=predict(m,s,device='cpu')
    for a in ARMS:
        np.testing.assert_array_equal(first[a][:256],changed[a][:256])
        assert updates[a][0]==cu[a][0]
    for k,v in m.state_dict().items():torch.testing.assert_close(v,original[k],rtol=0,atol=0)
