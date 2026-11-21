import copy

import numpy as np
import pytest
import torch

from ..ffn_intra_20261005.test_meta import case
from ..ffn_intra_20261005.stream import predict_shot
from .matched_meta import enable_head_only
from .whole_meta import select_peak, causal_logit, whole_risk, train_pair
from . import whole_meta
from . import prepare_whole
from ..ffn_intra_20261005.prepare import full_source


def test_peak_uses_whole_safe_and_strict_timely_set_with_first_tie():
    _, shot, _ = case()
    scores = np.zeros(160, np.float32)
    scores[129:132] = .9  # leads 31,30,29ms: only129 eligible for D
    scores[10] = .9
    assert select_peak(shot, scores) == 10
    shot['valid'][10] = False
    assert select_peak(shot, scores) == 129
    shot['is_disrupt'] = False
    scores[150] = .95
    assert select_peak(shot, scores) == 150
    shot['valid'][:] = False
    with pytest.raises(ValueError, match='eligible'):
        select_peak(shot, scores)


@pytest.mark.parametrize('query,events', [(0,0),(25,0),(26,1),(51,1),(52,2),(78,3),(104,4),(156,6)])
def test_causal_query_matches_emitted_score_and_update_clock(query, events):
    model, shot, _ = case(); enable_head_only(model)
    scores, _ = predict_shot(model, shot, intra=True, step=.1, radius=1., offline_source=True)
    logit, row = causal_logit(model, shot, query, step=.1, radius=1., create_graph=True)
    assert float(logit.detach().sigmoid()) == pytest.approx(float(scores[query]), abs=3e-7)
    assert row['mature_events'] == events
    assert row['unrolled_mature_slots'] == min(4,events)
    assert row['differentiated_updates'] == min(4,events)
    assert row['burn_in_events'] == max(0,events-4)
    if events == 0:
        grad, = torch.autograd.grad(logit, model.base.forecast_head.weight, allow_unused=True)
        assert grad is None


def test_unarrived_suffix_and_outcome_cannot_change_causal_logit():
    model, shot, _ = case(); enable_head_only(model)
    first, _ = causal_logit(model, shot, 52, step=.1, radius=1.)
    altered = copy.deepcopy(shot)
    altered['values'][53:] = 1e4
    altered['down'] = 999.; altered['is_disrupt'] = False
    second, _ = causal_logit(model, altered, 52, step=.1, radius=1.)
    torch.testing.assert_close(first, second, rtol=0,atol=0)


def test_missing_target_events_skip_exactly_as_actual_stream():
    model, shot, _ = case(); enable_head_only(model)
    shot['available'][1:26] = False
    scores, _ = predict_shot(model, shot, intra=True, step=.1, radius=1., offline_source=True)
    logit, row = causal_logit(model, shot, 104, step=.1, radius=1.)
    assert row['missing_target_events'] == 1
    assert float(logit.detach().sigmoid()) == pytest.approx(float(scores[104]), abs=3e-7)


@pytest.mark.parametrize('parameter', ['weight','bias'])
def test_risk_only_head_derivative_with_zero_burn_in(parameter):
    model, shot, _ = case(); model=model.double(); enable_head_only(model)
    p = getattr(model.base.forecast_head, parameter)
    torch.manual_seed(51); direction=torch.randn_like(p); direction/=direction.norm()
    original = p.detach().clone()
    def value(offset, graph=False):
        with torch.no_grad(): p.copy_(original+offset*direction)
        z,_ = causal_logit(model,shot,104,step=.1,radius=1.,create_graph=graph)
        return torch.nn.functional.softplus(-z)
    loss=value(0.,True)
    gradient,=torch.autograd.grad(loss,p)
    analytical=float((gradient*direction).sum())
    numerical=float((value(1e-4)-value(-1e-4)).detach())/2e-4
    assert abs(analytical)>1e-9
    assert analytical==pytest.approx(numerical,rel=.005,abs=3e-8)


def test_whole_loss_matches_sealed_peak_probability():
    model,shot,_=case();enable_head_only(model)
    for label in (False,True):
        shot['is_disrupt']=label
        scores,_=predict_shot(model,shot,intra=True,step=.1,radius=1.,offline_source=True)
        q=select_peak(shot,scores)
        p=float(np.clip(scores[q],1e-7,1-1e-7))
        margin=np.log(p/(1-p))-.97
        expected=np.logaddexp(0.,-margin if label else margin)
        loss,row=whole_risk(model,shot,step=.1,radius=1.,boundary=.97,create_graph=True)
        assert float(loss.detach())==pytest.approx(expected,abs=5e-7)
        assert row['peak_index']==q


def test_head_update_preserves_classifier_and_requires_balanced_pair():
    model,shot,e=case();enable_head_only(model)
    safe=copy.deepcopy(shot);safe['shot']=2;safe['is_disrupt']=False
    se=copy.deepcopy(e);se.update(shot=2,label=0)
    before={k:v.clone() for k,v in model.state_dict().items()}
    opt=torch.optim.Adam(model.base.forecast_head.parameters(),lr=3e-5)
    with pytest.raises(ValueError,match='balanced'):
        train_pair(model,[(shot,e)],opt,step=.1,radius=1.,boundary=0.)
    result=train_pair(model,[(shot,e),(safe,se)],opt,step=.1,radius=1.,boundary=0.)
    assert result['head_gradient_norm']>0
    assert not torch.equal(before['base.forecast_head.weight'],model.base.forecast_head.weight)
    for k,v in model.state_dict().items():
        if not k.startswith('base.forecast_head.'):
            assert torch.equal(v,before[k])


def test_target_data_and_invalid_index_rejected():
    model,shot,_=case()
    with pytest.raises(ValueError,match='query'):
        causal_logit(model,shot,160,step=.1,radius=1.)
    shot['device']='EAST'
    with pytest.raises(ValueError,match='source'):
        whole_risk(model,shot,step=.1,radius=1.,boundary=0.)


def test_truncated_gradient_holds_actual_burn_in_fixed_for_finite_difference(monkeypatch):
    model,shot,_=case();model=model.double();enable_head_only(model)
    real_step=whole_meta.signal_step;captured=[]
    def capture(*args,**kwargs):
        result=real_step(*args,**kwargs)
        captured.append(tuple(t.detach().clone() for t in result))
        return result
    monkeypatch.setattr(whole_meta,'signal_step',capture)
    z,row=causal_logit(model,shot,156,step=.1,radius=1.,create_graph=True)
    assert row['burn_in_events']==2
    p=model.base.forecast_head.weight
    torch.manual_seed(61);direction=torch.randn_like(p);direction/=direction.norm()
    original=p.detach().clone()
    gradient,=torch.autograd.grad(torch.nn.functional.softplus(-z),p)
    analytical=float((gradient*direction).sum())
    # Finite differences must freeze the same real delta at the truncation
    # boundary, otherwise they include a deliberately omitted burn-in derivative.
    def value(offset):
        counter=[0]
        def fixed_prefix(*args,**kwargs):
            j=counter[0];counter[0]+=1
            return captured[j] if j<2 else real_step(*args,**kwargs)
        monkeypatch.setattr(whole_meta,'signal_step',fixed_prefix)
        with torch.no_grad():p.copy_(original+offset*direction)
        score,_=causal_logit(model,shot,156,step=.1,radius=1.)
        return float(torch.nn.functional.softplus(-score).detach())
    numerical=(value(1e-4)-value(-1e-4))/2e-4
    assert abs(analytical)>1e-9
    assert analytical==pytest.approx(numerical,rel=.005,abs=3e-8)


@pytest.mark.parametrize('parameter',['weight','bias'])
def test_whole_risk_itself_has_head_gradient_and_early_negative_control(parameter):
    model,shot,_=case();enable_head_only(model)
    shot['valid'][:]=False;shot['valid'][104]=True
    risk,row=whole_risk(model,shot,step=.1,radius=1.,boundary=.97,create_graph=True)
    gradient,=torch.autograd.grad(risk,getattr(model.base.forecast_head,parameter))
    assert row['peak_index']==104
    assert torch.isfinite(gradient).all() and gradient.norm()>1e-8
    shot['valid'][:]=False;shot['valid'][0]=True
    early,_=whole_risk(model,shot,step=.1,radius=1.,boundary=.97,create_graph=True)
    no_path,=torch.autograd.grad(early,getattr(model.base.forecast_head,parameter),allow_unused=True)
    assert no_path is None


@pytest.mark.parametrize('bias',[-100.,0.,100.])
def test_whole_risk_matches_actual_whole_evaluator_including_saturation(bias):
    model,shot,e=case();enable_head_only(model)
    with torch.no_grad():
        model.base.risk_head.weight.zero_();model.base.risk_head.bias.fill_(bias)
    safe=copy.deepcopy(shot);safe['is_disrupt']=False;safe['shot']=2
    losses=[float(whole_risk(model,s,step=.1,radius=1.,boundary=np.log(.7/.3))[0].detach())
            for s in (shot,safe)]
    actual=full_source(model,[(shot,e),(safe,e)],step=.1,radius=1.,rule=None,threshold=.7)
    assert np.mean(losses)==pytest.approx(actual['balanced_loss'],abs=2e-7)


def test_missing_targets_distinguish_scheduled_slots_and_executed_updates():
    model,shot,_=case();enable_head_only(model)
    shot['available'][:]=False
    z,row=causal_logit(model,shot,156,step=.1,radius=1.,create_graph=True)
    assert row['mature_events']==6
    assert row['executed_updates']==row['differentiated_updates']==0
    gradient,=torch.autograd.grad(z,model.base.forecast_head.weight,allow_unused=True)
    assert gradient is None
    shot['available'][:52]=True
    z,row=causal_logit(model,shot,156,step=.1,radius=1.,create_graph=True)
    assert row['executed_updates']==2 and row['differentiated_updates']==0
    gradient,=torch.autograd.grad(z,model.base.forecast_head.weight,allow_unused=True)
    assert gradient is None  # Updates exist, but solely in the detached burn-in.


def test_no_admission_published_after_failed_final_integrity_check(tmp_path,monkeypatch):
    monkeypatch.setattr(prepare_whole,'_hash',lambda path:'changed')
    with pytest.raises(RuntimeError,match='changed'):
        prepare_whole.publish_admitted(tmp_path,{'weight':torch.zeros(1)}, {},20,'expected',{})
    assert not (tmp_path/'admitted.pt').exists()
