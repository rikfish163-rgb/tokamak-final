from copy import deepcopy

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

from test_joint_stream_20261107 import signal
from test_risk_meta_step_20261107 import teacher_case


def episode_sequence():
    from experiments.risk_aligned_online import simple_data

    s=signal(320)
    episodes=[]
    for o in (127,159,191,223):
        x,a=simple_data.window(s,o,128)
        episodes.append({k:torch.as_tensor(v[None]) for k,v in
            dict(x=x,a=a,truth=s['values'][o+1:o+26],mask=s['available'][o+1:o+26]).items()})
    qx,qa=simple_data.window(s,249,128)
    return episodes,torch.as_tensor(qx[None]),torch.as_tensor(qa[None])


def test_four_step_sequence_matches_actual_protected_adamw_updates_and_moments():
    from risk_feature_intra_20261107 import update
    from risk_meta_sequence_20261108 import teacher_sequence

    model,_=teacher_case()
    original=deepcopy(model.state_dict())
    native=deepcopy(model)
    native.decoder.requires_grad_(False)
    episodes,qx,qa=episode_sequence()
    optimizer=torch.optim.AdamW([native.weight],lr=1e-4,weight_decay=0.,foreach=False)
    with sdpa_kernel(SDPBackend.MATH):
        result=teacher_sequence(model,episodes,qx,qa,torch.ones(1))
        for i,e in enumerate(episodes):
            update(native,optimizer,e)
            torch.testing.assert_close(result['steps'][i]['weight'],native.weight,atol=2e-7,rtol=0)
            state=optimizer.state[native.weight]
            torch.testing.assert_close(result['steps'][i]['exp_avg'],state['exp_avg'],atol=2e-7,rtol=1e-5)
            torch.testing.assert_close(result['steps'][i]['exp_avg_sq'],state['exp_avg_sq'],atol=2e-7,rtol=1e-5)
        torch.testing.assert_close(result['logit'],native.risk(qx,qa)['logit'],atol=3e-6,rtol=0)
    assert len(result['steps'])==4
    assert result['priors'][0].item()==0 and result['priors'][-1].item()>0
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,original[key],atol=0,rtol=0)


def test_stopping_only_outer_risk_credit_preserves_the_complete_forward_sequence():
    from risk_meta_sequence_20261108 import teacher_sequence

    model,_=teacher_case();episodes,qx,qa=episode_sequence()
    with sdpa_kernel(SDPBackend.MATH):
        full=teacher_sequence(model,episodes,qx,qa,torch.ones(1))
        gradients=torch.autograd.grad(full['risk_loss'],tuple(model.decoder.parameters()))
        stopped=teacher_sequence(model,episodes,qx,qa,torch.ones(1),credit='none')
        zero=torch.autograd.grad(stopped['risk_loss'],tuple(model.decoder.parameters()),allow_unused=True)
    torch.testing.assert_close(full['logit'],stopped['logit'],atol=0,rtol=0)
    assert all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients)>0
    assert all(g is None or g.count_nonzero()==0 for g in zero)
    for a,b in zip(full['steps'],stopped['steps'],strict=True):
        torch.testing.assert_close(a['weight'],b['weight'],atol=0,rtol=0)


def test_last_step_credit_preserves_full_forward_but_changes_outer_gradient():
    from risk_meta_sequence_20261108 import teacher_sequence

    model,_=teacher_case();episodes,qx,qa=episode_sequence()
    parameters=tuple(model.decoder.parameters())
    with sdpa_kernel(SDPBackend.MATH):
        full=teacher_sequence(model,episodes,qx,qa,torch.ones(1))
        all_gradients=torch.autograd.grad(full['risk_loss'],parameters)
        last=teacher_sequence(model,episodes,qx,qa,torch.ones(1),credit='last')
        last_gradients=torch.autograd.grad(last['risk_loss'],parameters)
    torch.testing.assert_close(full['logit'],last['logit'],atol=0,rtol=0)
    assert all(torch.isfinite(g).all() for g in last_gradients)
    assert sum(float(g.abs().sum()) for g in last_gradients)>0
    assert sum(float((a-b).abs().sum()) for a,b in zip(all_gradients,last_gradients,strict=True))>0


def test_outer_labels_never_enter_any_inner_update():
    from risk_meta_sequence_20261108 import teacher_sequence

    model,_=teacher_case();episodes,qx,qa=episode_sequence()
    with sdpa_kernel(SDPBackend.MATH):
        zero=teacher_sequence(model,episodes,qx,qa,torch.zeros(1))
        one=teacher_sequence(model,episodes,qx,qa,torch.ones(1))
    for a,b in zip(zero['steps'],one['steps'],strict=True):
        torch.testing.assert_close(a['weight'],b['weight'],atol=0,rtol=0)
    torch.testing.assert_close(zero['logit'],one['logit'],atol=0,rtol=0)
    assert zero['risk_loss'].item()!=one['risk_loss'].item()


def test_four_window_source_sampler_keeps_every_target_mature_and_query_causal():
    from run_risk_teacher_20261107 import teacher_pools
    from risk_meta_sequence_20261108 import sequence_episode

    s=signal(500);s['valid']=np.arange(500)<490
    s['y']=(np.arange(500)>=380).astype(np.float32)
    pools=teacher_pools([s],steps=4)
    es,qx,qa,label,note=sequence_episode([s],pools,np.random.default_rng(7),'0_to_1','cpu')
    q=note['query']
    assert len(es)==4 and q==note['origins'][-1]+26
    assert note['label_time']==q+25
    torch.testing.assert_close(qx,torch.from_numpy(s['values'][q-127:q+1])[None])
    for o,e in zip(note['origins'],es,strict=True):
        assert s['valid'][o-127:o+26].all() and o+25<q
        assert not e['a'][...,2].any() and not e['mask'][...,2].any()
        torch.testing.assert_close(e['truth'],torch.from_numpy(s['values'][o+1:o+26])[None])
        assert not {'y','label','is_disrupt','down'}&set(e)
    assert label.tolist()==[1.] and not qa[...,2].any()
