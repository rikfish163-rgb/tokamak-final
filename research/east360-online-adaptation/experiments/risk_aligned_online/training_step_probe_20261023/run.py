"""Replay preparation and inspect bounded one-step omissions, not a policy."""
import argparse
import copy
import math
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..asymmetric_feedback_20261023.learning import routed_episode
from ..causal_input import apply_input_contract
from ..credit_horizon_20261023.run import READ,REFERENCE,load_model,schedule
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..east_joint_feedback_20261023.learning import enable,objective
from ..east_joint_source_replay_20261023.learning import replay_loss,validate_source
from ..labeled_alignment_20261023.learning import balanced_batch
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256,load_source_train
from ..simple_online import _durable_json
from ..simple_train import pools,sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import probe_steps,subtract

PREVIOUS=io23.ROOT/'experiments/risk_aligned_online/runs/asymmetric_feedback_cv20_20261023'
PREVIOUS_SHA='7f4bda909ac80ad9c3346738737cb7ca47a4555338283af60e8d5f1effcbb6db'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/准备阶段逐步归因_分析约定_20261023.md'


def state_context(s,end,model):
    x,m=simple_data.window(s,end,128)
    center=model.base.center.detach().cpu().numpy();scale=model.base.scale.detach().cpu().numpy()
    v=np.arcsinh((x-center)/scale);seen=m.any(0)
    median=[float(np.median(v[m[:,j],j])) if seen[j] else None for j in range(23)]
    return dict(end=int(end),time=float(s['times'][end]),available=seen.tolist(),asinh_window_median=median)


def anchors(plan,model,mapping,threshold):
    result=[];windows=[]
    for sid in plan['held']:
        s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
        s=dict(s,**transform({k:s[k] for k in ('values','available')},mapping))
        valid=s['valid'].copy()
        if s['is_disrupt']:valid &= np.round((s['down']-s['times'])*1000.,9)>30.
        ends=np.flatnonzero(valid)
        if not len(ends):raise ValueError('no eligible diagnostic endpoints')
        with np.load(REFERENCE/f'fold{plan["fold"]}'/'shots'/f'{sid}.npz') as z:
            prob=z['static'];cross=ends[prob[ends]>=threshold]
            alarm=s['is_disrupt'] and len(cross)>0
            end=int(cross[0] if alarm else ends[np.argmax(prob[ends])])
            archived=float(prob[end])
        windows.append(simple_data.window(s,end,128))
        result.append(dict(shot=sid,disruptive=s['is_disrupt'],end=end,
            lead_ms=float((s['down']-s['times'][end])*1000.) if s['is_disrupt'] else None,
            choice='first_timely_alarm' if alarm else 'fixed_original_peak',original_probability=archived,
            context=state_context(s,end,model)))
    x,m=[torch.as_tensor(np.stack(v),device='cuda') for v in zip(*windows)]
    return result,x,m


def main(root):
    if root.exists():raise FileExistsError(root)
    assert torch.cuda.is_available()
    assert sha(PREVIOUS/'completed.json')==PREVIOUS_SHA
    v=READ(PREVIOUS/'verification.json');assert v['status']=='VERIFIED' and v['completed_sha256']==PREVIOUS_SHA
    previous=READ(PREVIOUS/'completed.json');p=READ(PREVIOUS/'protocol.json')
    for name,digest in previous['artifact_sha256'].items():assert sha(PREVIOUS/name)==digest
    for name,digest in p['code_sha256'].items():assert sha(io23.ROOT/name)==digest
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic();root.mkdir(parents=True)
    _durable_json(root/'protocol.json',dict(reference_sha256=PREVIOUS_SHA,plan_sha256=sha(PLAN),
        code_sha256={str(q.relative_to(io23.ROOT)):sha(q) for q in Path(__file__).parent.glob('*.py')},
        source_sha256=PARENT_SHA256,folds=p['folds'],threshold=p['threshold'],
        scope='POST_HOC_DEVELOPMENT_DIAGNOSTIC; held outcomes select fixed explanation endpoints only',
        test360_accessed=False,training_controls_unchanged=True))
    read_outcome=io23.read_outcome;access=[];source_ids=io23.source_roles()['train']
    def source_guard(device,sid):
        assert device=='J-TEXT' and sid in source_ids
        access.append(dict(device=device,shot=sid,phase='source_train'));return read_outcome(device,sid)
    with patch.object(io23,'read_outcome',side_effect=source_guard):source,_=load_source_train()
    validate_source(source,source_ids);groups=pools(source);all_summary=[]
    for plan in p['folds']:
        f=plan['fold'];out=root/f'fold{f}';out.mkdir();folder=PREVIOUS/f'fold{f}'
        mapping=READ(folder/'input_map.json');warm=load_model(f);warm_state=state_cpu(warm)
        phase='fit'
        def outcome_guard(device,sid):
            allowed=plan['fit'] if phase=='fit' else plan['held']
            assert device=='EAST' and sid in allowed and phase in ('fit','posthoc_anchor')
            access.append(dict(device=device,shot=sid,fold=f,phase=phase));return read_outcome(device,sid)
        with patch.object(io23,'read_outcome',side_effect=outcome_guard):
            shots=[]
            for sid in plan['fit']:
                s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
                shots.append(dict(s,**transform({k:s[k] for k in ('values','available')},mapping)))
            entries,_=schedule(shots,plan['fit'],f);assert entries==READ(folder/'schedule.json')
            phase='posthoc_anchor';points,ax,am=anchors(plan,warm,mapping,p['threshold'])
            phase='replay_no_outcome_access'
            byid={s['shot']:s for s in shots};history=READ(folder/'training.json')
            _durable_json(out/'anchors.json',points)
            models={a:copy.deepcopy(warm) for a in ('joint','meta')};params={};opts={};before_logits={};rows=[]
            for a,m in models.items():
                names=enable(m);params[a]=[v for k,v in m.named_parameters() if k in names]
                opts[a]=torch.optim.Adam([
                    dict(params=[v for k,v in m.named_parameters() if k in m.encoder_names()],lr=1e-5),
                    dict(params=list(m.base.forecast_head.parameters()),lr=1e-4)])
                with torch.no_grad():before_logits[a]=m.base.risk_head(m(ax,am,mode='risk_state')['risk_state']).flatten()
                np.testing.assert_allclose(before_logits[a].sigmoid().cpu().numpy(),[r['original_probability'] for r in points],atol=2e-6,rtol=2e-6)
            boundary=math.log(p['threshold']/(1-p['threshold']));rng=np.random.default_rng(9131+f)
            for item,old in zip(entries,history):
                sx,sm,sy,sr=balanced_batch(source,groups,rng,warm.cfg);assert sr==old['source_examples']
                row=dict(step=item['step'],episodes=item['episodes'],source_examples=sr,
                    endpoint_contexts=[state_context(byid[e['shot']],e['ends'][-1],warm) for e in item['episodes']])
                for a,m in models.items():
                    opt=opts[a];pars=params[a];opt.zero_grad(set_to_none=True);parts={}
                    for e,receipt in zip(item['episodes'],old[a]['episodes']):
                        s=byid[e['shot']]
                        r=routed_episode(m,{k:s[k] for k in ('values','available')},e['count'],e['ends'],e['disruptive'],
                            boundary=boundary,track=a=='meta',device='cuda')
                        loss=objective(r,a)
                        for key,oldkey in (('before','risk_before'),('after','risk_after'),('forecast','forecast')):
                            assert float(r[key].detach())==receipt[oldkey]
                        assert float(loss.detach())==receipt['objective']
                        grad=torch.autograd.grad(loss/2,pars,retain_graph=True,allow_unused=True)
                        parts['east_disrupt' if e['disruptive'] else 'east_safe']=[torch.zeros_like(q) if g is None else g.detach() for q,g in zip(pars,grad)]
                        (loss/2).backward();del r,loss,grad
                    sl=replay_loss(m,sx,sm,sy);assert float(sl.detach())==old[a]['source_bce']
                    grad=torch.autograd.grad(sl,pars,retain_graph=True,allow_unused=True)
                    parts['source']=[torch.zeros_like(q) if g is None else g.detach() for q,g in zip(pars,grad)]
                    sl.backward();full=[q.grad.detach().clone() for q in pars]
                    for j,g in enumerate(full):torch.testing.assert_close(sum(v[j] for v in parts.values()),g,rtol=3e-6,atol=2e-7)
                    def reader():return m.base.risk_head(m(ax,am,mode='risk_state')['risk_state']).flatten()
                    with torch.no_grad():before=reader().detach()
                    cases={'full':full,**{'omit_'+k:subtract(full,g) for k,g in parts.items()},
                           'zero_current':[torch.zeros_like(g) for g in full]}
                    probes=probe_steps(m,opt,pars,cases,reader)
                    norm=torch.nn.utils.clip_grad_norm_(pars,1.,error_if_nonfinite=True)
                    assert float(norm)==old[a]['grad_norm'];opt.step()
                    with torch.no_grad():after=reader().detach()
                    torch.testing.assert_close(after,probes['full'],rtol=0,atol=0)
                    row[a]=dict(before_logits=before.cpu().tolist(),after_logits=after.cpu().tolist(),
                        hypothetical_logits={k:q.cpu().tolist() for k,q in probes.items()})
                    del parts,sl,grad,full,probes,cases
                rows.append(row)
                if (item['step']+1)%8==0:
                    _durable_json(out/'trace.json',rows)
                    print(dict(fold=f,step=item['step']+1,seconds=time.monotonic()-start),flush=True)
            for a,m in models.items():
                saved=torch.load(folder/f'{a}.pt',map_location='cpu',weights_only=True)['model']
                assert_unchanged(m,saved)
                for j,point in enumerate(points):
                    changes=np.array([r[a]['after_logits'][j]-r[a]['before_logits'][j] for r in rows])
                    first=rows[0][a]['before_logits'][j];last=rows[-1][a]['after_logits'][j]
                    assert abs(float(changes.sum())-(last-first))<1e-10
                    omitted={}
                    for name in ('east_safe','east_disrupt','source','zero_current'):
                        key=name if name=='zero_current' else 'omit_'+name
                        delta=np.array([r[a]['hypothetical_logits'][key][j]-r[a]['after_logits'][j] for r in rows])
                        omitted[name]=dict(higher_count=int((delta>0).sum()),lower_count=int((delta<0).sum()),median=float(np.median(delta)),sum=float(delta.sum()))
                    all_summary.append(dict(fold=f,arm=a,**point,initial_logit=first,final_logit=last,
                        initial_probability=float(torch.sigmoid(torch.tensor(first,dtype=torch.float64))),
                        final_probability=float(torch.sigmoid(torch.tensor(last,dtype=torch.float64))),
                        true_logit_change=last-first,down_steps=int((changes<0).sum()),omission_minus_full=omitted,
                        largest_decrease_steps=np.argsort(changes)[:5].tolist()))
                m.set_stage('frozen')
            assert_unchanged(warm,warm_state)
        _durable_json(out/'trace.json',rows)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(root/'access.json',access)
    done=dict(status='REPLAYED',reference_sha256=PREVIOUS_SHA,rows=all_summary,exact_checkpoint_matches=8,
        exact_one_step_probe_matches=512,test360_accessed=False,source_unchanged=True,
        seconds=time.monotonic()-start,scope='one-step conditional omission with previous Adam history retained; not cumulative data-removal causality',
        artifact_sha256={str(q.relative_to(root)):sha(q) for q in root.rglob('*') if q.is_file()})
    _durable_json(root/'completed.json',done)
    print({k:v for k,v in done.items() if k not in ('rows','artifact_sha256')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
