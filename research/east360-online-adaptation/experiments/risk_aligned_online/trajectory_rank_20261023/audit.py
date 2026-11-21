"""Separate numerical recount and full replay of the historical screen."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..history_retrieval_20261023.audit import choose_independently
from ..history_retrieval_20261023.learning import sample_queries,query_key
from ..history_retrieval_20261023.run import prepare
from ..simple_online import _durable_json
from ..simple_train import sha
from ..task_geometry_20261023.run import READ,load_model
from .learning import ARMS,observed_trajectory
from .run import build_trajectories,probe,verify_old_row


def check_history_loss(u):
    labels=np.asarray(u['selected_labels'])
    assert labels.tolist()==[0,0,1,1]
    for stage in ('before','after'):
        z=np.asarray(u['historical_logits_'+stage],dtype=float)
        assert z.shape==(4,) and np.isfinite(z).all()
        if u['loss_kind']=='BCE':value=np.mean(np.logaddexp(0.,z)-labels*z)
        else:
            assert u['loss_kind']=='pairwise'
            value=np.mean([np.logaddexp(0.,z[n]-z[p]) for n in (0,1) for p in (2,3)])
        # Saved objectives were reduced in float32; independent calculation is float64.
        assert np.isclose(value,u['loss_'+stage],rtol=2e-6,atol=2e-6)


def recount(rows):
    out={}
    for r in rows:
        assert r['label'] in (0,1)
        for stage in ('before','after'):
            z=r['logit_'+stage];value=np.logaddexp(0.,z)-r['label']*z
            assert np.isfinite(value) and abs(float(value)-r[stage])<2e-14
    for arm in ARMS:
        groups={};per=[]
        for group in ('safe','disruptive_timely','disruptive_early'):
            selected=[r for r in rows if r['arm']==arm and r['group']==group]
            part=[]
            for sid in sorted({r['shot'] for r in selected}):
                values=[r for r in selected if r['shot']==sid]
                item=dict(shot=sid,group=group,n=len(values),
                    before=float(np.mean([r['before'] for r in values])),after=float(np.mean([r['after'] for r in values])))
                part.append(item);per.append(item)
            if part:groups[group]={k:float(np.mean([r[k] for r in part])) for k in ('before','after')}
        primary=[groups[g] for g in ('safe','disruptive_timely')];d=[r for r in per if r['group']=='disruptive_timely']
        improved=sum(r['after']<r['before'] for r in d)
        out[arm]=dict(groups=groups,per_shot=per,balanced_before=float(np.mean([g['before'] for g in primary])),
            balanced_after=float(np.mean([g['after'] for g in primary])),improved_disruptive_shots=improved,
            passes_screen=bool(len(d)==4 and improved>=3 and all(g['after']<=g['before'] for g in primary)
                               and sum(g['after'] for g in primary)<sum(g['before'] for g in primary)))
    c=out['trajectory_rank']
    gate=bool(c['passes_screen'] and all(c['balanced_after']<out[a]['balanced_after'] for a in ARMS if a!='trajectory_rank'))
    return out,gate


def describe(values):
    a=np.asarray(values,dtype=float)
    assert a.ndim==1 and len(a) and np.isfinite(a).all()
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),
                p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)),min=float(a.min()),max=float(a.max()))


def temporal_spread(ordered_keys):
    # Three unit states are concatenated then divided by sqrt(3).
    a=np.asarray(ordered_keys,dtype=float).reshape(-1,3,ordered_keys.shape[-1]//3)*np.sqrt(3.)
    assert np.allclose(np.linalg.norm(a,axis=-1),1.,rtol=1e-10,atol=1e-10)
    return np.mean(np.sum((a-a.mean(1,keepdims=True))**2,axis=-1),axis=1)


def check_key_recount(actual,independent):
    # Per-row vs batched float64 reduction differed by at most1.67e-16 in
    # the first audit. Allow64 machine eps for unit-vector arithmetic only;
    # raw memory rebuild and full neural replay still require exact equality.
    np.testing.assert_allclose(actual,independent,rtol=0,atol=64*np.finfo(np.float64).eps)
    return float(np.max(np.abs(np.asarray(actual)-np.asarray(independent))))


def audit(root):
    if (root/'verification.json').exists():raise FileExistsError(root/'verification.json')
    p,done,rows=READ(root/'protocol.json'),READ(root/'completed.json'),READ(root/'probes.json')
    assert done['status']=='COMPLETED' and not done['test_accessed']
    for f,digest in done['artifact_sha256'].items():assert sha(root/f)==digest,f
    for f,digest in p['code_sha256'].items():assert sha(io23.ROOT/f)==digest,f
    old,ref=Path(p['old_reference']),Path(p['preparation_reference'])
    for f,digest in p['old_hashes'].items():assert sha(old/f)==digest,f
    for f,k in (('parent','parent_sha256'),('source_checkpoint','source_sha256'),('plan_path','plan_sha256')):
        assert sha(Path(p[f]))==p[k]
    assert sha(ref/'input_map.json')==p['input_map_sha256']
    ids=p['selected_ids'];assert len(set(ids))==20 and not set(ids)&set(READ(ref/'protocol.json')['contract']['shot_ids'])
    assert READ(root/'label_access.json')==[dict(device='EAST',shot=sid) for sid in ids]
    summary,gate=recount(rows);assert summary==done['summary'] and gate==done['recommend_full_stream']
    assert len(rows)==3840 and done['queries']==640 and done['historical_windows']==640
    lookup={(r['shot'],r['query'],r['arm']):r for r in rows};assert len(lookup)==len(rows)
    before_lookup={(r['shot'],r['query']):r for r in READ(old/'probes.json') if r['arm']=='nearest'}
    stats={a:dict(history_loss_decreased=0,actual_updates=0,query_harmful_dot=0,
                  changed_selection_vs_state=0,changed_selection_vs_trajectory=0,groups={},
                  common_shift=[],separation_change=[]) for a in ARMS}
    for r in rows:
        u=r['update'];check_history_loss(u);s=stats[r['arm']]
        assert u['last_observation']==r['count']-1 and u['first_affected_prediction']==r['count']
        assert u['retrieval_ends']==[r['count']-51,r['count']-26,r['count']-1]
        assert r['query']==r['count']+24 and u['retrieval_ends'][0]>=127
        assert r['shot'] not in u['selected_shots']
        assert len(set(u['selected_shots'][:2]))==len(set(u['selected_shots'][2:]))==2
        assert u['gradient_steps']==int(u['parameter_delta_norm']>0) and u['parameter_delta_norm']<=.00101
        s['actual_updates']+=u['gradient_steps'];s['history_loss_decreased']+=int(u['loss_after']<u['loss_before'])
        s['query_harmful_dot']+=int(r['gradient_dot']<0)
        group=s['groups'].setdefault(r['group'],dict(n=0,harmful_dot=0,actual_bce_worse=0))
        group['n']+=1;group['harmful_dot']+=int(r['gradient_dot']<0);group['actual_bce_worse']+=int(r['after']>r['before'])
        other=lookup[(r['shot'],r['query'],'state_bce')];traj=lookup[(r['shot'],r['query'],'trajectory_rank')]
        s['changed_selection_vs_state']+=int(u['selected_indices']!=other['update']['selected_indices'])
        s['changed_selection_vs_trajectory']+=int(u['selected_indices']!=traj['update']['selected_indices'])
        z0,z1=(np.asarray(u['historical_logits_'+stage]) for stage in ('before','after'))
        s['common_shift'].append(float((z1-z0).mean()))
        s['separation_change'].append(float((z1[2:].mean()-z1[:2].mean())-(z0[2:].mean()-z0[:2].mean())))
        if r['arm']=='state_bce':verify_old_row(r,before_lookup[(r['shot'],r['query'])])
    assert sum(s['actual_updates'] for s in stats.values())==done['neural_steps']
    for s in stats.values():
        for k in ('common_shift','separation_change'):s[k]=describe(s[k])
    cached=torch.load(root/'memory.pt',map_location='cpu',weights_only=True)
    torch.set_num_threads(2);torch.manual_seed(131);model,threshold=load_model();origin=state_cpu(model)
    assert threshold==p['threshold']
    readshot,outcome=simple_data.read_shot,io23.read_outcome;access=[]
    def allowed_signal(device,sid,**kwargs):
        assert device=='EAST' and sid in ids,'only authorized history signals'
        return readshot(device,sid,**kwargs)
    def allowed_outcome(device,sid):
        assert device=='EAST' and sid in ids,'only authorized history outcomes'
        access.append(sid);return outcome(device,sid)
    replay=[];query_spread=[];length_diffs=[];sampling=[]
    with patch.object(simple_data,'read_shot',side_effect=allowed_signal),patch.object(io23,'read_outcome',side_effect=allowed_outcome):
        memory,manifest,records,signals=prepare(model,ids,READ(ref/'input_map.json'))
        memory,states=build_trajectories(model,memory,manifest,signals)
        torch.testing.assert_close(torch.as_tensor(states),cached['states'],rtol=0,atol=0)
        for k,v in memory.items():torch.testing.assert_close(torch.as_tensor(v).cpu(),cached['memory'][k],rtol=0,atol=0)
        unit=states.astype(float)/np.linalg.norm(states.astype(float),axis=-1,keepdims=True)
        ordered=unit.reshape(len(unit),-1);ordered/=np.linalg.norm(ordered,axis=-1,keepdims=True)
        pooled=unit.mean(1);pooled/=np.linalg.norm(pooled,axis=-1,keepdims=True)
        key_recount_error=max(check_key_recount(memory['trajectory_keys'],ordered),
                              check_key_recount(memory['pooled_keys'],pooled))
        for sid in ids:
            shot=records[sid];signal=signals[sid]
            for index,row in enumerate(sample_queries(shot)):
                q=row['query'];y=int(shot['y'][q]);entry=dict(row,shot=sid,index=index)
                sampling.append(dict(entry,label=y,lead_ms=round(float((shot['down']-shot['times'][q])*1000),9) if shot['is_disrupt'] else None))
                again=probe(model,signal,entry,y,memory)
                key=query_key(model,signal,row['count'],device='cuda')
                tr,_=observed_trajectory(model,signal,row['count'],device='cuda')
                query_spread.append(dict(shot=sid,query=q,group=row['group'],spread=float(temporal_spread(tr['trajectory'][None])[0])))
                lengths=[]
                for r in again:
                    oldrow=lookup[(sid,q,r['arm'])];assert r==oldrow,(sid,q,r['arm'])
                    if r['arm'].startswith('state_'):keys,query=memory['keys'],key
                    elif r['arm']=='pooled_rank':keys,query=memory['pooled_keys'],tr['pooled']
                    else:
                        keys,query=memory['trajectory_keys'],tr['trajectory']
                        if r['arm']=='reversed_rank':query=query.reshape(3,-1)[::-1].reshape(-1)
                    selected,distances=choose_independently(dict(memory,keys=keys),query,'nearest',sid)
                    assert selected==r['update']['selected_indices']
                    np.testing.assert_allclose(distances,r['update']['distances'],rtol=0,atol=2e-14)
                    lengths.append(r['update']['parameter_delta_norm'])
                length_diffs.append(max(lengths)-min(lengths))
            replay.append(dict(shot=sid,queries=32,arms=6,exact=True));print(dict(replayed=sid,rows=len(replay)*192),flush=True)
    assert sampling==READ(root/'sampling.json')==READ(old/'sampling.json') and access==ids
    assert max(length_diffs)<2e-7;assert_unchanged(model,origin)
    memory_spread=temporal_spread(memory['trajectory_keys'])
    temporal=dict(history={g:describe([v for v,m in zip(memory_spread,manifest) if m['group']==g])
                           for g in ('safe','disruptive_timely','disruptive_early')},
                  current={g:describe([r['spread'] for r in query_spread if r['group']==g])
                           for g in ('safe','disruptive_timely','disruptive_early')},records=query_spread)
    _durable_json(root/'temporal_diagnostic.json',temporal)
    report=dict(status='VERIFIED',completed_sha256=sha(root/'completed.json'),audit_sha256=sha(Path(__file__)),
        audit_tests_sha256=sha(io23.ROOT/'experiments/risk_aligned_online/tests/test_trajectory_rank_audit.py'),
        temporal_diagnostic_sha256=sha(root/'temporal_diagnostic.json'),summary=summary,primary_gate=gate,
        independent_bce_recounts=len(rows),independent_retrieval_recounts=len(rows),
        exact_cuda_replayed_rows=len(rows),replay=replay,exact_old_baseline_rows=640,
        memory_rebuilt=True,parent_unchanged=True,no_test_access=True,
        max_step_length_difference=max(length_diffs),independent_key_max_abs_error=key_recount_error,mechanism=stats,
        caveat='same implementation CUDA replay; no full stream; parent trained all20; not independent validation')
    _durable_json(root/'verification.json',report)
    print(dict(status='VERIFIED',rows=len(rows),gate=gate),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);audit(p.parse_args().root)
