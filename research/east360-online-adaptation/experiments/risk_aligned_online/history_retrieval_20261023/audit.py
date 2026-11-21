"""Separate recount, full CUDA replay and post-hoc gradient decomposition.

The future query label is used ONLY for mechanism diagnosis, never selection
or adaptation. This checks the historical screen, not a full TEST stream.
"""
import argparse
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..history_tangent_20261023.learning import flat, logit
from ..paired_feedback_20261022.model import initial_encoder
from ..simple_online import _durable_json
from ..simple_train import sha
from ..task_geometry_20261023.run import READ, load_model
from .learning import ARMS, query_key, sample_queries
from .run import prepare, probe


def choose_independently(memory, query, arm, excluded):
    """Plain sorted-loop implementation, separate from the experimental selector."""
    keys, labels, shots = (np.asarray(memory[k]) for k in ('keys','labels','shots'))
    chosen=[]; distances=[]
    for y in (0,1):
        candidates=[i for i in range(len(keys)) if labels[i]==y and shots[i]!=excluded]
        center=np.asarray(query,dtype=float)
        if arm=='centroid':center=np.mean([keys[i] for i in candidates],axis=0)
        center=center/np.linalg.norm(center)
        ranked=sorted((float(1-np.dot(keys[i],center)),i) for i in candidates)
        seen=set()
        for d,i in ranked:
            if int(shots[i]) not in seen:
                chosen.append(i);distances.append(d);seen.add(int(shots[i]))
            if len(seen)==2:break
        assert len(seen)==2
    return chosen,distances


def recount(rows):
    result={}
    for r in rows:
        assert r['label'] in (0,1)
        for phase in ('before','after'):
            z=float(r['logit_'+phase])
            loss=float(np.logaddexp(0.,z)-r['label']*z)
            assert np.isfinite(loss) and abs(loss-r[phase])<2e-14
    for arm in ARMS:
        per_shot=[];groups={}
        for group in ('safe','disruptive_timely','disruptive_early'):
            part=[r for r in rows if r['arm']==arm and r['group']==group]
            shots=[]
            for sid in sorted({r['shot'] for r in part}):
                rs=[r for r in part if r['shot']==sid]
                item=dict(shot=sid,group=group,n=len(rs),
                    before=float(np.mean([r['before'] for r in rs])),
                    after=float(np.mean([r['after'] for r in rs])))
                shots.append(item);per_shot.append(item)
            if shots:groups[group]={k:float(np.mean([r[k] for r in shots])) for k in ('before','after')}
        primary=[groups[g] for g in ('safe','disruptive_timely')]
        disruptive=[r for r in per_shot if r['group']=='disruptive_timely']
        improved=sum(r['after']<r['before'] for r in disruptive)
        result[arm]=dict(groups=groups,per_shot=per_shot,
            balanced_before=float(np.mean([g['before'] for g in primary])),
            balanced_after=float(np.mean([g['after'] for g in primary])),
            improved_disruptive_shots=improved,
            passes_screen=bool(len(disruptive)==4 and improved>=3
                and all(g['after']<=g['before'] for g in primary)
                and sum(g['after'] for g in primary)<sum(g['before'] for g in primary)))
    c=result['nearest']
    gate=bool(c['passes_screen'] and all(c['balanced_after']<result[a]['balanced_after'] for a in ('initial','centroid')))
    return result,gate


def contribution_summary(item_gradients, query_gradient):
    g,q=np.asarray(item_gradients,dtype=float),np.asarray(query_gradient,dtype=float)
    if g.ndim!=2 or g.shape[0]!=4 or q.shape!=(g.shape[1],) or not np.isfinite(g).all() or not np.isfinite(q).all():
        raise ValueError('four finite example gradients and one matching query gradient required')
    contributions=(g@q)/4
    dot=float(contributions.sum())
    return dict(weighted_dots=contributions.tolist(),total_dot=dot,
                direction_harmful=dot<0)


def decompose(model, signal, memory, manifest, record):
    fast=initial_encoder(model)
    x,m=[torch.as_tensor(v[None],device='cuda') for v in simple_data.window(signal,record['query'],128)]
    z=logit(model,x,m,fast).reshape(())
    risk=F.binary_cross_entropy_with_logits(z.double(),z.new_tensor(float(record['label'])).double())
    q=flat(torch.autograd.grad(risk,tuple(fast.values()))).detach().double()
    selected=record['update']['selected_indices']
    targets=torch.as_tensor(memory['labels'][selected],device='cuda',dtype=torch.float32)
    z=logit(model,memory['x'][selected],memory['mask'][selected],fast)
    losses=F.binary_cross_entropy_with_logits(z,targets,reduction='none')
    gradients=[]
    for j in range(4):
        gradients.append(flat(torch.autograd.grad(losses[j],tuple(fast.values()),retain_graph=True)).detach())
    direct=flat(torch.autograd.grad(losses.mean(),tuple(fast.values()))).detach()
    # Per-example and batched reductions accumulate in different float32 order.
    residual=float((torch.stack(gradients).double().mean(0)-direct.double()).norm())
    relative=residual/max(float(direct.double().norm()),1e-12)
    assert relative<2e-5
    result=contribution_summary(torch.stack(gradients).cpu().numpy(),q.cpu().numpy())
    assert np.isclose(float(direct.double()@q),record['gradient_dot'],rtol=2e-6,atol=1e-9)
    assert np.isclose(result['total_dot'],record['gradient_dot'],rtol=2e-5,atol=1e-9)
    components=[dict(manifest[i],memory_index=i,probability=float(z[j].detach().sigmoid()),
        historical_bce=float(losses[j].detach()),weighted_dot=result['weighted_dots'][j]) for j,i in enumerate(selected)]
    return dict(shot=record['shot'],query=record['query'],group=record['group'],
        query_label=record['label'],query_bce_before=record['before'],query_bce_after=record['after'],
        cosine=record['cosine'],components=components,gradient_sum_relative_error=relative,**result)


def audit(root):
    if (root/'verification.json').exists():raise FileExistsError(root/'verification.json')
    p,done,rows=READ(root/'protocol.json'),READ(root/'completed.json'),READ(root/'probes.json')
    assert done['status']=='COMPLETED' and not done['test_accessed']
    for name,digest in done['artifact_sha256'].items():assert sha(root/name)==digest,name
    for name,digest in p['code_sha256'].items():assert sha(io23.ROOT/name)==digest,name
    ref=Path(p['reference'])
    for name,digest in p['reference_hashes'].items():assert sha(ref/name)==digest,name
    for filename,digest in ((p['parent'],p['parent_sha256']),(p['source_checkpoint'],p['source_sha256']),
                            (p['plan_path'],p['plan_sha256'])):
        assert sha(Path(filename))==digest,filename
    ids=p['selected_ids'];assert len(set(ids))==20
    assert not set(ids)&set(READ(ref/'protocol.json')['contract']['shot_ids'])
    access=READ(root/'label_access.json')
    assert access==[dict(device='EAST',shot=sid) for sid in ids]
    summary,gate=recount(rows)
    assert summary==done['summary'] and gate==done['recommend_full_stream']
    manifest=READ(root/'memory_manifest.json');sampling=READ(root/'sampling.json')
    assert len(manifest)==len(set((r['shot'],r['end']) for r in manifest))==done['historical_windows']==640
    assert len(sampling)==len(set((r['shot'],r['query']) for r in sampling))==done['queries']==640
    assert len(rows)==1920 and sum(r['update']['gradient_steps'] for r in rows)==done['neural_steps']
    cached=torch.load(root/'memory.pt',map_location='cpu',weights_only=True)
    torch.set_num_threads(2);torch.manual_seed(131)
    model,threshold=load_model();origin=state_cpu(model);assert threshold==p['threshold']
    readshot,outcome=simple_data.read_shot,io23.read_outcome
    observed=[]
    def allowed_signal(device,sid,**kwargs):
        assert device=='EAST' and sid in ids,'non-history signal forbidden during audit'
        return readshot(device,sid,**kwargs)
    def allowed_outcome(device,sid):
        assert device=='EAST' and sid in ids,'non-history outcome forbidden during audit'
        observed.append(sid);return outcome(device,sid)
    replay=[];sampled=[];mechanism=[];length_difference=[]
    historical_fit=Counter();negative_groups=Counter()
    lookup={(r['shot'],r['query'],r['arm']):r for r in rows};assert len(lookup)==len(rows)
    with patch.object(simple_data,'read_shot',side_effect=allowed_signal),patch.object(io23,'read_outcome',side_effect=allowed_outcome):
        memory,built,records,signals=prepare(model,ids,READ(ref/'input_map.json'))
        assert manifest==built
        for k,v in memory.items():torch.testing.assert_close(torch.as_tensor(v).cpu(),cached[k],rtol=0,atol=0)
        assert sum(records[s]['is_disrupt'] for s in ids)==4
        for sid in ids:
            shot=records[sid];signal=signals[sid]
            for index,r in enumerate(sample_queries(shot)):
                query=r['query'];y=int(shot['y'][query]);entry=dict(r,shot=sid,index=index)
                sampled.append(dict(entry,label=y,lead_ms=round(float((shot['down']-shot['times'][query])*1000),9)
                                    if shot['is_disrupt'] else None))
                assert r['count']>=256 and query==r['count']+24
                again=probe(model,signal,entry,y,memory)
                current_key=query_key(model,signal,r['count'],device='cuda')
                first_key=query_key(model,signal,128,device='cuda')
                actual_lengths=[]
                for a in again:
                    old=lookup[(sid,query,a['arm'])];assert a==old,(sid,query,a['arm'])
                    selected,dist=choose_independently(memory,first_key if a['arm']=='initial' else current_key,a['arm'],sid)
                    u=old['update'];assert selected==u['selected_indices']
                    np.testing.assert_allclose(dist,u['distances'],rtol=0,atol=2e-14)
                    assert u['last_observation']==r['count']-1 and u['first_affected_prediction']==r['count']
                    assert u['selected_labels']==[0,0,1,1] and sid not in u['selected_shots']
                    assert len(set(u['selected_shots'][:2]))==len(set(u['selected_shots'][2:]))==2
                    assert 0<u['parameter_delta_norm']<=.00101 and u['gradient_steps']==1
                    actual_lengths.append(u['parameter_delta_norm'])
                    historical_fit[a['arm']]+=int(u['loss_after']<u['loss_before'])
                length_difference.append(max(actual_lengths)-min(actual_lengths))
                if r['group']=='disruptive_timely':
                    result=decompose(model,signal,memory,manifest,again[0]);mechanism.append(result)
                    negative_groups.update(c['group'] for c in result['components'] if c['label']==0)
            replay.append(dict(shot=sid,queries=32,arms=3,exact=True))
            print(dict(replayed=sid,rows=len(replay)*96),flush=True)
    assert sampled==sampling and observed==ids and max(length_difference)<2e-7
    assert_unchanged(model,origin)
    mechanism_path=root/'gradient_decomposition.json';assert not mechanism_path.exists()
    _durable_json(mechanism_path,dict(scope='post-hoc historical timely-positive diagnosis only',
        sign='negative dot means the descent update increases future-query BCE to first order',
        nearest_negative_groups=dict(negative_groups),records=mechanism))
    harmful={group:sum(sum(c['weighted_dot'] for c in r['components'] if c['group']==group)<0 for r in mechanism)
             for group in ('safe','disruptive_early','disruptive_timely')}
    report=dict(status='VERIFIED',completed_sha256=sha(root/'completed.json'),audit_sha256=sha(Path(__file__)),
        audit_tests_sha256=sha(io23.ROOT/'experiments/risk_aligned_online/tests/test_history_retrieval_audit.py'),
        decomposition_sha256=sha(mechanism_path),summary=summary,full_stream_gate=gate,
        independent_bce_recounts=len(rows),independent_retrieval_recounts=len(rows),cuda_replay=replay,
        exact_replayed_rows=len(rows),memory_rebuilt=True,history_loss_decreased=dict(historical_fit),
        max_matched_step_length_difference=max(length_difference),
        nearest_positive_queries=len(mechanism),nearest_negative_groups=dict(negative_groups),
        harmful_dot_by_memory_group=harmful,
        harmful_combined_directions=sum(r['direction_harmful'] for r in mechanism),
        max_gradient_decomposition_relative_error=max(r['gradient_sum_relative_error'] for r in mechanism),
        parent_unchanged=True,no_test_access=True,
        caveat='20 preparation shots, parent trained all20; not independent validation; no full-stream metric')
    _durable_json(root/'verification.json',report)
    print(dict(status='VERIFIED',replayed_rows=len(rows),gate=gate,negative_groups=dict(negative_groups),
               harmful_dot_by_memory_group=harmful),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);audit(parser.parse_args().root)
