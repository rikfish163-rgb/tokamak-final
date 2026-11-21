"""Post-run checks, independent first-crossing recount and sampled CUDA replay.

The experiment loaded a class-labelled split manifest before prediction. Only
HDF outcome/DownTime access was gated; do not claim all TEST metadata was blind.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.audit import paired_flips
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import block_episode,episode_pool
from .run import ARMS
from .stream import predict_five


def recount(sid,times,score,disrupt,down,threshold):
    assert score.shape==times.shape and np.isfinite(score).all()
    assert ((score>=0)&(score<=1)).all()
    valid=np.ones(len(times),bool) if not disrupt else (down-times)*1000>=-1e-6
    crossing=np.flatnonzero(valid & (score>=threshold))
    lead=round(float((down-times[crossing[0]])*1000),9) if disrupt and len(crossing) else None
    return dict(shot=sid,is_disrupt=bool(disrupt),alarm=bool(len(crossing)),
                lead_ms=lead,timely=bool(lead is not None and lead>30))


def audit(root,replay):
    read=lambda path:json.loads(path.read_text())
    p=read(root/'protocol.json');completed=read(root/'completed.json')
    assert completed['status']=='COMPLETED'
    assert sha(Path(p['plan_path']))==p['plan_sha256']
    for file,digest in p['code_sha256'].items():assert sha(io23.ROOT/file)==digest,file
    assert sha(DEFAULT_PARENT)==p['source_sha256']==PARENT_SHA256
    assert sha(Path(p['warm_checkpoint']))==p['warm_sha256']
    reference=Path(p['reference']);assert sha(reference/'protocol.json')==p['reference_protocol_sha256']
    assert read(root/'input_map.json')==read(reference/'input_map.json')
    fit=set(p['fit_ids']);held=set(p['holdout_ids']);order=p['contract']['shot_ids']
    assert len(fit)==15 and len(held)==5 and not fit&held
    assert fit|held==set(p['selected_ids']) and not (fit|held)&set(order)
    assert set(read(root/'pre_eval_label_access.json'))==fit|held
    mapping=read(root/'input_map.json');shots={}
    for sid in sorted(fit|held):
        s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        s=simple_data.attach_outcome(s,'EAST',{})
        shots[sid]=dict(s,**transform({k:s[k] for k in ('values','available')},mapping))
    eligible=episode_pool([shots[sid] for sid in p['fit_ids']],p['fit_ids'])
    assert eligible==read(root/'eligibility.json')
    byid={r['shot']:r for r in eligible}
    manifest=[json.loads(line) for line in (root/'training_manifest.jsonl').read_text().splitlines()]
    training=read(root/'training.json');assert len(manifest)==len(training)==p['steps']==64
    for step,(m,h) in enumerate(zip(manifest,training)):
        assert m['step']==h['step']==step and [e['is_disrupt'] for e in m['episodes']]==[False,True]
        for e in m['episodes']:
            assert e['shot'] in fit and dict(count=e['count'],ends=e['ends']) in byid[e['shot']]['blocks']
        for arm in ARMS[1:]:
            for e,r in zip(m['episodes'],h[arm]['records']):
                assert all(r[k]==v for k,v in e.items())
                assert r['update_counts']==list(range(256,e['count']+1,128))
                assert r['differentiated_counts']==r['update_counts'][-2:]
    prep=read(root/'preparation_receipts.json');states={}
    warm_state=torch.load(Path(p['warm_checkpoint']),map_location='cpu',weights_only=True)['model']
    changed={}
    for arm in ARMS:
        ck=root/f'{arm}.pt';assert sha(ck)==prep[arm]['sha256']
        state=torch.load(ck,map_location='cpu',weights_only=True)['model'];states[arm]=state
        changed[arm]=[k for k in state if not torch.equal(state[k],warm_state[k])]
        assert set(changed[arm])<=set(p['persistent_optimized_parameters'])
        assert bool(changed[arm])==(arm!='warm')
    full=read(root/'per_shot.json');names=list(completed['metrics'])
    counts={k:dict(tp=0,fp=0,n_disrupt=0,n_safe=0,leads=[],early=0) for k in names}
    seals=read(root/'prediction_receipts.json');assert [r['shot'] for r in seals]==order
    steps={a:0 for a in ARMS};nonzero={a:0 for a in ARMS};early_fp={k:[] for k in names}
    for index,r in enumerate(seals):
        sid=r['shot'];path=root/'shots'/f'{sid}.npz';up=root/'shots'/f'{sid}_updates.json'
        assert sha(path)==r['prediction_sha256'] and sha(up)==r['updates_sha256']
        d,down=io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as trace, np.load(reference/'shots'/f'{sid}.npz',allow_pickle=False) as old:
            np.testing.assert_array_equal(trace['static'],old['input_only_static'])
            np.testing.assert_array_equal(trace['warm_intra'],old['input_only_intra'])
            for name in names:
                row=recount(sid,trace['times'],trace[name],d,down,p['threshold'])
                assert row==full[name]['predictions'][index],(sid,name)
                c=counts[name];c['n_disrupt' if d else 'n_safe']+=1
                if row['timely']:
                    c['tp']+=1;c['leads'].append(row['lead_ms']);c['early']+=row['lead_ms']>3000
                if not d and row['alarm']:c['fp']+=1
                if not d and (trace[name][:256]>=p['threshold']).any():early_fp[name].append(sid)
                np.testing.assert_array_equal(trace[name][:256],trace['static'][:256])
        updates=read(up)
        for arm in ARMS:
            for u in updates[arm]:
                assert u['first_affected_prediction']==u['last_observation']+1
                assert max(e+25 for e in u['forecast_origins'])<=u['last_observation']
                steps[arm]+=u['gradient_steps'];nonzero[arm]+=u['parameter_delta_norm']>0
    for name,c in counts.items():
        report=completed['metrics'][name]
        for key in ('tp','fp','n_disrupt','n_safe'):assert c[key]==report[key],(name,key)
        assert c['early']==report['very_early_over3000']
        med=float(np.median(c['leads'])) if c['leads'] else None
        assert med==report['median_lead_ms']
        assert c['tp']/c['n_disrupt']==report['tpr'] and c['fp']/c['n_safe']==report['fpr']
    replays=[];gradient_checks=[]
    if replay:
        torch.set_num_threads(2);torch.manual_seed(131)
        base,_=load_parent(DEFAULT_PARENT);parent=PairedFeedback(base,np.ones(23)).cuda();parent.set_stage('frozen')
        models={k:copy.deepcopy(parent) for k in ARMS}
        for k,m in models.items():m.load_state_dict(states[k]);m.set_stage('frozen')
        for sid in (order[0],order[len(order)//2],order[-1]):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            pred,updates=predict_five(models,transform({k:s[k] for k in ('values','available')},mapping))
            with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as saved:
                for name in names:np.testing.assert_array_equal(pred[name],saved[name])
            assert updates==read(root/'shots'/f'{sid}_updates.json')
            replays.append(dict(shot=sid,arms=len(names),max_abs_error=0.,update_receipts_exact=True))
        # Verify risk-only outer gradients on actual historical signals, not just
        # a synthetic fixture or the combined objective's forecast component.
        for arm in ('warm','maximum_meta'):
            for e in manifest[0]['episodes']:
                m=copy.deepcopy(models[arm]);m.base.forecast_head.requires_grad_(True)
                s=shots[e['shot']];signal={k:s[k] for k in ('values','available')}
                out=block_episode(m,signal,e['count'],e['ends'],e['is_disrupt'],
                    boundary=math.log(p['threshold']/(1-p['threshold'])),objective='maximum',device='cuda',create_graph=True)
                params=tuple(m.base.forecast_head.parameters())
                risk=torch.cat([g.flatten() for g in torch.autograd.grad(out['risk_after'],params,retain_graph=True)])
                forecast=torch.cat([g.flatten() for g in torch.autograd.grad(out['forecast'],params,retain_graph=True)])
                scaled=risk/(p['inner_learning_rate']*len(out['differentiated_counts']))
                norm=float(risk.norm());assert math.isfinite(norm) and norm>0
                out['risk_after'].backward()
                assert all(x.grad is None for k,x in m.named_parameters() if k not in p['persistent_optimized_parameters'])
                gradient_checks.append(dict(arm=arm,shot=e['shot'],is_disrupt=e['is_disrupt'],
                    risk_only_head_grad_norm=norm,forecast_grad_norm=float(forecast.norm()),
                    scaled_risk_grad_norm=float(scaled.norm()),
                    gradient_cosine=float(torch.nn.functional.cosine_similarity(scaled,forecast,dim=0))))
    result=dict(status='VERIFIED',all_shot_traces_recounted=len(names)*len(order),
        parent_sha256=sha(DEFAULT_PARENT),protocol_sha256=sha(root/'protocol.json'),completed_sha256=sha(root/'completed.json'),
        optimized_tensors=changed,steps=steps,nonzero_parameter_updates=nonzero,
        paired_vs_static={n:paired_flips(full['static']['predictions'],full[n]['predictions']) for n in names[1:]},
        paired_vs_forecast={n:paired_flips(full['forecast_intra']['predictions'],full[n]['predictions']) for n in ('point_meta_intra','maximum_meta_intra')},
        false_alarms_before_first_update=early_fp,sampled_cuda_replays=replays,risk_only_gradient_checks=gradient_checks,
        full_independent_reinference=False,test_manifest_classes_read=True,
        label_access_claim='No TEST HDF outcome/DownTime used before all seals; split manifest had class lists',
        forecast_control_claim='Label-selected episodes, same budget, through-update forecast objective; not wholly label-free preparation')
    _durable_json(root/'verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();audit(args.root,args.replay)
