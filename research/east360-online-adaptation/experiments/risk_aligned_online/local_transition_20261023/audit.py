"""Independent shot recount, parameter-scope checks and sampled CUDA replay."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.audit import paired_flips
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..fixed_test_protocol import load_test_contract
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..target_feedback_meta_20261023.audit import recount
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.audit import counts
from .model import LocalTransition
from .run import READ,NAMES,VARIANTS,load_model,predict


def audit(root,replay):
    p,done,full=[READ(root/f) for f in ('protocol.json','completed.json','per_shot.json')]
    assert done['status']=='COMPLETED' and done['original_parameters_unchanged']
    assert sha(DEFAULT_PARENT)==p['source_sha256']==PARENT_SHA256
    ref=Path(p['reference'])
    assert sha(ref/'conditional_moments.pt')==p['parent_sha256']
    assert sha(Path(p['plan_path']))==p['plan_sha256']
    for path,digest in p['code_sha256'].items():assert sha(io23.ROOT/path)==digest,path
    for path,digest in p['reference_hashes'].items():assert sha(ref/path)==digest,path
    assert p['contract']==load_test_contract('EAST')
    order,ids=p['contract']['shot_ids'],p['selected_ids']
    assert len(ids)==len(set(ids))==20 and not set(ids)&set(order)
    assert p['source_train_ids']==io23.source_roles()['train']
    expected=[dict(device='J-TEXT',shot=sid) for sid in p['source_train_ids']]
    expected.extend(dict(device='EAST',shot=sid) for sid in ids)
    assert READ(root/'pre_eval_label_access.json')==expected
    manifest=[json.loads(line) for line in (root/'training_manifest.jsonl').read_text().splitlines()]
    training=READ(root/'training.json')
    assert len(manifest)==len(training)==p['steps']==512
    for step,(r,h) in enumerate(zip(manifest,training)):
        assert r['step']==h['step']==step
        for domain,allowed in (('source',p['source_train_ids']),('target',ids)):
            assert len(r[domain])==32 and [e['label'] for e in r[domain]]==[0]*16+[1]*16
            assert all(e['shot'] in allowed and e['end']>=0 for e in r[domain])
        assert all(np.isfinite(v) for name in ('static_trained','adaptive_trained') for v in h[name].values())
    parent,threshold=load_model(device='cpu');assert threshold==p['threshold']
    models={};base=state_cpu(parent.base);changed={}
    for name,digest in done['checkpoint_sha256'].items():
        path=root/f'{name}.pt';assert sha(path)==digest
        model=LocalTransition(copy.deepcopy(parent.base))
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'])
        assert_unchanged(model.base,base)
        new={k:v for k,v in model.state_dict().items() if not k.startswith('base.')}
        assert set(new)==set(p['slow_parameters']) and sum(v.numel() for v in new.values())==2055
        assert torch.isfinite(torch.cat([v.flatten() for v in new.values()])).all()
        assert model.readout.weight.abs().sum()>0
        changed[name]=dict(readout_norm=float(model.readout.weight.detach().norm()),prior_norm=float(model.prior.detach().norm()))
        models[name]=model.eval().requires_grad_(False)
    seals=READ(root/'prediction_receipts.json');assert [s['shot'] for s in seals]==order
    refseals={r['shot']:r for r in READ(ref/'prediction_receipts.json')}
    totals={a:dict(tp=0,fp=0,n_disrupt=0,n_safe=0,leads=[],early=0) for a in NAMES}
    stats={a:dict(points=0,fit_observed=0,fast_changed=0,fast_delta_sum=0.,held_error_sum=0.,held_targets=0,correction_norm_sum=0.) for a in VARIANTS}
    contrasts={'static_fit_vs_static_prior':('static_prior','static_fit'),
        'adaptive_fit_vs_adaptive_prior':('adaptive_prior','adaptive_fit'),
        'adaptive_fit_vs_static_prior':('static_prior','adaptive_fit'),
        'adaptive_shuffled_vs_adaptive_fit':('adaptive_fit','adaptive_shuffled')}
    difference={k:dict(points=0,absolute_sum=0.,max_absolute=0.) for k in contrasts}
    for index,s in enumerate(seals):
        sid=s['shot'];path=root/'shots'/f'{sid}.npz';sp=root/'shots'/f'{sid}_stats.json'
        old=ref/'shots'/f'{sid}.npz'
        assert sha(path)==s['prediction_sha256'] and sha(sp)==s['stats_sha256']
        assert sha(old)==s['reference_sha256']==refseals[sid]['prediction_sha256']
        d,down=io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as z,np.load(old,allow_pickle=False) as original:
            assert set(z.files)=={'times',*NAMES}
            np.testing.assert_array_equal(z['times'],original['times'])
            np.testing.assert_array_equal(z['parent'],original['conditional_moments_static'])
            n=len(z['times'])
            for arm in VARIANTS:np.testing.assert_array_equal(z[arm][:15],z['parent'][:15])
            for arm in NAMES:
                row=recount(sid,z['times'],z[arm],d,down,threshold)
                assert row==full[arm]['predictions'][index],(sid,arm)
                c=totals[arm];c['n_disrupt' if d else 'n_safe']+=1
                c['tp']+=row['timely'];c['fp']+=not d and row['alarm']
                if row['timely']:c['leads'].append(row['lead_ms']);c['early']+=row['lead_ms']>3000
            for name,(a,b) in contrasts.items():
                diff=np.abs(z[a].astype(float)-z[b]);v=difference[name]
                v['points']+=len(diff);v['absolute_sum']+=float(diff.sum());v['max_absolute']=max(v['max_absolute'],float(diff.max()))
        ss=READ(sp);assert set(ss)==set(VARIANTS)
        for arm,v in ss.items():
            assert v['points']==n and 0<=v['fast_changed']<=v['fit_observed']<=n
            assert v['held_targets']<=23*n and v['held_error_sum']>=0
            if VARIANTS[arm][1]=='prior':assert v['fast_changed']==v['fast_delta_sum']==0
            for key in stats[arm]:stats[arm][key]+=v[key]
    for arm,c in totals.items():
        m=done['metrics'][arm]
        for key in ('tp','fp','n_disrupt','n_safe'):assert m[key]==c[key]
        assert c['n_disrupt']==c['n_safe']==180
        assert c['early']==m['very_early_over3000'] and float(np.median(c['leads']))==m['median_lead_ms']
        assert m['tpr']==c['tp']/180 and m['fpr']==c['fp']/180 and m['J']==m['tpr']-m['fpr']
    replays=[];risk_gradients={}
    if replay:
        torch.set_num_threads(2);torch.manual_seed(131)
        models={k:m.cuda() for k,m in models.items()};frozen={k:state_cpu(m) for k,m in models.items()}
        mapping=READ(ref/'input_map.json')
        for sid in (order[0],order[len(order)//2],order[-1]):
            signal=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            out,ss=predict(models,transform({k:signal[k] for k in ('values','available')},mapping))
            with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z:
                for arm in VARIANTS:np.testing.assert_array_equal(out[arm],z[arm])
            assert ss==READ(root/'shots'/f'{sid}_stats.json')
            replays.append(dict(shot=sid,curves=len(VARIANTS),max_abs_difference=0.,stats_exact=True))
        for k,m in models.items():assert_unchanged(m,frozen[k])
        # An actual risk-only gradient must reach the learned local solver;
        # a forecasting-only gradient would not demonstrate task coupling.
        m=models['adaptive_trained']
        for name,param in m.named_parameters():param.requires_grad_(name in p['slow_parameters'])
        rows=[manifest[0]['target'][0],manifest[0]['target'][-1]];xs=[];ms=[]
        for r in rows:
            signal=apply_input_contract(simple_data.read_shot('EAST',r['shot'],rate_contract='nominal'))
            mapped=transform({k:signal[k] for k in ('values','available')},mapping)
            x,mask=simple_data.window(mapped,r['end'],128);xs.append(x);ms.append(mask)
        x=torch.tensor(np.stack(xs),device='cuda');mask=torch.tensor(np.stack(ms),device='cuda')
        r=m(x,mask,mode='fit');loss=F.binary_cross_entropy_with_logits(r['logit'],torch.tensor([v['label'] for v in rows],device='cuda',dtype=torch.float32))
        params=[(name,v) for name,v in m.named_parameters() if v.requires_grad]
        grads=torch.autograd.grad(loss,[v for _,v in params])
        risk_gradients={name:float(g.norm()) for (name,_),g in zip(params,grads)}
        assert all(np.isfinite(v) and v>0 for v in risk_gradients.values())
        for k,m in models.items():assert_unchanged(m,frozen[k])
    for v in difference.values():v['mean_absolute']=v['absolute_sum']/v['points']
    for v in stats.values():v['mean_held_smoothl1']=v['held_error_sum']/v['held_targets']
    result=dict(status='VERIFIED',completed_sha256=sha(root/'completed.json'),protocol_sha256=sha(root/'protocol.json'),
        audit_sha256=sha(Path(__file__)),all_traces_independently_recounted=360*len(NAMES),
        sampled_cuda_replays=replays,full_independent_reinference=False,base_weights_exact=True,
        added_parameter_norms=changed,risk_only_gradient_norms=risk_gradients,statistics=stats,
        probability_differences=difference,independent_confirmation=False,
        descriptive_wilson95={a:counts(full[a]['predictions']) for a in NAMES},
        uncertainty_caveat='single seed; all TEST data development-exposed; Wilson does not address temporal dependence or selection',
        paired_vs_parent={a:paired_flips(full['parent']['predictions'],full[a]['predictions']) for a in VARIANTS},
        paired_contrasts={k:paired_flips(full[a]['predictions'],full[b]['predictions']) for k,(a,b) in contrasts.items()})
    _durable_json(root/'verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--replay',action='store_true')
    args=p.parse_args();audit(args.root,args.replay)
