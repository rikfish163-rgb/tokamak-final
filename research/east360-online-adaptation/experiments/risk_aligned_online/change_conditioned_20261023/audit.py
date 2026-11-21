"""Recount, verify training scope, replay and inspect risk-task gradients."""
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
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256,load_source_train
from ..simple_online import _durable_json
from ..simple_train import sha
from ..target_feedback_meta_20261023.audit import recount
from ..warmstart_alignment_20261022.alignment import transform
from .model import ChangeConditioned,patches,fixed_scale
from .run import READ,VARIANTS,NAMES,load_model,batches,predict


def audit(root,replay):
    p,done,full=[READ(root/f) for f in ('protocol.json','completed.json','per_shot.json')]
    assert done['status']=='COMPLETED' and done['original_parameters_unchanged']
    assert sha(root/'protocol.json')==done['protocol_sha256']
    assert sha(DEFAULT_PARENT)==p['source_sha256']==PARENT_SHA256
    assert sha(Path(p['plan_path']))==p['plan_sha256']
    for f,h in p['code_sha256'].items():assert sha(io23.ROOT/f)==h,f
    ref=Path(p['reference'])
    for f,h in p['reference_hashes'].items():assert sha(ref/f)==h,f
    assert sha(ref/'conditional_moments.pt')==p['parent_sha256']
    assert p['contract']==load_test_contract('EAST');order=p['contract']['shot_ids']
    ids=p['selected_ids'];assert len(ids)==len(set(ids))==20 and not set(ids)&set(order)
    assert p['source_train_ids']==io23.source_roles()['train']
    access=[dict(device='J-TEXT',shot=s) for s in p['source_train_ids']]+[dict(device='EAST',shot=s) for s in ids]
    assert READ(root/'pre_eval_label_access.json')==access
    manifest=[json.loads(line) for line in (root/'training_manifest.jsonl').read_text().splitlines()]
    training=READ(root/'training.json');scale=READ(root/'scale_fit.json')
    assert len(manifest)==len(training)==p['steps']==512 and manifest[:32]==scale['manifest']
    for i,(r,log) in enumerate(zip(manifest,training)):
        assert r['step']==log['step']==i
        for domain,allowed in (('source',p['source_train_ids']),('target',ids)):
            assert len(r[domain])==32 and [e['label'] for e in r[domain]]==[0]*16+[1]*16
            assert all(e['shot'] in allowed and e['end']>=0 for e in r[domain])
        assert all(np.isfinite(v) for name in ('level','change') for v in log[name].values())
    parent,threshold=load_model(device='cpu');assert threshold==p['threshold']
    models={};base=state_cpu(parent.base)
    for name,digest in done['checkpoint_sha256'].items():
        path=root/f'{name}.pt';assert sha(path)==digest
        payload=torch.load(path,map_location='cpu',weights_only=True);assert payload['key_kind']==name
        model=ChangeConditioned(copy.deepcopy(parent.base),torch.tensor(scale['scale']),name)
        model.load_state_dict(payload['model']);assert_unchanged(model.base,base)
        torch.testing.assert_close(model.change_scale,torch.tensor(scale['scale']),rtol=0,atol=0)
        params={k:v for k,v in model.named_parameters() if not k.startswith('base.')}
        assert set(params)==set(p['slow_parameters']) and sum(v.numel() for v in params.values())==2055
        assert all(torch.isfinite(v).all() for v in params.values())
        models[name]=model.eval().requires_grad_(False)
    seals=READ(root/'prediction_receipts.json');assert [s['shot'] for s in seals]==order
    oldseals={r['shot']:r for r in READ(ref/'prediction_receipts.json')}
    total={a:dict(tp=0,fp=0,n_disrupt=0,n_safe=0,leads=[],early=0) for a in NAMES}
    stats={a:dict(points=0,fit_observed=0,fast_changed=0,fast_delta_sum=0.,held_error_sum=0.,held_targets=0,correction_norm_sum=0.) for a in VARIANTS}
    for i,s in enumerate(seals):
        sid=s['shot'];path=root/'shots'/f'{sid}.npz';sp=root/'shots'/f'{sid}_stats.json';old=ref/'shots'/f'{sid}.npz'
        assert sha(path)==s['prediction_sha256'] and sha(sp)==s['stats_sha256']
        assert sha(old)==s['reference_sha256']==oldseals[sid]['prediction_sha256']
        d,down=io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as z,np.load(old,allow_pickle=False) as original:
            assert set(z.files)=={'times',*NAMES};np.testing.assert_array_equal(z['times'],original['times'])
            np.testing.assert_array_equal(z['parent'],original['conditional_moments_static']);n=len(z['times'])
            for a in VARIANTS:np.testing.assert_array_equal(z[a][:23],z['parent'][:23])
            for a in NAMES:
                row=recount(sid,z['times'],z[a],d,down,threshold)
                assert row==full[a]['predictions'][i],(sid,a)
                v=total[a];v['n_disrupt' if d else 'n_safe']+=1
                v['tp']+=row['timely'];v['fp']+=not d and row['alarm']
                if row['timely']:v['leads'].append(row['lead_ms']);v['early']+=row['lead_ms']>3000
        ss=READ(sp);assert set(ss)==set(VARIANTS)
        for a,v in ss.items():
            assert v['points']==n and 0<=v['fast_changed']<=v['fit_observed']<=n
            if VARIANTS[a][1]=='prior':assert v['fast_changed']==v['fast_delta_sum']==0
            for k in stats[a]:stats[a][k]+=v[k]
    for a,v in total.items():
        m=done['metrics'][a]
        for k in ('tp','fp','n_disrupt','n_safe'):assert m[k]==v[k]
        assert v['n_disrupt']==v['n_safe']==180
        assert m['median_lead_ms']==float(np.median(v['leads'])) and m['very_early_over3000']==v['early']
        assert m['tpr']==v['tp']/180 and m['fpr']==v['fp']/180 and m['J']==m['tpr']-m['fpr']
    for v in stats.values():v['mean_held_smoothl1']=v['held_error_sum']/v['held_targets']
    contrasts={}
    for name in models:
        for mode in ('fit','shuffled','mean'):contrasts[name+'_'+mode+'_vs_prior']=(name+'_prior',name+'_'+mode)
        contrasts[name+'_correct_vs_shuffled']=(name+'_shuffled',name+'_fit')
        contrasts[name+'_correct_vs_mean']=(name+'_mean',name+'_fit')
    contrasts['change_fit_vs_level_fit']=('level_fit','change_fit')
    replays=[];gradients={};key_probe={};scale_recomputed=False
    if replay:
        torch.set_num_threads(2);torch.manual_seed(131)
        models={k:v.cuda() for k,v in models.items()};frozen={k:state_cpu(v) for k,v in models.items()}
        mapping=READ(ref/'input_map.json')
        for sid in (order[0],order[len(order)//2],order[-1]):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            pred,ss=predict(models,transform({k:s[k] for k in ('values','available')},mapping))
            with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z:
                for a in VARIANTS:np.testing.assert_array_equal(pred[a],z[a])
            assert ss==READ(root/'shots'/f'{sid}_stats.json')
            replays.append(dict(shot=sid,curves=8,max_abs_error=0.,stats_exact=True))
        source,_=load_source_train();target=[]
        for sid in ids:
            s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
            s.update(transform({k:s[k] for k in ('values','available')},mapping));target.append(s)
        pieces=[];probe={k:[] for k in models};last=None
        with torch.no_grad():
            for x,m,y,r in batches(source,target,parent.cfg,32):
                assert r==manifest[r['step']];pieces.append(patches(models['level'].base,x,m));last=(x,m,y)
                for name,model in models.items():
                    _,keys,_,_=model.components(x,m);features=keys[:,:13,:8]
                    probe[name].extend(features.std(1,correction=0).mean(-1).cpu().tolist())
            sc,ct=fixed_scale(pieces)
            np.testing.assert_array_equal(sc.cpu().numpy(),np.array(scale['scale'],dtype=np.float32))
            np.testing.assert_array_equal(ct.cpu().numpy(),np.array(scale['observations']))
            scale_recomputed=True
        key_probe={k:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v))) for k,v in probe.items()}
        del source,target,pieces
        for name,model in models.items():
            for k,v in model.named_parameters():v.requires_grad_(k in p['slow_parameters'])
            x,m,y=last;r=model(x,m,mode='fit');loss=F.binary_cross_entropy_with_logits(r['logit'],y)
            params=[(k,v) for k,v in model.named_parameters() if v.requires_grad]
            gs=torch.autograd.grad(loss,[v for _,v in params])
            gradients[name]={k:float(g.norm()) for (k,_),g in zip(params,gs)}
            assert all(np.isfinite(v) and v>0 for v in gradients[name].values())
            assert_unchanged(model,frozen[name])
    result=dict(status='VERIFIED',audit_sha256=sha(Path(__file__)),completed_sha256=sha(root/'completed.json'),
        protocol_sha256=sha(root/'protocol.json'),all_traces_independently_recounted=len(NAMES)*360,
        sampled_cuda_replays=replays,full_independent_reinference=False,base_weights_exact=True,
        scale_recomputed=scale_recomputed,risk_only_gradient_norms=gradients,training_key_within_window_std=key_probe,
        statistics=stats,paired_contrasts={k:paired_flips(full[a]['predictions'],full[b]['predictions']) for k,(a,b) in contrasts.items()},
        independent_confirmation=False,scope='single seed development test; no deployed Transformer backprop or physical-law claim')
    _durable_json(root/'verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--replay',action='store_true')
    args=p.parse_args();audit(args.root,args.replay)
