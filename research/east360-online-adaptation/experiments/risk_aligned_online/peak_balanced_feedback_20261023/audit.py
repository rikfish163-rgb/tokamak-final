"""Independent scoring, fixed-control replay and factorial attribution checks."""
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
from ..target_feedback_meta_20261023.audit import recount
from ..target_feedback_meta_20261023.learning import block_episode
from ..target_feedback_meta_20261023.stream import predict_five
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import predict_many
from .learning import peak_manifest,balanced_head_gradient
from .run import ARMS,TRAIN_ARMS,HEAD


def audit(root,replay):
    read=lambda path:json.loads(path.read_text())
    p=read(root/'protocol.json');completed=read(root/'completed.json');reference=Path(p['reference'])
    assert completed['status']=='COMPLETED'
    assert sha(DEFAULT_PARENT)==p['source_sha256']==PARENT_SHA256
    assert sha(Path(p['plan_path']))==p['plan_sha256']
    for f,digest in p['code_sha256'].items():assert sha(io23.ROOT/f)==digest,f
    for path,key in (('protocol.json','reference_protocol_sha256'),('completed.json','reference_completed_sha256'),
                     ('warm.pt','reference_warm_sha256'),('maximum_meta.pt','reference_maximum_sha256')):
        assert sha(reference/path)==p[key]
    assert read(root/'input_map.json')==read(reference/'input_map.json')
    fit=set(p['fit_ids']);held=set(p['holdout_ids']);order=p['contract']['shot_ids']
    assert len(fit)==15 and len(held)==5 and not fit&held and fit|held==set(p['selected_ids'])
    assert not (fit|held)&set(order) and set(read(root/'pre_eval_label_access.json'))==fit|held
    m=read(root/'manifests.json');pool=read(root/'eligibility.json')
    assert m['random']==[json.loads(line) for line in (reference/'training_manifest.jsonl').read_text().splitlines()]
    scores={}
    for row in pool:
        assert row['shot'] in fit
        if row['is_disrupt']:continue
        with np.load(root/'calibration'/f"{row['shot']}.npz",allow_pickle=False) as trace:scores[row['shot']]=trace['static'].copy()
    rebuilt,peaks=peak_manifest(m['random'],pool,scores)
    assert rebuilt==m['peak'] and peaks==m['peaks']
    for row in peaks:
        assert any(e['shot']==row['shot'] and e['count']==row['count'] and e.get('selection')=='frozen_safe_peak'
                   for step in m['peak'] for e in step['episodes'])
    history=read(root/'training.json');assert len(history)==64
    limits={a:[] for a in ('random_balanced','peak_balanced')}
    for i,h in enumerate(history):
        assert h['step']==i
        for arm in TRAIN_ARMS:
            expected=m['peak' if arm.startswith('peak') else 'random'][i]['episodes']
            assert len(h[arm]['records'])==len(expected)==2
            for e,r in zip(expected,h[arm]['records']):
                assert all(r[k]==v for k,v in e.items())
                assert r['update_counts']==list(range(256,e['count']+1,128))
                assert r['differentiated_counts']==r['update_counts'][-2:]
                if arm.endswith('balanced'):
                    info=r['gradient_balance']
                    assert 0<=info['risk_scale']<=1
                    assert info['capped_risk_norm']<=info['forecast_norm']*(1+1e-6)+1e-12
                    limits[arm].append(info['risk_scale'])
    prep=read(root/'preparation_receipts.json');states={}
    warm=torch.load(reference/'warm.pt',map_location='cpu',weights_only=True)['model']
    old=torch.load(reference/'maximum_meta.pt',map_location='cpu',weights_only=True)['model']
    changed={}
    for arm in ARMS:
        path=root/f'{arm}.pt';assert sha(path)==prep[arm]['sha256']
        s=torch.load(path,map_location='cpu',weights_only=True)['model'];states[arm]=s
        changed[arm]=[k for k in s if not torch.equal(s[k],warm[k])]
        assert set(changed[arm])<=set(HEAD)
        assert bool(changed[arm])==(arm!='warm')
        if arm=='random_original':assert all(torch.equal(s[k],old[k]) for k in s)
    full=read(root/'per_shot.json');names=list(full);counts={k:dict(tp=0,fp=0,d=0,s=0,leads=[],early=0) for k in names}
    receipts=read(root/'prediction_receipts.json');assert [r['shot'] for r in receipts]==order
    update_count={a:0 for a in ARMS};nonzero={a:0 for a in ARMS}
    for index,r in enumerate(receipts):
        sid=r['shot'];path=root/'shots'/f'{sid}.npz';up=root/'shots'/f'{sid}_updates.json'
        assert sha(path)==r['prediction_sha256'] and sha(up)==r['updates_sha256']
        d,down=io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as trace,np.load(reference/'shots'/f'{sid}.npz',allow_pickle=False) as previous:
            for name,key in (('static','static'),('warm_intra','warm_intra'),('random_original_intra','maximum_meta_intra')):
                np.testing.assert_array_equal(trace[name],previous[key])
            for name in names:
                np.testing.assert_array_equal(trace[name][:256],trace['static'][:256])
                row=recount(sid,trace['times'],trace[name],d,down,p['threshold'])
                assert row==full[name]['predictions'][index]
                c=counts[name];c['d' if d else 's']+=1
                c['tp']+=row['timely'];c['fp']+=not d and row['alarm']
                if row['timely']:c['leads'].append(row['lead_ms']);c['early']+=row['lead_ms']>3000
        for arm,updates in read(up).items():
            for u in updates:
                assert u['first_affected_prediction']==u['last_observation']+1
                assert max(e+25 for e in u['forecast_origins'])<=u['last_observation']
                update_count[arm]+=u['gradient_steps'];nonzero[arm]+=u['parameter_delta_norm']>0
    for name,c in counts.items():
        reported=completed['metrics'][name]
        assert c['d']==c['s']==180
        assert c['tp']==reported['tp'] and c['fp']==reported['fp'] and c['early']==reported['very_early_over3000']
        assert c['tp']/180==reported['tpr'] and c['fp']/180==reported['fpr']
        assert (float(np.median(c['leads'])) if c['leads'] else None)==reported['median_lead_ms']
    cuda=[];gradient_checks=[];calibration_replay=[]
    if replay:
        torch.set_num_threads(2);torch.manual_seed(131)
        base,_=load_parent(DEFAULT_PARENT);parent=PairedFeedback(base,np.ones(23)).cuda();parent.set_stage('frozen')
        models={a:copy.deepcopy(parent) for a in ARMS}
        for a,model in models.items():model.load_state_dict(states[a]);model.set_stage('frozen')
        mapping=read(root/'input_map.json')
        for sid in (order[0],order[len(order)//2],order[-1]):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            pred,updates=predict_five(models,transform({k:s[k] for k in ('values','available')},mapping))
            with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
                for name in names:np.testing.assert_array_equal(pred[name],trace[name])
            assert updates==read(root/'shots'/f'{sid}_updates.json')
            cuda.append(dict(shot=sid,arms=len(names),max_abs_error=0.,update_receipts_exact=True))
        for sid in scores:
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=transform({k:s[k] for k in ('values','available')},mapping)
            pred=predict_many(models['warm'],dict(static=signal),device='cuda')['static']
            np.testing.assert_array_equal(pred,scores[sid]);calibration_replay.append(sid)
        for arm in ('random_balanced','peak_balanced'):
            for record in history[0][arm]['records']:
                model=copy.deepcopy(models['warm']);model.base.forecast_head.requires_grad_(True)
                s=apply_input_contract(simple_data.read_shot('EAST',record['shot'],rate_contract='nominal'))
                signal=transform({k:s[k] for k in ('values','available')},mapping)
                ep=block_episode(model,signal,record['count'],record['ends'],record['is_disrupt'],
                    boundary=math.log(p['threshold']/(1-p['threshold'])),objective='maximum',device='cuda',create_graph=True)
                _,info=balanced_head_gradient(ep,tuple(model.base.forecast_head.parameters()))
                assert info==record['gradient_balance'],(arm,record['shot'])
                gradient_checks.append(dict(arm=arm,shot=record['shot'],exact=True,**info))
    metric=completed['metrics'];j=lambda a:metric[a+'_intra']['J']
    factor=dict(sampling_with_original=j('peak_original')-j('random_original'),
                sampling_with_balanced=j('peak_balanced')-j('random_balanced'),
                balance_with_random=j('random_balanced')-j('random_original'),
                balance_with_peak=j('peak_balanced')-j('peak_original'),
                interaction=j('peak_balanced')-j('random_balanced')-j('peak_original')+j('random_original'))
    out=dict(status='VERIFIED',protocol_sha256=sha(root/'protocol.json'),completed_sha256=sha(root/'completed.json'),
        audit_code_sha256=sha(Path(__file__)),all_shot_traces_recounted=len(order)*len(names),changed_tensors=changed,
        all_safe_training_peaks_covered=len(peaks),per_episode_gradient_caps_valid=True,
        median_risk_gradient_scale={k:float(np.median(v)) for k,v in limits.items()},
        gradient_steps=update_count,nonzero_updates=nonzero,sampled_cuda_replays=cuda,
        full_independent_reinference=False,calibration_peak_trace_replays=calibration_replay,
        real_first_step_gradient_replays=gradient_checks,descriptive_factorial_J_effects=factor,
        paired_vs_static={name:paired_flips(full['static']['predictions'],full[name]['predictions']) for name in names if name!='static'},
        paired_vs_random_original={name:paired_flips(full['random_original_intra']['predictions'],full[name]['predictions']) for name in names if name not in ('static','warm_intra','random_original_intra')})
    _durable_json(root/'verification.json',out);print(json.dumps(out),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();audit(args.root,args.replay)
