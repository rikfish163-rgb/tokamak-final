"""Independent first-crossing recount and checks on preparation/online updates."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.audit import paired_flips
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged,predict_pairs
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..target_feedback_meta_20261023.audit import recount
from ..temporal_dependence_20261023.run import load_model
from ..warmstart_alignment_20261022.alignment import transform
from .learning import candidate_ends,aggregation_for
from .run import ARMS,NAMES


def audit(root,replay):
    read=lambda p:json.loads(p.read_text())
    p=read(root/'protocol.json');done=read(root/'completed.json');full=read(root/'per_shot.json')
    assert done['status']=='COMPLETED' and sha(DEFAULT_PARENT)==p['source_sha256']==PARENT_SHA256
    assert sha(Path(p['plan_path']))==p['plan_sha256']
    for file,digest in p['code_sha256'].items():assert sha(io23.ROOT/file)==digest,file
    ref=Path(p['reference'])
    for file,digest in p['reference_hashes'].items():assert sha(ref/file)==digest,file
    assert read(root/'input_map.json')==read(ref/'input_map.json')
    assert (root/'forecast_manifest.jsonl').read_bytes()==(ref/'forecast_manifest.jsonl').read_bytes()
    ids=p['contract']['shot_ids'];selected=p['selected_ids']
    assert len(ids)==len(set(ids))==360 and len(selected)==len(set(selected))==20 and not set(ids)&set(selected)
    access=read(root/'pre_eval_label_access.json')
    assert {a['device'] for a in access}=={'J-TEXT','EAST'}
    assert {a['shot'] for a in access if a['device']=='EAST'}==set(selected)
    assert {a['shot'] for a in access if a['device']=='J-TEXT'}==set(io23.source_roles()['train'])
    candidates=read(root/'candidate_manifest.json');labels={}
    for sid in selected:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        assert candidates[str(sid)]==candidate_ends(s).tolist();labels[sid]=s['is_disrupt']
    assert sum(labels.values())==4
    history=read(root/'risk_training.json')
    manifest=[json.loads(line) for line in (root/'risk_training_manifest.jsonl').read_text().splitlines()]
    assert len(history)==len(manifest)==p['steps']==64
    rng=np.random.default_rng(2131);groups={d:[s for s in selected if labels[s]==d] for d in (False,True)}
    for i,(h,m) in enumerate(zip(history,manifest)):
        assert h['step']==m['step']==i
        assert m['target_ids']==[groups[d][int(rng.integers(len(groups[d])))] for d in (False,True)]
        assert len(m['source'])==32 and [r['label'] for r in m['source']]==[0]*16+[1]*16
        assert all(r['shot'] in io23.source_roles()['train'] for r in m['source'])
        for a in ARMS:
            assert np.isfinite(h[a]['gradient_norm']) and np.isfinite(h[a]['source_bce'])
            assert [b['shot'] for b in h[a]['bags']]==m['target_ids']
            for b in h[a]['bags']:
                assert b['is_disrupt']==labels[b['shot']]
                assert b['candidates']==len(candidates[str(b['shot'])]) and b['peak_end'] in candidates[str(b['shot'])]
                assert b['aggregation']==aggregation_for(a,b['is_disrupt']) and np.isfinite(b['loss'])
    warm=torch.load(ref/'input_only.pt',map_location='cpu',weights_only=True)['model']
    prep=read(root/'preparation_receipts.json');states={};changed={}
    for a in ARMS:
        path=root/f'{a}.pt';assert sha(path)==prep[a]['checkpoint_sha256']
        state=torch.load(path,map_location='cpu',weights_only=True)['model'];states[a]=state
        changed[a]=[k for k in state if not torch.equal(state[k],warm[k])]
        assert changed[a]==prep[a]['changed_parameters']
        assert set(changed[a])<=set(p['parameter_scope'])|{'base.forecast_head.weight','base.forecast_head.bias'}
        assert set(changed[a])&set(p['parameter_scope']) and prep[a]['encoder_delta_norm']>0
    seals=read(root/'prediction_receipts.json');assert [r['shot'] for r in seals]==ids
    refseals={r['shot']:r for r in read(ref/'prediction_receipts.json')}
    counts={a:dict(tp=0,fp=0,d=0,s=0,leads=[],early=0) for a in NAMES}
    update_summary={a:dict(steps=0,nonzero=0,loss_decreased=0,max_delta=0.) for a in ARMS}
    for index,seal in enumerate(seals):
        sid=seal['shot'];path=root/'shots'/f'{sid}.npz';up=root/'shots'/f'{sid}_updates.json';refpath=ref/'shots'/f'{sid}.npz'
        assert sha(path)==seal['prediction_sha256'] and sha(up)==seal['updates_sha256']
        assert sha(refpath)==seal['reference_sha256']==refseals[sid]['prediction_sha256']
        d,down=io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as z,np.load(refpath,allow_pickle=False) as old:
            n=len(z['times']);np.testing.assert_array_equal(z['times'],old['times'])
            np.testing.assert_array_equal(z['original_static'],old['input_only_static'])
            np.testing.assert_array_equal(z['original_intra'],old['input_only_intra'])
            for a in ('original',*ARMS):np.testing.assert_array_equal(z[a+'_static'][:256],z[a+'_intra'][:256])
            for a in NAMES:
                r=recount(sid,z['times'],z[a],d,down,p['threshold']);assert r==full[a]['predictions'][index],(sid,a)
                c=counts[a];c['d' if d else 's']+=1;c['tp']+=r['timely'];c['fp']+=not d and r['alarm']
                if r['timely']:c['leads'].append(r['lead_ms']);c['early']+=r['lead_ms']>3000
        updates=read(up);assert set(updates)==set(ARMS)
        for a in ARMS:
            assert [u['first_affected_prediction'] for u in updates[a]]==list(range(256,n,128))
            for u in updates[a]:
                count=u['first_affected_prediction'];assert u['last_observation']==count-1
                assert u['forecast_origins']==[count-1-25*(j+1) for j in range(4)]
                assert max(e+25 for e in u['forecast_origins'])<=count-1
                if u['gradient_steps']:
                    assert u['gradient_steps']==1 and u['parameter_delta_norm']<=.00101
                    stat=update_summary[a];stat['steps']+=1;stat['nonzero']+=u['parameter_delta_norm']>0
                    stat['loss_decreased']+=u['loss_after']<u['loss_before'];stat['max_delta']=max(stat['max_delta'],u['parameter_delta_norm'])
    for a,c in counts.items():
        m=done['metrics'][a];assert c['d']==c['s']==180 and c['tp']==m['tp'] and c['fp']==m['fp']
        assert c['early']==m['very_early_over3000'] and float(np.median(c['leads']))==m['median_lead_ms']
        assert c['tp']/180==m['tpr'] and c['fp']/180==m['fpr'] and m['J']==m['tpr']-m['fpr']
    replays=[]
    if replay:
        torch.set_num_threads(2);torch.manual_seed(131);models={}
        for a in ARMS:
            m,threshold=load_model(ref);assert threshold==p['threshold']
            m.load_state_dict(states[a]);m.set_stage('frozen');models[a]=m
        frozen={a:state_cpu(m) for a,m in models.items()};mapping=read(root/'input_map.json')
        for sid in (ids[0],ids[len(ids)//2],ids[-1]):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            pred,updates=predict_pairs(models,transform({k:s[k] for k in ('values','available')},mapping))
            with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z:
                for a,v in pred.items():np.testing.assert_array_equal(v,z[a])
            assert updates==read(root/'shots'/f'{sid}_updates.json')
            replays.append(dict(shot=sid,arms=len(pred),max_abs_difference=0.,update_receipts_exact=True))
        for a,m in models.items():assert_unchanged(m,frozen[a])
    pairs=[(a+'_intra',a+'_static',a+'_intra') for a in ('original',*ARMS)]
    pairs += [('mean_to_safe_peak','mean_static','safe_peak_static'),('safe_peak_to_peak','safe_peak_static','peak_static')]
    pairs += [('original_to_'+a,'original_static',a+'_static') for a in ARMS]
    result=dict(status='VERIFIED',protocol_sha256=sha(root/'protocol.json'),completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)),all_traces_independently_recounted=2880,changed_parameters=changed,
        source_archive_unchanged=True,update_summary=update_summary,sampled_cuda_replays=replays,
        full_independent_reinference=False,comparisons={n:paired_flips(full[a]['predictions'],full[b]['predictions']) for n,a,b in pairs},
        independent_confirmation=False)
    _durable_json(root/'verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--replay',action='store_true')
    a=p.parse_args();audit(a.root,a.replay)
