"""Independent first-crossing recount, frozen-state checks, and CUDA replay."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.audit import paired_flips
from ..deepcoral_prefix_20261022.run import assert_unchanged,predict_pairs,state_cpu
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..target_feedback_meta_20261023.audit import recount
from ..temporal_dependence_20261023.run import load_model
from ..warmstart_alignment_20261022.alignment import transform


def audit(root,replay):
    read=lambda p:json.loads(p.read_text())
    p=read(root/'protocol.json');done=read(root/'completed.json');full=read(root/'per_shot.json')
    ref=Path(p['reference']);mapping=read(root/'input_map.json');oldmap=read(ref/'input_map.json')
    assert done['status']=='COMPLETED' and sha(DEFAULT_PARENT)==p['source_sha256']==PARENT_SHA256
    assert sha(Path(p['plan_path']))==p['plan_sha256']
    for file,digest in p['code_sha256'].items():assert sha(io23.ROOT/file)==digest,file
    for file,digest in p['reference_hashes'].items():assert sha(ref/file)==digest,file
    assert mapping['calibration_ids']==p['selected_ids'] and len(p['selected_ids'])==20
    assert not set(p['selected_ids'])&set(p['contract']['shot_ids'])
    assert mapping['offset']==oldmap['offset'] and mapping['fitted_input_parameters']==1
    assert [g for c,g in enumerate(mapping['gain']) if c!=12]==[g for c,g in enumerate(oldmap['gain']) if c!=12]
    assert mapping['gain'][12]==done['gain_after']>0 and oldmap['gain'][12]==done['gain_before']
    access=read(root/'pre_eval_label_access.json')
    assert {r['shot'] for r in access if r['device']=='EAST'}==set(p['selected_ids'])
    assert all(r['shot'] in io23.source_roles()['train'] for r in access if r['device']=='J-TEXT')
    assert {r['device'] for r in access}=={'EAST','J-TEXT'}
    train=read(root/'gain_training.json')
    manifest=[json.loads(line) for line in (root/'gain_manifest.jsonl').read_text().splitlines()]
    oldmanifest=[json.loads(line) for line in (ref/'risk_training_manifest.jsonl').read_text().splitlines()]
    assert len(train)==len(manifest)==p['steps']==256
    for i,(t,m) in enumerate(zip(train,manifest)):
        assert t['step']==m['step']==i and m['target']==oldmanifest[i]['target']
        assert all(r['shot'] in p['selected_ids'] for r in m['target'])
        assert np.isfinite([t['bce'],t['gradient_norm'],t['gain']]).all() and t['gain']>0
    assert train[-1]['gain']==mapping['gain'][12]
    assert (root/'forecast_manifest.jsonl').read_bytes()==(ref/'forecast_manifest.jsonl').read_bytes()
    checkpoint=root/'learned.pt';prep=read(root/'preparation_receipts.json')
    assert sha(checkpoint)==prep['checkpoint_sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)['model']
    before=torch.load(ref/'input_only.pt',map_location='cpu',weights_only=True)['model']
    changed=[k for k in state if not torch.equal(state[k],before[k])]
    assert set(changed)=={'base.forecast_head.weight','base.forecast_head.bias'}
    seals=read(root/'prediction_receipts.json');ids=p['contract']['shot_ids']
    assert [r['shot'] for r in seals]==ids and len(ids)==len(set(ids))==360
    refseals={r['shot']:r for r in read(ref/'prediction_receipts.json')}
    counts={a:dict(tp=0,fp=0,d=0,s=0,leads=[],early=0) for a in p['arms']}
    update_summary=dict(steps=0,nonzero=0,same_sample_loss_decreased=0,max_delta=0.)
    for index,seal in enumerate(seals):
        sid=seal['shot'];path=root/'shots'/f'{sid}.npz';up=root/'shots'/f'{sid}_updates.json'
        assert sha(path)==seal['prediction_sha256'] and sha(up)==seal['updates_sha256']
        oldpath=ref/'shots'/f'{sid}.npz'
        assert sha(oldpath)==seal['reference_prediction_sha256']==refseals[sid]['prediction_sha256']
        disrupt,down=io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as z,np.load(oldpath,allow_pickle=False) as old:
            np.testing.assert_array_equal(z['times'],old['times']);n=len(z['times'])
            np.testing.assert_array_equal(z['original_static'],old['input_only_static'])
            np.testing.assert_array_equal(z['original_intra'],old['input_only_intra'])
            for name in ('original','learned'):
                np.testing.assert_array_equal(z[name+'_static'][:256],z[name+'_intra'][:256])
            for a in p['arms']:
                r=recount(sid,z['times'],z[a],disrupt,down,p['threshold'])
                assert r==full[a]['predictions'][index],(sid,a)
                c=counts[a];c['d' if disrupt else 's']+=1
                c['tp']+=r['timely'];c['fp']+=not disrupt and r['alarm']
                if r['timely']:c['leads'].append(r['lead_ms']);c['early']+=r['lead_ms']>3000
        updates=read(up);assert set(updates)=={'learned'}
        assert [u['first_affected_prediction'] for u in updates['learned']]==list(range(256,n,128))
        for u in updates['learned']:
            count=u['first_affected_prediction']
            assert u['last_observation']==count-1 and max(e+25 for e in u['forecast_origins'])<=count-1
            assert u['forecast_origins']==[count-1-25*(j+1) for j in range(4)]
            if u['gradient_steps']:
                assert u['gradient_steps']==1 and u['parameter_delta_norm']<=.00101
                update_summary['steps']+=1
                update_summary['nonzero']+=u['parameter_delta_norm']>0
                update_summary['same_sample_loss_decreased']+=u['loss_after']<u['loss_before']
                update_summary['max_delta']=max(update_summary['max_delta'],u['parameter_delta_norm'])
    for a,c in counts.items():
        m=done['metrics'][a]
        assert c['d']==c['s']==180 and c['tp']==m['tp'] and c['fp']==m['fp']
        assert c['early']==m['very_early_over3000'] and float(np.median(c['leads']))==m['median_lead_ms']
        assert c['tp']/180==m['tpr'] and c['fp']/180==m['fpr'] and m['J']==m['tpr']-m['fpr']
    replayed=[]
    if replay:
        torch.set_num_threads(2);torch.manual_seed(131);model,threshold=load_model(ref)
        model.load_state_dict(state);model.set_stage('frozen');frozen=state_cpu(model)
        assert threshold==p['threshold']
        for sid in (ids[0],ids[len(ids)//2],ids[-1]):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            pred,updates=predict_pairs(dict(learned=model),transform({k:s[k] for k in ('values','available')},mapping))
            with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z:
                for a,v in pred.items():np.testing.assert_array_equal(v,z[a])
            assert updates==read(root/'shots'/f'{sid}_updates.json')
            replayed.append(dict(shot=sid,arms=2,max_abs_difference=0.,update_receipts_exact=True))
        assert_unchanged(model,frozen)
    result=dict(status='VERIFIED',protocol_sha256=sha(root/'protocol.json'),completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)),all_traces_independently_recounted=1440,
        network_changed_only_forecast_head=changed,source_archive_unchanged=True,update_summary=update_summary,
        sampled_cuda_replays=replayed,full_independent_reinference=False,
        comparisons={name:paired_flips(full[a]['predictions'],full[b]['predictions']) for name,a,b in (
            ('static_mapping_change','original_static','learned_static'),
            ('original_intra','original_static','original_intra'),
            ('learned_intra','learned_static','learned_intra'))},
        independent_confirmation=False,
        label_access_claim='No TEST HDF outcome or DownTime used before sealing; class split manifest already known')
    _durable_json(root/'verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--replay',action='store_true')
    a=p.parse_args();audit(a.root,a.replay)
