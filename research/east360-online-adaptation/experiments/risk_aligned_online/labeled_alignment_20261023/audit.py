"""Independent shot-level recount, artifact checks and sampled CUDA replay."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import predict_pairs
from ..deepcoral_prefix_20261022.audit import paired_flips
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_train import sha
from ..simple_online import _durable_json
from ..warmstart_alignment_20261022.alignment import transform,fit_statistics,make_map
from ..warmstart_alignment_20261022.run import predict_many
from .learning import choose_budget
from .run import ARMS


def audit(root, replay):
    p=json.loads((root/'protocol.json').read_text()); completed=json.loads((root/'completed.json').read_text())
    assert completed['status']=='COMPLETED'
    plan=io23.ROOT/'docs/research/source_event_rebuild_20261022/少量有标签EAST对齐_事前约定_20261023.md'
    assert sha(plan)==p['plan_sha256']
    full=json.loads((root/'per_shot.json').read_text())
    selection=json.loads((root/'selection.json').read_text())
    order=p['contract']['shot_ids']; ids=p['selected_ids']
    assert not set(ids)&set(order) and len(ids)==p['budget']
    assert selection['selected']==choose_budget(selection['audit']['accepted'],p['budget'],test_ids=order)
    assert ids==[r['shot'] for r in selection['selected']]
    for file,digest in p['code_sha256'].items(): assert sha(io23.ROOT/file)==digest,file
    assert sha(DEFAULT_PARENT)==p['source_sha256']==PARENT_SHA256
    accesses=json.loads((root/'pre_eval_label_access.json').read_text())
    assert {r['shot'] for r in accesses if r['device']=='EAST'}==set(ids)
    assert {r['shot'] for r in accesses if r['device']=='J-TEXT'}==set(io23.source_roles()['train'])
    manifest_rows=0
    with (root/'risk_training_manifest.jsonl').open() as f:
        for line in f:
            r=json.loads(line);manifest_rows+=1
            for domain,allowed in (('source',set(io23.source_roles()['train'])),('target',set(ids))):
                assert all(x['shot'] in allowed for x in r[domain])
                assert [x['label'] for x in r[domain]]==[0]*16+[1]*16
    assert manifest_rows==p['steps']
    mapping=json.loads((root/'input_map.json').read_text())
    records=[]
    for sid in ids:
        s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        records.append({k:s[k] for k in ('shot','device','values','available')})
    target_stats=fit_statistics(records,device='EAST',allowed_ids=ids)
    assert make_map(mapping['source_statistics'],target_stats,'density_gain')==mapping
    seals=json.loads((root/'prediction_receipts.json').read_text());assert [r['shot'] for r in seals]==order
    names=list(completed['metrics']);counts={k:dict(tp=0,fp=0,n_disrupt=0,n_safe=0,leads=[],early=0) for k in names}
    steps={k:0 for k in ARMS};delta_positive={k:0 for k in ARMS}
    for index,receipt in enumerate(seals):
        sid=receipt['shot'];path=root/'shots'/f'{sid}.npz';up=root/'shots'/f'{sid}_updates.json'
        assert sha(path)==receipt['prediction_sha256'];assert sha(up)==receipt['updates_sha256']
        d,down=io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as trace:
            t=trace['times'];valid=np.ones(len(t),bool) if not d else ((down-t)*1000>=-1e-6)
            for name in names:
                score=trace[name];assert score.shape==t.shape and np.isfinite(score).all()
                assert ((score>=0)&(score<=1)).all()
                crossing=np.flatnonzero(valid & (score>=p['threshold']))
                lead=round(float((down-t[crossing[0]])*1000),9) if d and len(crossing) else None
                # Recount each classification independently, not just aggregate
                # totals: equal totals can hide different rescued/harmed shots.
                assert full[name]['predictions'][index]==dict(shot=sid,is_disrupt=bool(d),
                    alarm=bool(len(crossing)),lead_ms=lead,timely=bool(lead is not None and lead>30))
                c=counts[name];c['n_disrupt' if d else 'n_safe']+=1
                if d and lead is not None and lead>30:
                    c['tp']+=1;c['leads'].append(lead);c['early']+=lead>3000
                if not d and len(crossing):c['fp']+=1
            for arm in ARMS:
                np.testing.assert_array_equal(trace[f'{arm}_static'][:256],trace[f'{arm}_intra'][:256])
        updates=json.loads(up.read_text())
        for arm in ARMS:
            for u in updates[arm]:
                assert u['first_affected_prediction']==u['last_observation']+1
                assert max(e+25 for e in u['forecast_origins'])<=u['last_observation']
                steps[arm]+=u['gradient_steps'];delta_positive[arm]+=u['parameter_delta_norm']>0
    for name,c in counts.items():
        reported=completed['metrics'][name]
        for key in ('tp','fp','n_disrupt','n_safe'):assert c[key]==reported[key],(name,key)
        assert c['early']==reported['very_early_over3000']
        assert np.median(c['leads'])==reported['median_lead_ms'] if c['leads'] else reported['median_lead_ms'] is None
        assert c['tp']/c['n_disrupt']==reported['tpr'] and c['fp']/c['n_safe']==reported['fpr']
    replay_receipts=[]
    if replay:
        torch.set_num_threads(2);torch.manual_seed(131)
        base,_=load_parent(DEFAULT_PARENT);parent=PairedFeedback(base,np.ones(23)).cuda();parent.set_stage('frozen')
        models={};prep=json.loads((root/'preparation_receipts.json').read_text())
        for arm in ARMS:
            ck=root/f'{arm}.pt';assert sha(ck)==prep[arm]['checkpoint_sha256']
            state=torch.load(ck,map_location='cpu',weights_only=True)['model']
            models[arm]=copy.deepcopy(parent);models[arm].load_state_dict(state);models[arm].set_stage('frozen')
            for name,v in models[arm].base.state_dict().items():
                allowed={x.removeprefix('base.') for x in models[arm].encoder_names()}|{'forecast_head.weight','forecast_head.bias'}
                if name not in allowed or (arm=='input_only' and not name.startswith('forecast_head.')):
                    assert torch.equal(v.cpu(),parent.base.state_dict()[name].cpu()),(arm,name)
        for sid in (order[0],order[len(order)//2],order[-1]):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal={k:s[k] for k in ('values','available')}
            new,updates=predict_pairs(models,transform(signal,mapping))
            new['raw_source']=predict_many(parent,dict(raw=signal),device='cuda')['raw']
            with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as saved:
                for name in names:np.testing.assert_array_equal(new[name],saved[name])
            assert updates==json.loads((root/'shots'/f'{sid}_updates.json').read_text())
            replay_receipts.append(dict(shot=sid,arms=len(names),prediction_max_abs=0.,update_receipts_exact=True))
    result=dict(status='VERIFIED',all_shot_traces_recounted=len(names)*len(order),
        metadata_training_roles=88,signal_fitting_shots=len(ids),outcome_access_gate='TRAIN only before evaluation',
        prefix_mapping_recomputed=True,gradient_steps=steps,positive_parameter_steps=delta_positive,
        sampled_cuda_replays=replay_receipts,full_independent_reinference=False,
        protocol_sha256=sha(root/'protocol.json'),completed_sha256=sha(root/'completed.json'),
        parent_sha256=sha(DEFAULT_PARENT),counts={k:{a:b for a,b in v.items() if a!='leads'} for k,v in counts.items()},
        paired_intra_changes={a:paired_flips(full[f'{a}_static']['predictions'],full[f'{a}_intra']['predictions']) for a in ARMS},
        paired_static_changes={a:paired_flips(full['input_only_static']['predictions'],full[f'{a}_static']['predictions']) for a in ARMS[1:]})
    _durable_json(root/'verification.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--replay',action='store_true')
    a=p.parse_args();audit(a.root,a.replay)
