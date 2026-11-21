"""Independent recount, intervention coverage, paired differences and CUDA replay."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.audit import paired_flips
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..target_feedback_meta_20261023.audit import recount
from ..temporal_dependence_20261023.audit import lead_median,transition_groups
from ..temporal_dependence_20261023.run import load_model
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import predict_many
from .control import predict_reversed


def audit(root):
    read=lambda p:json.loads(p.read_text())
    p=read(root/'protocol.json');done=read(root/'completed.json');full=read(root/'per_shot.json')
    assert done['status']=='COMPLETED'
    assert sha(DEFAULT_PARENT)==PARENT_SHA256==p['source_sha256']
    assert sha(Path(p['plan_path']))==p['plan_sha256']
    for n,h in p['code_sha256'].items():assert sha(io23.ROOT/n)==h,n
    ref=Path(p['temporal_reference'])
    for n,h in p['temporal_hashes'].items():assert sha(ref/n)==h,n
    ids=p['contract']['shot_ids'];seals=read(root/'prediction_receipts.json')
    assert [r['shot'] for r in seals]==ids
    groups=transition_groups(full['raw_original']['predictions'],full['mapped_original']['predictions'])
    counts={a:dict(tp=0,fp=0,d=0,s=0,leads=[],early=0) for a in p['arms']}
    coverage={v:dict(total=0,eligible=0,changed=0,original_alarms=0,eligible_original_alarms=0,
                    max_abs_probability_change=0.,shot_mean_abs_change=[]) for v in ('raw','mapped')}
    group_counts={g:{a:dict(tp=0,fp=0,alarm=0) for a in p['arms']} for g in groups}
    label={sid:g for g,vs in groups.items() for sid in vs}
    for i,seal in enumerate(seals):
        sid=seal['shot'];path=root/'shots'/f'{sid}.npz';assert sha(path)==seal['prediction_sha256']
        d,down=io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as z,np.load(ref/'shots'/f'{sid}.npz',allow_pickle=False) as old:
            for a in p['arms']:
                r=recount(sid,z['times'],z[a],d,down,p['threshold']);assert r==full[a]['predictions'][i]
                c=counts[a];c['d' if d else 's']+=1;c['tp']+=r['timely'];c['fp']+=not d and r['alarm']
                if r['timely']:c['leads'].append(r['lead_ms']);c['early']+=r['lead_ms']>3000
                for k,v in dict(tp=r['timely'],fp=not d and r['alarm'],alarm=r['alarm']).items():
                    group_counts[label[sid]][a][k]+=v
            for view,c in coverage.items():
                original=z[view+'_original'];altered=z[view+'_reversed'];e=z[view+'_eligible'];changed=z[view+'_changed']
                np.testing.assert_array_equal(original,old[view+'_original'])
                assert not e[:127].any() and not (changed & ~e).any()
                np.testing.assert_array_equal(original[~e],altered[~e])
                c['total']+=len(e);c['eligible']+=int(e.sum());c['changed']+=int(changed.sum())
                valid=np.ones(len(e),bool) if not d else (down-z['times'])*1000>=-1e-6
                crossings=np.flatnonzero(valid & (original>=p['threshold']))
                if len(crossings):c['original_alarms']+=1;c['eligible_original_alarms']+=int(e[crossings[0]])
                delta=abs(original.astype(float)-altered)
                c['max_abs_probability_change']=max(c['max_abs_probability_change'],float(delta.max()))
                c['shot_mean_abs_change'].append(float(delta.mean()))
    for a,c in counts.items():
        m=done['metrics'][a];assert c['d']==c['s']==180 and c['tp']==m['tp'] and c['fp']==m['fp']
        assert c['early']==m['very_early_over3000'] and lead_median(c['leads'])==m['median_lead_ms']
        assert m['tpr']==c['tp']/180 and m['fpr']==c['fp']/180 and m['J']==m['tpr']-m['fpr']
    torch.set_num_threads(2);torch.manual_seed(131);model,threshold=load_model(Path(p['preparation_reference']))
    assert threshold==p['threshold'];frozen=state_cpu(model);mapping=read(root/'input_map.json');replays=[]
    for sid in (ids[0],ids[len(ids)//2],ids[-1]):
        s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        raw={k:s[k] for k in ('values','available')};signals=dict(raw=raw,mapped=transform(raw,mapping))
        originals=predict_many(model,signals,device='cuda')
        with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z:
            for view,signal in signals.items():
                r=predict_reversed(model,signal)
                np.testing.assert_array_equal(r['probability'],z[view+'_reversed'])
                for k in ('eligible','changed'):np.testing.assert_array_equal(r[k],z[view+'_'+k])
                np.testing.assert_array_equal(originals[view],z[view+'_original'])
        replays.append(dict(shot=sid,arms=4,all_exact=True))
    assert_unchanged(model,frozen)
    for c in coverage.values():
        c['equal_shot_mean_abs_change']=float(np.mean(c.pop('shot_mean_abs_change')))
        c['eligible_fraction']=c['eligible']/c['total']
    report=dict(status='VERIFIED',protocol_sha256=sha(root/'protocol.json'),completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)),recounted_shot_arms=len(ids)*len(p['arms']),coverage=coverage,
        source_unchanged=True,neural_updates=0,sampled_cuda_replays=replays,groups=groups,group_counts=group_counts,
        paired={v:paired_flips(full[v+'_original']['predictions'],full[v+'_reversed']['predictions']) for v in coverage},
        full_independent_reinference=False,physical_causality_identified=False)
    _durable_json(root/'verification.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);audit(p.parse_args().root)
