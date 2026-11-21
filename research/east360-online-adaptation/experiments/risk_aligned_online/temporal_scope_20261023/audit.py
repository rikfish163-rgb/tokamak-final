"""Independent counts, one-fold full inference replay, post-hoc ranking ceiling."""
import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..diverse_support_20261023.run import make_parent,READ
from ..increment_forecast_20261023.learning import predict
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_train import sha
from ..simple_online import _durable_json
from .run import ARMS,REFERENCE,REF_SHA


def recount(root,p,arms):
    expected=READ(root/'per_shot.json');counts={a:dict(tp=0,fp=0,n_disrupt=0,n_safe=0) for a in arms}
    maxima={a:[] for a in arms};checks=0
    for plan in p['folds']:
        for sid in plan['held']:
            s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
            timely=s['valid']&(np.round((s['down']-s['times'])*1000,9)>30.) if s['is_disrupt'] else s['valid']
            with np.load(root/f'fold{plan["fold"]}'/'shots'/f'{sid}.npz') as z:
                for a in arms:
                    score=z[a];indices=np.flatnonzero(s['valid']&(score>=p['threshold']));alarm=bool(len(indices))
                    lead=float(np.round((s['down']-s['times'][indices[0]])*1000,9)) if s['is_disrupt'] and alarm else None
                    row=dict(shot=sid,is_disrupt=s['is_disrupt'],alarm=alarm,lead_ms=lead,timely=lead is not None and lead>30.)
                    assert row==next(r for r in expected[a]['predictions'] if r['shot']==sid)
                    c=counts[a];c['tp']+=int(row['timely']);c['fp']+=int(alarm and not s['is_disrupt'])
                    c['n_disrupt']+=int(s['is_disrupt']);c['n_safe']+=int(not s['is_disrupt']);checks+=1
                    maxima[a].append(dict(shot=sid,is_disrupt=s['is_disrupt'],eligible_max=float(score[timely].max()) if timely.any() else -1.))
    for a in arms:
        for k,v in counts[a].items():assert v==expected[a][k]
    ceilings={}
    for a,rows in maxima.items():
        x=np.array([r['eligible_max'] for r in rows]);d=np.array([r['is_disrupt'] for r in rows]);result={}
        for cap in (.1,.15,.2):
            choices=[]
            for t in np.r_[np.nextafter(1.,2.),np.unique(x[x>=0])]:
                tp=int((x[d]>=t).sum());fp=int((x[~d]>=t).sum())
                if fp/(~d).sum()<=cap:choices.append((tp,-fp,float(t)))
            tp,negfp,t=max(choices);result[str(cap)]=dict(tp=tp,fp=-negfp,threshold=t,tpr=tp/int(d.sum()),fpr=-negfp/int((~d).sum()))
        ceilings[a]=result
    return dict(shot_arm_checks=checks,counts=counts,posthoc_maxima=maxima,oracle_single_threshold=ceilings,
        warning='USES ALL EVALUATION LABELS. Not deployed calibration or independently validated performance.')


def main(root):
    if (root/'verification.json').exists():raise FileExistsError(root/'verification.json')
    start=time.monotonic();p=READ(root/'protocol.json');done=READ(root/'completed.json')
    for name,h in done['artifact_sha256'].items():assert sha(root/name)==h,name
    for name,h in p['code_sha256'].items():assert sha(io23.ROOT/name)==h,name
    for name,h in p['input_sha256'].items():assert sha(name)==h,name
    assert sha(REFERENCE/'completed.json')==REF_SHA and sha(DEFAULT_PARENT)==PARENT_SHA256
    torch.set_num_threads(2);torch.manual_seed(131)
    result=recount(root,p,ARMS)
    old=recount(REFERENCE,READ(REFERENCE/'protocol.json'),READ(REFERENCE/'protocol.json')['arms'])
    _durable_json(root/'threshold_diagnostic.json',dict(current=result,reference=old))
    replays=[];access=READ(root/'training_label_access.json')
    for plan in p['folds']:
        f=plan['fold'];folder=root/f'fold{f}';ref=REFERENCE/f'fold{f}'
        selected=READ(folder/'selection.json')['selected']
        assert selected==sorted(READ(ref/'selection.json')['selected']['diverse'])
        used=[r['shot'] for r in access if r['fold']==f]
        assert used==selected and len(used)==20 and not set(selected)&set(plan['held'])
        for name in ('risk_manifest.jsonl','forecast_manifest.jsonl'):assert (folder/name).read_bytes()==(ref/'diverse'/name).read_bytes()
        model,_=make_parent();origin=state_cpu(model)
        model.load_state_dict(torch.load(folder/'prepared.pt',map_location='cuda',weights_only=True)['model']);model.set_stage('frozen')
        extra=[k for k,_ in model.named_parameters() if k.startswith('base.encoder.layers.1.')]
        assert_unchanged(model,origin,(*model.encoder_names(),*extra,'base.forecast_head.weight','base.forecast_head.bias',
            'reconstructor.weight','reconstructor.bias','event_head.weight','event_head.bias'))
        assert all(k not in model.encoder_names() for k in extra)
        if f==0:
            rows=result['posthoc_maxima']['source']
            chosen=[next(r['shot'] for r in rows if r['shot'] in plan['held'] and r['is_disrupt']==flag) for flag in (False,True)]
            frozen=state_cpu(model)
            def deny(*args,**kwargs):raise AssertionError('outcome denied during replay')
            with patch.object(io23,'read_outcome',side_effect=deny):
                for sid in chosen:
                    s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                    scores,updates,_=predict(model,{k:s[k] for k in ('values','available')})
                    with np.load(folder/'shots'/f'{sid}.npz') as z:
                        for a,v in scores.items():np.testing.assert_array_equal(v,z['both_'+a])
                    assert updates==READ(folder/'shots'/f'{sid}_updates.json')
                    replays.append(dict(fold=f,shot=sid))
            assert_unchanged(model,frozen)
    verified=dict(status='VERIFIED_COUNTS_AND_SELECTED_REPLAY',completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)),diagnostic_sha256=sha(root/'threshold_diagnostic.json'),
        new_shot_arm_checks=result['shot_arm_checks'],reference_shot_arm_checks=old['shot_arm_checks'],
        replays=replays,training_manifests_exactly_match=True,training_reproduced=False,
        source_unchanged=sha(DEFAULT_PARENT)==PARENT_SHA256,seconds=time.monotonic()-start)
    _durable_json(root/'verification.json',verified);print(verified,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
