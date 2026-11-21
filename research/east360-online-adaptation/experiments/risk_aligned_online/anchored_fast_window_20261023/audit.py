"""Recount first alarms, inspect true fast writes, replay changed shots."""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from ..internal_fast_window_20261023.audit import recount,correct
from .run import (READ,sha,io23,PREVIOUS,make_parent,models_for_calibration,REFERENCE,
    read_signal,transform,state_cpu,assert_unchanged,DEFAULT_PARENT,PARENT_SHA256,_durable_json)
from .model import AnchoredAdapter
from .learning import ARMS,predict


def main(root):
    p=READ(root/'protocol.json'); done=READ(root/'completed.json')
    for f,h in done['artifact_sha256'].items(): assert sha(root/f)==h,f
    for f,h in p['code_sha256'].items(): assert sha(io23.ROOT/f)==h,f
    for f,h in p['input_sha256'].items(): assert sha(f)==h,f
    rows=READ(root/'outcomes.json'); seals=READ(root/'prediction_seals.json'); stored=READ(root/'per_shot.json')
    assert p['shot_ids']==[r['shot'] for r in rows]==[s['shot'] for s in seals]
    train=READ(root/'training.json'); old_train=READ(PREVIOUS/'training.json')
    assert len(train)==p['additional_preparation_steps']
    for a,b in zip(train,old_train): assert a['samples']==b['samples']
    counts={a:dict(tp=0,fp=0,very_early_over3000=0) for a in ARMS}; checks=0
    changes={a:dict(good=[],bad=[]) for a in ('fixed_on','normalized_on','normalized_reverse')}
    inner={a:dict(endpoints=0,updated=0,nonzero_delta=0,loss_decreased=0,loss_increased=0,
                  max_delta_norm=0.,max_residual_norm=0.) for a in ('fixed','normalized')}
    reverse_differences=[]
    for i,(r,s) in enumerate(zip(rows,seals)):
        sid=r['shot']; assert io23.read_outcome('EAST',sid)==(r['is_disrupt'],r['down'])
        f=root/'shots'/f'{sid}.npz'; assert sha(f)==s['prediction_sha256']==r['prediction_sha256']
        with np.load(f) as z,np.load(PREVIOUS/'shots'/f'{sid}.npz') as old:
            np.testing.assert_array_equal(z['parent'],old['parent']); np.testing.assert_array_equal(z['normalized_off'],z['parent'])
            for a in ARMS:
                rr=dict(shot=sid,**recount(z['times'],z[a],r['is_disrupt'],r['down'],p['threshold']))
                assert rr==stored[a]['predictions'][i]; checks+=1
                counts[a]['tp']+=int(rr['timely']); counts[a]['fp']+=int(rr['alarm'] and not rr['is_disrupt'])
                counts[a]['very_early_over3000']+=int(rr['timely'] and rr['lead_ms']>3000)
            before=stored['parent']['predictions'][i]
            for a in changes:
                after=stored[a]['predictions'][i]
                if correct(before)!=correct(after): changes[a]['good' if correct(after) else 'bad'].append(dict(shot=sid,before=before,after=after))
            reverse_differences.append(float(np.max(np.abs(z['normalized_on']-z['normalized_reverse']))))
        for name,record in inner.items():
            f=root/'shots'/f'{sid}_{name}_inner.npz'; assert sha(f)==s['inner_sha256'][name]
            with np.load(f) as t:
                np.testing.assert_array_equal(t['updated'],(t['observations']>=2)&(t['curvature_trace']>0))
                assert all(np.isfinite(t[k]).all() for k in t.files)
                assert np.all(t['loss_after']<=t['loss_before']+1e-6+1e-5*t['loss_before'])
                record['endpoints']+=len(t['updated']); record['updated']+=int(t['updated'].sum())
                record['nonzero_delta']+=int((t['delta_norm']>0).sum())
                record['loss_decreased']+=int((t['loss_after']<t['loss_before']).sum())
                record['loss_increased']+=int((t['loss_after']>t['loss_before']).sum())
                for key in ('delta_norm','residual_norm'): record['max_'+key]=max(record['max_'+key],float(t[key].max()))
    for a,record in counts.items():
        for k,v in record.items(): assert done['metrics'][a][k]==v
    torch.set_num_threads(2); torch.manual_seed(131); base,_=make_parent(); parent=models_for_calibration(base)['mapped_fixed']
    models={a:AnchoredAdapter(copy.deepcopy(parent.base),rule=a).cuda().eval() for a in ('fixed','normalized')}
    layers=torch.load(root/'layers.pt',weights_only=True,map_location='cpu')
    for name,m in models.items(): m.layer.load_state_dict(layers[name])
    snapshots={k:state_cpu(m) for k,m in models.items()}; mapping=READ(REFERENCE/'input_map.json')
    selected=list(dict.fromkeys([p['shot_ids'][0],p['shot_ids'][len(rows)//2],p['shot_ids'][-1]]+
        [r['shot'] for a in ('fixed_on','normalized_on') for direction in ('good','bad') for r in changes[a][direction][:1]]))
    def deny(*args,**kwargs): raise AssertionError('replay cannot access outcomes')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in selected:
            s=read_signal(sid);curves,traces=predict(models['fixed'],models['normalized'],transform({k:s[k] for k in ('values','available')},mapping))
            with np.load(root/'shots'/f'{sid}.npz') as z:
                for a in ARMS: np.testing.assert_array_equal(curves[a],z[a])
            for name,t in traces.items():
                with np.load(root/'shots'/f'{sid}_{name}_inner.npz') as z:
                    for k,v in t.items(): np.testing.assert_array_equal(v,z[k])
    for k,m in models.items(): assert_unchanged(m,snapshots[k])
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    verification=dict(status='VERIFIED_ANCHORED_WINDOW_EXPERIMENT',completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)),independent_recounts=checks,replayed_shots=selected,counts=counts,
        changes=changes,inner=inner,median_per_shot_reverse_probability_difference=float(np.median(reverse_differences)),
        positive_intra_gain={a:done['metrics'][a]['J']>done['metrics']['parent']['J'] for a in ('fixed_on','normalized_on')},
        fpr_cap_met={a:done['metrics'][a]['fpr']<=.2 for a in ('fixed_on','normalized_on')},independent_confirmation=False)
    _durable_json(root/'verification.json',verification)
    print({k:v for k,v in verification.items() if k!='changes'},flush=True)
    print({a:{k:len(v) for k,v in c.items()} for a,c in changes.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
