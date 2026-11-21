"""Independent first-crossing recount, update audit, and sampled exact CUDA replay."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.audit import paired_flips
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..fixed_test_protocol import load_test_contract
from ..simple_online import _durable_json
from ..simple_train import sha
from ..task_geometry_20261023.run import READ,load_model
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.audit import counts
from .learning import ARMS,predict
from .run import NAMES


def recount(sid,times,score,disrupt,down,threshold):
    assert score.shape==times.shape and np.isfinite(score).all() and ((score>=0)&(score<=1)).all()
    valid=np.ones(len(times),bool) if not disrupt else (down-times)*1000>=-1e-6
    crossings=np.flatnonzero(valid&(score>=threshold))
    lead=round(float((down-times[crossings[0]])*1000),9) if disrupt and len(crossings) else None
    return dict(shot=sid,is_disrupt=bool(disrupt),alarm=bool(len(crossings)),lead_ms=lead,
                timely=bool(lead is not None and lead>30))


def stable_entropy(z):
    a=abs(float(z));e=np.exp(-a)
    return float(np.log1p(e)+a*e/(1+e))


def audit(root):
    p,done,full=READ(root/'protocol.json'),READ(root/'completed.json'),READ(root/'per_shot.json')
    assert done['status']=='COMPLETED' and p['contract']==load_test_contract('EAST')
    for artifact,key in (('protocol.json','protocol_sha256'),('per_shot.json','per_shot_sha256'),('prediction_receipts.json','seals_sha256')):
        assert sha(root/artifact)==done[key]
    assert sha(Path(p['source_checkpoint']))==p['source_sha256'] and sha(Path(p['parent']))==p['parent_sha256']
    assert sha(Path(p['plan_path']))==p['plan_sha256']
    for name,digest in p['code_sha256'].items():assert sha(io23.ROOT/name)==digest,name
    ref=Path(p['reference'])
    for name,digest in p['reference_hashes'].items():assert sha(ref/name)==digest
    assert sha(Path(p['selection_screen'])/'completed.json')==p['selection_screen_sha256']
    assert READ(root/'pre_eval_label_access.json')==[]
    order=p['contract']['shot_ids'];seals=READ(root/'prediction_receipts.json')
    assert [r['shot'] for r in seals]==order
    assert len(p['selected_ids'])==20 and not set(order)&set(p['selected_ids'])
    reference_seals={r['shot']:r for r in READ(ref/'prediction_receipts.json')}
    recomputed={a:[] for a in NAMES};early_fp=[]
    stats={a:dict(steps=0,nonzero=0,loss_decreased=0,absolute_logit_increased=0,negative_logit_steps=0,
                  max_step=0.,parent_distances=[],update_interval_ms=[]) for a in ARMS}
    for index,seal in enumerate(seals):
        sid=seal['shot'];path=root/'shots'/f'{sid}.npz';up=root/'shots'/f'{sid}_updates.json'
        assert sha(path)==seal['prediction_sha256'] and sha(up)==seal['updates_sha256']
        refpath=ref/'shots'/f'{sid}.npz'
        assert sha(refpath)==seal['reference_sha256']==reference_seals[sid]['prediction_sha256']
        disrupt,down=io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as z,np.load(refpath,allow_pickle=False) as old:
            assert set(z.files)=={'times',*NAMES}
            np.testing.assert_array_equal(z['times'],old['times'])
            for a,b in (('static','conditional_moments_static'),('forecast','conditional_moments_intra')):
                np.testing.assert_array_equal(z[a],old[b])
            n=len(z['times'])
            for a in ARMS:np.testing.assert_array_equal(z[a][:256],z['static'][:256])
            np.testing.assert_array_equal(z['local_reset'][:384],z['cumulative'][:384])
            if not disrupt and (z['static'][:256]>=p['threshold']).any():early_fp.append(sid)
            for a in NAMES:
                r=recount(sid,z['times'],z[a],disrupt,down,p['threshold'])
                assert r==full[a]['predictions'][index],(sid,a)
                recomputed[a].append(r)
            updates=READ(up);assert set(updates)==set(ARMS)
            for a in ARMS:
                assert [u['first_affected_prediction'] for u in updates[a]]==list(range(256,n,128))
                previous=None
                for u in updates[a]:
                    count=u['first_affected_prediction'];assert u['last_observation']==count-1
                    s=stats[a];s['steps']+=u['gradient_steps'];s['nonzero']+=u['parameter_delta_norm']>0
                    assert u['parameter_delta_norm']<=.00101
                    s['max_step']=max(s['max_step'],u['parameter_delta_norm'])
                    if previous is not None:s['update_interval_ms'].append(float((z['times'][count-1]-z['times'][previous-1])*1000))
                    previous=count
                    if u['gradient_steps']:
                        s['parent_distances'].append(u['parent_distance'])
                        if a=='local_reset':assert u['parent_distance']==u['parameter_delta_norm']
                        for when in ('before','after'):
                            np.testing.assert_allclose(u['loss_'+when],stable_entropy(u['observed_logit_'+when]),rtol=1e-4,atol=8e-7)
                        s['loss_decreased']+=u['loss_after']<u['loss_before']
                        s['absolute_logit_increased']+=abs(u['observed_logit_after'])>abs(u['observed_logit_before'])
                        s['negative_logit_steps']+=u['observed_logit_before']<0
    for a,rows in recomputed.items():
        c=counts(rows);m=done['metrics'][a]
        for key in ('tp','fp','n_disrupt','n_safe','tpr','fpr','J'):assert c[key]==m[key],(a,key)
        leads=[r['lead_ms'] for r in rows if r['timely']]
        assert float(np.median(leads))==m['median_lead_ms']
        assert sum(l>3000 for l in leads)==m['very_early_over3000']
    flips={a:paired_flips(recomputed['static'],recomputed[a]) for a in NAMES[1:]}
    for a in ARMS:
        m,b=done['metrics'][a],done['metrics']['static']
        assert done['passes_descriptive_screen'][a]==bool(m['fpr']<=.2 and m['tp']>=b['tp'] and m['J']>b['J'])
    for s in stats.values():
        for key in ('parent_distances','update_interval_ms'):
            values=s.pop(key)
            s[key+'_summary']=dict(min=float(np.min(values)),median=float(np.median(values)),max=float(np.max(values)))
    # Fixed positional sample plus first member of every paired-flip category.
    # Selection is for reproduction, never for updating a deployed policy.
    selected={order[0],order[len(order)//2],order[-1]}
    for a in ARMS:
        for ids in flips[a].values():
            if ids:selected.add(ids[0])
    torch.set_num_threads(2);model,_=load_model();original=state_cpu(model);mapping=READ(ref/'input_map.json');replay=[]
    with patch.object(io23,'read_outcome',side_effect=AssertionError('replay must not inspect outcomes')):
        for sid in sorted(selected):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=transform({k:s[k] for k in ('values','available')},mapping)
            curves,updates=predict(model,signal)
            with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z:
                for a in ARMS:np.testing.assert_array_equal(curves[a],z[a])
            assert updates==READ(root/'shots'/f'{sid}_updates.json')
            replay.append(dict(shot=sid,curves=len(ARMS),exact=True));print(dict(replayed=sid),flush=True)
    assert_unchanged(model,original)
    report=dict(status='VERIFIED',completed_sha256=sha(root/'completed.json'),audit_sha256=sha(Path(__file__)),
        independently_recounted_shots=len(order)*len(NAMES),sampled_cuda_replays=replay,
        full_independent_reinference=False,parent_unchanged=True,update_summary=stats,paired_flips=flips,
        preupdate_unremovable_false_alarms=early_fp,
        descriptive_wilson95={a:counts(r) for a,r in recomputed.items()},
        caveat='development-selected deterministic single parent; intervals descriptive, not independent future confirmation')
    _durable_json(root/'verification.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);audit(p.parse_args().root)
