"""Independent shot recount, signed constraint and real-history loss checks."""
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
from ..entropy_stream_20261023.audit import recount
from ..fixed_test_protocol import load_test_contract
from ..history_tangent_20261023.audit import summary
from ..simple_online import _durable_json
from ..simple_train import sha
from ..task_geometry_20261023.run import READ,load_model
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.audit import counts
from .learning import ARMS,predict
from .run import NAMES,load_geometry


def binary_loss(logits,labels):
    logits=np.asarray(logits,dtype=float);labels=np.asarray(labels,dtype=float)
    return np.logaddexp(0.,logits)-labels*logits


def audit(root):
    p,done,full=READ(root/'protocol.json'),READ(root/'completed.json'),READ(root/'per_shot.json')
    assert done['status']=='COMPLETED' and p['contract']==load_test_contract('EAST')
    for f,digest in done['artifact_hashes'].items():assert sha(root/f)==digest,f
    for f,digest in p['code_sha256'].items():assert sha(io23.ROOT/f)==digest,f
    ref=Path(p['reference'])
    for f,digest in p['reference_hashes'].items():assert sha(ref/f)==digest,f
    assert sha(Path(p['source_checkpoint']))==p['source_sha256'] and sha(Path(p['parent']))==p['parent_sha256']
    assert sha(Path(p['plan_path']))==p['plan_sha256']
    assert READ(root/'pre_eval_hdf5_label_access.json')==[]
    order=p['contract']['shot_ids'];ids=p['selected_ids'];assert len(ids)==20 and not set(ids)&set(order)
    cal=READ(ref/'calibration.json');assert [r['shot'] for r in cal]==ids
    labels=np.array([r['is_disrupt'] for r in cal],dtype=float)
    np.testing.assert_array_equal(labels,p['labels']);np.testing.assert_array_equal(np.roll(labels,1),p['wrong_labels'])
    assert labels.sum()==4
    geometry=torch.load(ref/'geometry.pt',map_location='cpu',weights_only=True)
    rows=geometry['rows'].double().numpy();unit=rows/np.linalg.norm(rows,axis=1,keepdims=True)
    constraints=torch.load(root/'constraints.pt',map_location='cpu',weights_only=True)
    for name,lab in (('signed',labels),('wrong_labels',np.roll(labels,1))):
        np.testing.assert_allclose(constraints[name].numpy(),unit*(1-2*lab)[:,None],atol=2e-14,rtol=2e-14)
    origin=geometry['anchors']['logits'].numpy().astype(float);initial_bce=binary_loss(origin,labels)
    seals=READ(root/'prediction_receipts.json');assert [r['shot'] for r in seals]==order
    drift=READ(root/'historical_drift.json');assert set(drift)==set(map(str,order))
    refseals={r['shot']:r for r in READ(ref/'prediction_receipts.json')}
    recomputed={a:[] for a in NAMES};length_difference=[]
    stats={a:dict(steps=0,nonzero=0,forecast_decreased=0,history_any_worse=0,
        history_mean_worse=0,history_all_nonworse=0,history_anchor_worse=0,
        retained=[],step_length=[],parent_distance=[],history_bce_delta=[],final_history_bce_delta=[],
        linear_margin_min=[],cast_margin_min=[],complementarity=[]) for a in ARMS}
    for index,seal in enumerate(seals):
        sid=seal['shot'];path=root/'shots'/f'{sid}.npz';up=root/'shots'/f'{sid}_updates.json'
        assert sha(path)==seal['prediction_sha256'] and sha(up)==seal['updates_sha256']
        old=ref/'shots'/f'{sid}.npz';assert sha(old)==seal['reference_sha256']==refseals[sid]['prediction_sha256']
        disruptive,down=io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as z,np.load(old,allow_pickle=False) as previous:
            np.testing.assert_array_equal(z['times'],previous['times']);assert set(z.files)=={'times',*NAMES}
            for a,b in p['old_curve_mapping'].items():np.testing.assert_array_equal(z[a],previous[b])
            for a in ARMS:np.testing.assert_array_equal(z[a][:256],z['static'][:256])
            for a in NAMES:
                r=recount(sid,z['times'],z[a],disruptive,down,p['threshold'])
                assert r==full[a]['predictions'][index],(sid,a)
                recomputed[a].append(r)
            updates=READ(up);assert set(updates)==set(ARMS)
            for a in ARMS:
                assert [r['first_affected_prediction'] for r in updates[a]]==list(range(256,len(z['times']),128))
                total=np.zeros(20);last_after=initial_bce.copy();s=stats[a]
                for r,refstep in zip(updates[a],updates['signed']):
                    assert r['last_observation']+1==r['first_affected_prediction']
                    assert max(r['forecast_origins'])+25<=r['last_observation']
                    assert r['requested_step_norm']==refstep['requested_step_norm'] and r['requested_step_norm']<=.00100001
                    diff=abs(r['parameter_delta_norm']-refstep['parameter_delta_norm']);length_difference.append(diff);assert diff<2e-7
                    s['steps']+=r['gradient_steps'];s['nonzero']+=r['parameter_delta_norm']>0
                    if not r['gradient_steps']:continue
                    s['forecast_decreased']+=r['loss_after']<r['loss_before']
                    change=np.asarray(r['history_bce_change']);before=np.asarray(r['history_bce_before'])
                    assert change.shape==before.shape==(20,) and np.isfinite(change).all() and np.isfinite(before).all()
                    np.testing.assert_allclose(before,last_after,atol=3e-6,rtol=1e-6)
                    last_after=before+change;total+=change
                    s['history_any_worse']+=bool((change>1e-6).any())
                    s['history_mean_worse']+=float(change.mean())>1e-6
                    s['history_all_nonworse']+=bool((change<=1e-6).all())
                    s['history_anchor_worse']+=int((change>1e-6).sum())
                    s['history_bce_delta'].extend(change.tolist())
                    s['retained'].append(r['projected_gradient_norm']/r['raw_gradient_norm'])
                    s['step_length'].append(r['parameter_delta_norm']);s['parent_distance'].append(r['parent_distance'])
                    s['linear_margin_min'].append(r['true_linear_margin_min'])
                    if a!='norm_only':
                        scale=max(1.,r['raw_norm'])
                        assert r['min_margin_after']>=-2e-6*scale
                        assert r['kkt_complementarity']<=1e-8*scale*scale
                        s['cast_margin_min'].append(r['min_margin_after']);s['complementarity'].append(r['kkt_complementarity'])
                final_change=binary_loss(origin+np.asarray(drift[str(sid)][a]),labels)-initial_bce
                np.testing.assert_allclose(total,final_change,rtol=3e-5,atol=2e-5)
                s['final_history_bce_delta'].extend(final_change.tolist())
    for a,rs in recomputed.items():
        c=counts(rs);m=done['metrics'][a]
        for k in ('tp','fp','tpr','fpr','J','n_disrupt','n_safe'):assert c[k]==m[k],(a,k)
        leads=[r['lead_ms'] for r in rs if r['timely']]
        assert float(np.median(leads))==m['median_lead_ms'] and sum(x>3000 for x in leads)==m['very_early_over3000']
    for a in ARMS:
        m,b=done['metrics'][a],done['metrics']['static']
        assert done['passes_descriptive_screen'][a]==bool(m['fpr']<=.2 and m['tp']>=b['tp'] and m['J']>b['J'])
    flips={a:paired_flips(recomputed['static'],recomputed[a]) for a in NAMES[1:]}
    for s in stats.values():
        changes=np.asarray(s['history_bce_delta']).reshape(-1,20)
        final=np.asarray(s['final_history_bce_delta']).reshape(len(order),20)
        s['per_history_anchor']=[dict(shot=ids[j],is_disrupt=bool(labels[j]),
            update_worsened=int((changes[:,j]>1e-6).sum()),update_delta=summary(changes[:,j]),
            final_worsened=int((final[:,j]>1e-6).sum()),final_delta=summary(final[:,j])) for j in range(20)]
        s['class_mean_update_delta']={str(int(label)):summary(changes[:,labels==label].mean(1)) for label in (0.,1.)}
        s['class_mean_final_delta']={str(int(label)):summary(final[:,labels==label].mean(1)) for label in (0.,1.)}
        for key in ('retained','step_length','parent_distance','history_bce_delta','final_history_bce_delta',
                    'linear_margin_min','cast_margin_min','complementarity'):
            values=s.pop(key);s[key+'_summary']=summary(values) if values else None
    selected={order[0],order[len(order)//2],order[-1]}
    for a in ARMS:
        for shots in flips[a].values():
            if shots:selected.add(shots[0])
    torch.set_num_threads(2);model,_=load_model();original=state_cpu(model);cones,anchors,_=load_geometry()
    for a,c in cones.items():torch.testing.assert_close(c.h.cpu(),constraints[a],rtol=0,atol=0)
    mapping=READ(Path(READ(ref/'protocol.json')['reference'])/'input_map.json');replay=[]
    with patch.object(io23,'read_outcome',side_effect=AssertionError('no outcomes during replay')):
        for sid in sorted(selected):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=transform({k:s[k] for k in ('values','available')},mapping)
            curves,updates,replayed_drift=predict(model,signal,cones,anchors)
            with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z:
                for a in ARMS:np.testing.assert_array_equal(curves[a],z[a])
            assert updates==READ(root/'shots'/f'{sid}_updates.json') and replayed_drift==drift[str(sid)]
            replay.append(dict(shot=sid,curves=len(ARMS),exact=True));print(dict(replayed=sid),flush=True)
    assert_unchanged(model,original)
    report=dict(status='VERIFIED',completed_sha256=sha(root/'completed.json'),audit_sha256=sha(Path(__file__)),
        independently_recounted_shots=len(order)*len(NAMES),history_constraints_rebuilt=True,
        sampled_cuda_replays=replay,full_independent_reinference=False,parent_unchanged=True,
        update_summary=stats,paired_flips=flips,max_step_length_difference=max(length_difference),
        history_BCE_telescope_verified=True,descriptive_wilson95={a:counts(r) for a,r in recomputed.items()},
        caveat='single development-exposed parent; frozen tangent inequalities are not nonlinear safety guarantees')
    _durable_json(root/'verification.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);audit(p.parse_args().root)
