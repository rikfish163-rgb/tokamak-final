"""Recount every shot; verify preparation geometry and sample CUDA trajectories."""
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
from ..simple_online import _durable_json
from ..simple_train import sha
from ..task_geometry_20261023.run import READ,load_model
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.audit import counts
from .learning import ARMS,history_basis,predict
from .run import NAMES


def summary(values):
    a=np.asarray(values,dtype=float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('finite nonempty values required')
    return dict(min=float(a.min()),median=float(np.median(a)),mean=float(a.mean()),max=float(a.max()))


def audit(root):
    p,done,full=READ(root/'protocol.json'),READ(root/'completed.json'),READ(root/'per_shot.json')
    assert done['status']=='COMPLETED' and p['contract']==load_test_contract('EAST')
    for f,digest in done['artifact_hashes'].items(): assert sha(root/f)==digest,f
    for f,digest in p['code_sha256'].items(): assert sha(io23.ROOT/f)==digest,f
    assert sha(Path(p['source_checkpoint']))==p['source_sha256']
    assert sha(Path(p['parent']))==p['parent_sha256'] and sha(Path(p['plan_path']))==p['plan_sha256']
    ref=Path(p['reference'])
    for f,digest in p['reference_hashes'].items(): assert sha(ref/f)==digest,f
    ids=p['selected_ids'];order=p['contract']['shot_ids']
    assert len(ids)==20 and not set(ids)&set(order)
    access=READ(root/'pre_eval_label_access.json')
    assert len(access)==20 and [r['shot'] for r in access]==ids
    assert all(r['device']=='EAST' for r in access)
    seals=READ(root/'prediction_receipts.json');assert [r['shot'] for r in seals]==order
    calibration=READ(root/'calibration.json');assert [r['shot'] for r in calibration]==ids
    assert sum(r['is_disrupt'] for r in calibration)==4
    geom=torch.load(root/'geometry.pt',map_location='cpu',weights_only=True)
    q=geom['bases']['history'].numpy().astype('float64');rows=geom['rows'].numpy().astype('float64')
    np.testing.assert_allclose(q@q.T,np.eye(len(q)),atol=2e-6,rtol=2e-6)
    # Independent least-squares check: every historical derivative lies in Q.
    row_residual=rows-(rows@q.T)@q
    assert np.linalg.norm(row_residual)/np.linalg.norm(rows)<2e-6
    assert geom['anchors']['x'].shape==(20,128,23)
    recomputed={a:[] for a in NAMES};stats={a:dict(steps=0,nonzero=0,loss_decreased=0,loss_increased=0,
        step_lengths=[],requested=[],parent_distances=[],tangent_step=[],tangent_actual=[],retained=[]) for a in ARMS}
    length_differences=[];preupdate_fp=[]
    for i,seal in enumerate(seals):
        sid=seal['shot'];path=root/'shots'/f'{sid}.npz';up=root/'shots'/f'{sid}_updates.json'
        assert sha(path)==seal['prediction_sha256'] and sha(up)==seal['updates_sha256']
        old=ref/'shots'/f'{sid}.npz';assert sha(old)==seal['reference_sha256']
        disruptive,down=io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as z,np.load(old,allow_pickle=False) as original:
            assert set(z.files)=={'times',*NAMES}
            np.testing.assert_array_equal(z['times'],original['times'])
            for a,b in (('static','conditional_moments_static'),('forecast','conditional_moments_intra')):
                np.testing.assert_array_equal(z[a],original[b])
            for a in ARMS: np.testing.assert_array_equal(z[a][:256],z['static'][:256])
            if not disruptive and (z['static'][:256]>=p['threshold']).any():preupdate_fp.append(sid)
            for a in NAMES:
                r=recount(sid,z['times'],z[a],disruptive,down,p['threshold'])
                assert r==full[a]['predictions'][i],(sid,a)
                recomputed[a].append(r)
            updates=READ(up);assert set(updates)==set(ARMS)
            for a in ARMS:
                assert [r['first_affected_prediction'] for r in updates[a]]==list(range(256,len(z['times']),128))
                for r,h in zip(updates[a],updates['history']):
                    assert r['last_observation']+1==r['first_affected_prediction']
                    assert all(e+25<=r['last_observation'] for e in r['forecast_origins'])
                    assert r['requested_step_norm']==h['requested_step_norm'] and r['requested_step_norm']<=.00100001
                    difference=abs(r['parameter_delta_norm']-h['parameter_delta_norm']);length_differences.append(difference)
                    assert difference<2e-7
                    if a=='history':
                        assert r['historical_tangent_step_norm']<1e-8
                    s=stats[a];s['steps']+=r['gradient_steps'];s['nonzero']+=r['parameter_delta_norm']>0
                    if r['gradient_steps']:
                        s['loss_decreased']+=r['loss_after']<r['loss_before'];s['loss_increased']+=r['loss_after']>r['loss_before']
                        s['step_lengths'].append(r['parameter_delta_norm']);s['requested'].append(r['requested_step_norm'])
                        s['parent_distances'].append(r['parent_distance']);s['tangent_step'].append(r['historical_tangent_step_norm'])
                        s['tangent_actual'].append(r['historical_tangent_actual_norm'])
                        s['retained'].append(r['projected_gradient_norm']/r['raw_gradient_norm'])
    for a,rs in recomputed.items():
        c=counts(rs);m=done['metrics'][a]
        for k in ('tp','fp','tpr','fpr','J','n_disrupt','n_safe'):assert c[k]==m[k],(a,k)
        leads=[r['lead_ms'] for r in rs if r['timely']]
        assert float(np.median(leads))==m['median_lead_ms']
        assert sum(l>3000 for l in leads)==m['very_early_over3000']
    for a in ARMS:
        m,b=done['metrics'][a],done['metrics']['static']
        assert done['passes_descriptive_screen'][a]==bool(m['fpr']<=.2 and m['tp']>=b['tp'] and m['J']>b['J'])
    flips={a:paired_flips(recomputed['static'],recomputed[a]) for a in NAMES[1:]}
    for s in stats.values():
        for key in ('step_lengths','requested','parent_distances','tangent_step','tangent_actual','retained'):
            s[key+'_summary']=summary(s.pop(key))
    drift=READ(root/'historical_drift.json');assert set(drift)==set(map(str,order))
    drift_stats={}
    for a in ARMS:
        values=np.asarray([drift[str(s)][a] for s in order]);assert values.shape==(360,20)
        drift_stats[a]=dict(abs_logit_change=summary(np.abs(values).reshape(-1)),
            signed_disruptive=summary(values[:,[r['is_disrupt'] for r in calibration]].reshape(-1)),
            signed_safe=summary(values[:,[not r['is_disrupt'] for r in calibration]].reshape(-1)))
    torch.set_num_threads(2);torch.manual_seed(131);model,_=load_model();original=state_cpu(model)
    mapping=READ(ref/'input_map.json');read_outcome=io23.read_outcome
    def historical_only(device,sid):
        assert device=='EAST' and sid in ids
        return read_outcome(device,sid)
    with patch.object(io23,'read_outcome',side_effect=historical_only):
        records=[]
        for sid in ids:
            s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
            s.update(transform({k:s[k] for k in ('values','available')},mapping));records.append(s)
        bases,anchors,recreated_rows,receipt=history_basis(model,records)
        assert receipt==calibration
        torch.testing.assert_close(recreated_rows.cpu(),geom['rows'],rtol=0,atol=0)
        for k,v in bases.items():torch.testing.assert_close(v.cpu(),geom['bases'][k],rtol=0,atol=0)
        for k,v in anchors.items():torch.testing.assert_close(v.cpu(),geom['anchors'][k],rtol=0,atol=0)
    selected={order[0],order[len(order)//2],order[-1]}
    for a in ARMS:
        for shots in flips[a].values():
            if shots:selected.add(shots[0])
    replay=[]
    with patch.object(io23,'read_outcome',side_effect=AssertionError('no outcomes during replay')):
        for sid in sorted(selected):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=transform({k:s[k] for k in ('values','available')},mapping)
            curves,updates,replay_drift=predict(model,signal,bases,anchors=anchors)
            with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z:
                for a in ARMS:np.testing.assert_array_equal(curves[a],z[a])
            assert updates==READ(root/'shots'/f'{sid}_updates.json') and replay_drift==drift[str(sid)]
            replay.append(dict(shot=sid,curves=len(ARMS),exact=True));print(dict(replayed=sid),flush=True)
    assert_unchanged(model,original)
    report=dict(status='VERIFIED',completed_sha256=sha(root/'completed.json'),audit_sha256=sha(Path(__file__)),
        independently_recounted_shots=len(order)*len(NAMES),geometry_recomputed_exact=True,
        sampled_cuda_replays=replay,full_independent_reinference=False,parent_unchanged=True,
        update_summary=stats,paired_flips=flips,historical_drift=drift_stats,
        max_actual_step_length_difference=max(length_differences),preupdate_false_alarms=preupdate_fp,
        descriptive_wilson95={a:counts(r) for a,r in recomputed.items()},
        caveat='development-exposed single parent; historical geometry is not a physical invariant or independent confirmation')
    _durable_json(root/'verification.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);audit(p.parse_args().root)
