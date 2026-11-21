"""Independent shot recount and finite-history acceptance audit."""
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
from ..label_cone_20261023.audit import binary_loss
from ..simple_online import _durable_json
from ..simple_train import sha
from ..task_geometry_20261023.run import READ,load_model
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.audit import counts
from .learning import ARMS,predict,HISTORY_TOL,FRACTIONS
from .run import NAMES


def check_guard(r):
    trials=r['trials'];fraction=r['fraction'];delta=np.asarray(r['history_bce_change'])
    absolute=np.asarray(r['history_change_from_parent'])
    assert 1<=len(trials)<=len(FRACTIONS)
    assert [t['fraction'] for t in trials]==list(FRACTIONS[:len(trials)])
    for i,t in enumerate(trials):
        accepted=t['max_step_bce_increase']<=HISTORY_TOL and t['max_parent_bce_increase']<=HISTORY_TOL
        assert t['accepted']==accepted
        if i<len(trials)-1:assert not accepted
    assert delta.max()<=HISTORY_TOL and absolute.max()<=HISTORY_TOL
    assert r['accepted_step_norm']==r['requested_step_norm']*fraction
    if fraction:
        assert fraction==trials[-1]['fraction'] and trials[-1]['accepted']
        assert r['reason']=='history_guard_accepted'
        assert abs(float(delta.max())-trials[-1]['max_step_bce_increase'])<1e-12
        assert abs(float(absolute.max())-trials[-1]['max_parent_bce_increase'])<1e-12
    else:
        assert len(trials)==len(FRACTIONS) and not trials[-1]['accepted']
        assert r['reason']=='history_guard_rejected' and r['parameter_delta_norm']==0
        assert (delta==0).all()


def audit(root):
    p,done,full=READ(root/'protocol.json'),READ(root/'completed.json'),READ(root/'per_shot.json')
    assert done['status']=='COMPLETED' and p['contract']==load_test_contract('EAST')
    for f,digest in done['artifact_hashes'].items():assert sha(root/f)==digest,f
    for f,digest in p['code_sha256'].items():assert sha(io23.ROOT/f)==digest,f
    ref=Path(p['reference'])
    for f,digest in p['reference_hashes'].items():assert sha(ref/f)==digest,f
    assert sha(Path(p['source_checkpoint']))==p['source_sha256'] and sha(Path(p['parent']))==p['parent_sha256']
    assert sha(Path(p['plan_path']))==p['plan_sha256']
    assert sha(Path(p['prepared_reference'])/'input_map.json')==p['input_map_sha256']
    assert READ(root/'pre_eval_hdf5_label_access.json')==[]
    assert p['history_tolerance']==HISTORY_TOL and p['backtrack_fractions']==list(FRACTIONS)
    order=p['contract']['shot_ids'];ids=p['selected_ids'];assert len(ids)==20 and not set(ids)&set(order)
    anchors=torch.load(root/'anchors.pt',map_location='cpu',weights_only=True)
    old_geometry=torch.load(Path(p['history_reference'])/'geometry.pt',map_location='cpu',weights_only=True)
    cal=READ(Path(p['history_reference'])/'calibration.json');assert [r['shot'] for r in cal]==ids
    for k,v in old_geometry['anchors'].items():torch.testing.assert_close(anchors[k],v,rtol=0,atol=0)
    labels=np.array([r['is_disrupt'] for r in cal],dtype=float)
    np.testing.assert_array_equal(anchors['labels'].numpy(),labels);assert labels.sum()==4
    origin=anchors['logits'].numpy().astype(float);initial=binary_loss(origin,labels)
    seals=READ(root/'prediction_receipts.json');assert [r['shot'] for r in seals]==order
    old_seals={r['shot']:r for r in READ(ref/'prediction_receipts.json')}
    drifts=READ(root/'historical_drift.json');assert set(drifts)==set(map(str,order))
    recomputed={a:[] for a in NAMES};length_differences=[]
    stats={a:dict(proposals=0,updates=0,forecast_decreased=0,history_any_worse=0,
        history_mean_worse=0,history_parent_any_worse=0,step_lengths=[],requested_lengths=[],
        distances=[],history_changes=[],final_history_changes=[],conditions=[],linear_margins=[],
        fractions={},trial_counts={}) for a in ARMS}
    for index,seal in enumerate(seals):
        sid=seal['shot'];path=root/'shots'/f'{sid}.npz';up=root/'shots'/f'{sid}_updates.json'
        assert sha(path)==seal['prediction_sha256'] and sha(up)==seal['updates_sha256']
        old=ref/'shots'/f'{sid}.npz';assert sha(old)==seal['reference_sha256']==old_seals[sid]['prediction_sha256']
        disruptive,down=io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as z,np.load(old,allow_pickle=False) as prev:
            assert set(z.files)=={'times',*NAMES};np.testing.assert_array_equal(z['times'],prev['times'])
            for a,b in p['old_curve_mapping'].items():np.testing.assert_array_equal(z[a],prev[b])
            for a in ARMS:np.testing.assert_array_equal(z[a][:256],z['static'][:256])
            for a in NAMES:
                r=recount(sid,z['times'],z[a],disruptive,down,p['threshold'])
                assert r==full[a]['predictions'][index],(sid,a);recomputed[a].append(r)
            updates=READ(up);assert set(updates)==set(ARMS)
            for a in ARMS:
                assert [r['first_affected_prediction'] for r in updates[a]]==list(range(256,len(z['times']),128))
                total=np.zeros(20);last=initial.copy();s=stats[a]
                for r,guarded in zip(updates[a],updates['guarded']):
                    count=r['first_affected_prediction'];assert r['last_observation']==count-1
                    assert all(count-128<=end and end+25<=count-1 for end in r['forecast_origins'])
                    assert r['parameter_delta_norm']<=.00101
                    if not r['observed_targets']:
                        assert not r['gradient_steps'] and r['parameter_delta_norm']==0;continue
                    before=np.asarray(r['history_bce_before']);change=np.asarray(r['history_bce_change'])
                    from_parent=np.asarray(r['history_change_from_parent'])
                    assert before.shape==change.shape==from_parent.shape==(20,)
                    assert np.isfinite(before).all() and np.isfinite(change).all() and np.isfinite(from_parent).all()
                    np.testing.assert_allclose(before,last,rtol=1e-11,atol=1e-11)
                    last=before+change;total+=change
                    np.testing.assert_allclose(from_parent,last-initial,rtol=1e-10,atol=1e-11)
                    if a=='guarded':check_guard(r)
                    elif a=='norm_only':
                        assert r['requested_step_norm']==guarded['accepted_step_norm']
                        diff=abs(r['parameter_delta_norm']-guarded['parameter_delta_norm'])
                        length_differences.append(diff);assert diff<2e-7
                    if a!='norm_only':
                        scale=max(1.,r['raw_norm'])
                        assert r['min_margin_after']>=-2e-6*scale and r['kkt_complementarity']<=1e-8*scale*scale
                        s['conditions'].append(r['gram_condition']);s['linear_margins'].append(r['current_linear_margin_min'])
                    s['proposals']+=1;s['updates']+=r['gradient_steps']
                    assert r['gradient_steps']==int(r['parameter_delta_norm']>0)
                    s['forecast_decreased']+=r['loss_after']<r['loss_before']
                    s['history_any_worse']+=bool((change>HISTORY_TOL).any())
                    s['history_mean_worse']+=float(change.mean())>HISTORY_TOL
                    s['history_parent_any_worse']+=bool((from_parent>HISTORY_TOL).any())
                    s['step_lengths'].append(r['parameter_delta_norm']);s['requested_lengths'].append(r['requested_step_norm'])
                    s['distances'].append(r['parent_distance']);s['history_changes'].append(change)
                    if a=='guarded':
                        key=str(r['fraction']);s['fractions'][key]=s['fractions'].get(key,0)+1
                        key=str(len(r['trials']));s['trial_counts'][key]=s['trial_counts'].get(key,0)+1
                final=binary_loss(origin+np.asarray(drifts[str(sid)][a]),labels)-initial
                # Stored logit differences are float32, while per-step BCE is
                # float64. Only this cross-representation check uses tolerance.
                np.testing.assert_allclose(total,final,rtol=3e-5,atol=2e-5)
                s['final_history_changes'].append(total)
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
        changes=np.asarray(s.pop('history_changes'));final=np.asarray(s.pop('final_history_changes'))
        s['history_delta']=summary(changes.reshape(-1));s['final_history_delta']=summary(final.reshape(-1))
        s['per_class_update']={str(int(y)):summary(changes[:,labels==y].mean(1)) for y in (0.,1.)}
        s['per_class_final']={str(int(y)):summary(final[:,labels==y].mean(1)) for y in (0.,1.)}
        for key in ('step_lengths','requested_lengths','distances','conditions','linear_margins'):
            values=s.pop(key);s[key]=summary(values) if values else None
    selected={order[0],order[len(order)//2],order[-1]}
    for a in ARMS:
        for shots in flips[a].values():
            if shots:selected.add(shots[0])
    torch.set_num_threads(2);model,_=load_model();original=state_cpu(model)
    anchors={k:v.cuda() for k,v in anchors.items()};mapping=READ(Path(p['prepared_reference'])/'input_map.json');replay=[]
    with patch.object(io23,'read_outcome',side_effect=AssertionError('no TEST labels during replay')):
        for sid in sorted(selected):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=transform({k:s[k] for k in ('values','available')},mapping)
            curves,updates,drift=predict(model,signal,anchors)
            with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z:
                for a in ARMS:np.testing.assert_array_equal(curves[a],z[a])
            assert updates==READ(root/'shots'/f'{sid}_updates.json') and drift==drifts[str(sid)]
            replay.append(dict(shot=sid,curves=len(ARMS),exact=True));print(dict(replayed=sid),flush=True)
    assert_unchanged(model,original)
    report=dict(status='VERIFIED',completed_sha256=sha(root/'completed.json'),audit_sha256=sha(Path(__file__)),
        independently_recounted_shots=len(order)*len(NAMES),history_data_rebuilt=True,
        sampled_cuda_replays=replay,full_independent_reinference=False,parent_unchanged=True,
        update_summary=stats,paired_flips=flips,max_step_length_difference=max(length_differences),
        history_BCE_telescope_verified=True,guarded_actual_and_parent_envelope_verified=True,
        descriptive_wilson95={a:counts(r) for a,r in recomputed.items()},
        caveat='development-exposed single parent; fixed20-window protection is not whole-shot or new-shot safety')
    _durable_json(root/'verification.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);audit(p.parse_args().root)
