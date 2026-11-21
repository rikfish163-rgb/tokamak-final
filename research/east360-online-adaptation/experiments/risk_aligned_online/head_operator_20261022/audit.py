"""Rescore immutable head/operator traces and replay selected full shots.

Operating curves use all revealed outcomes after completion and are explicitly
posthoc. They never replace the fixed deployment threshold.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..fixed_test_protocol import load_test_contract,score_fixed_test
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..operating_curve_audit import operating_curve,operational_maxima
from ..warmstart_alignment_20261022.alignment import transform
from ..deepcoral_prefix_20261022.audit import paired_flips
from ..deepcoral_prefix_20261022.run import predict_pairs
from .run import ARMS,REFERENCE,load_calibrated,token_predictions


def validate_update(arm,u,n):
    count=u['first_affected_prediction']
    assert u['last_observation']==count-1
    assert count%128==0 and 256<=count<n
    assert u['forecast_origins']==[count-1-25*(j+1) for j in range(4)]
    assert all(count-128<=e and e+25<count for e in u['forecast_origins'])
    assert u['gradient_steps'] in (0,1)
    if u['gradient_steps']:
        assert u['observed_targets']>0
        assert all(np.isfinite(u[k]) for k in ('gradient_norm','loss_before','loss_after'))
        if arm.startswith('token_'):
            assert 0<=u['distance_from_origin']<=1.00001
            assert 0<=u['step_norm']<=.05001
        else:
            assert 0<=u['parameter_delta_norm']<=.00101
    else:
        assert u['observed_targets']==0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    root=p.parse_args().directory
    completed=json.loads((root/'completed.json').read_text());assert completed['status']=='COMPLETED'
    protocol=json.loads((root/'protocol.json').read_text())
    assert protocol['arms']==list(ARMS)
    for path,digest in protocol['code_sha256'].items(): assert sha(io23.ROOT/path)==digest
    assert sha(protocol['plan_path'])==protocol['plan_sha256']
    assert sha(REFERENCE/'protocol.json')==protocol['reference_protocol_sha256']
    assert sha(REFERENCE/'density_map.json')==protocol['density_mapping_sha256']
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    order=load_test_contract('EAST')['shot_ids']
    seals=json.loads((root/'prediction_receipts.json').read_text())
    assert [x['shot'] for x in seals]==order
    old_seals={x['shot']:x for x in json.loads((REFERENCE/'prediction_receipts.json').read_text())}
    activation=json.loads((root/'activation_receipt.json').read_text())
    assert activation['after_sealed_ids']==order[:20] and activation['preceding_seals']==seals[:20]
    assert sha(REFERENCE/'density.pt')==activation['calibrated_checkpoint_sha256']
    mapping=json.loads((REFERENCE/'density_map.json').read_text())
    assert mapping['calibration_ids']==order[:20]
    full=json.loads((root/'per_shot.json').read_text())
    rows={k:[] for k in ARMS};stats={k:dict(updates=0,loss_decreased=0,
        step_norm_max=0.,radius_max=0.,gradient_norm_max=0.) for k in ARMS[1:]}
    # Update receipts were not sealed by this run; record their audit-time
    # hashes explicitly instead of falsely describing them as prediction seals.
    audited_update_files=[]
    for index,(sid,seal) in enumerate(zip(order,seals)):
        path=root/'shots'/f'{sid}.npz';upath=root/'shots'/f'{sid}_updates.json'
        refpath=REFERENCE/'shots'/f'{sid}.npz';refupath=REFERENCE/'shots'/f'{sid}_updates.json'
        assert sha(path)==seal['prediction_sha256'] and sha(refpath)==old_seals[sid]['prediction_sha256']
        assert sha(refupath)==old_seals[sid]['updates_sha256']
        updates=json.loads(upath.read_text())
        audited_update_files.append(dict(shot=sid,sha256=sha(upath)))
        shot=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(path,allow_pickle=False) as trace,np.load(refpath,allow_pickle=False) as ref:
            np.testing.assert_array_equal(trace['times'],shot['times'])
            np.testing.assert_array_equal(trace['static'],ref['density_static'])
            np.testing.assert_array_equal(trace['encoder_calibrated'],ref['density_intra'])
            for arm in ARMS:
                if index<20: np.testing.assert_array_equal(trace[arm],trace['static'])
                else: np.testing.assert_allclose(trace[arm][:256],trace['static'][:256],rtol=0,atol=2e-6)
                rows[arm].append(dict(shot=sid,times=shot['times'],scores=trace[arm].copy(),
                    valid=shot['valid'],is_disrupt=shot['is_disrupt'],down=shot['down']))
        if index<20:
            assert not updates
            continue
        assert set(updates)=={'encoder_original','token_original','token_calibrated'}
        updates['encoder_calibrated']=json.loads(refupath.read_text())['density']
        expected=list(range(256,len(shot['values']),128))
        for arm,records in updates.items():
            assert [r['first_affected_prediction'] for r in records]==expected
            for u in records:
                validate_update(arm,u,len(shot['values']))
                if not u['gradient_steps']: continue
                s=stats[arm];s['updates']+=1;s['loss_decreased']+=u['loss_after']<u['loss_before']
                s['gradient_norm_max']=max(s['gradient_norm_max'],u['gradient_norm'])
                key='step_norm' if arm.startswith('token_') else 'parameter_delta_norm'
                s['step_norm_max']=max(s['step_norm_max'],u[key])
                if arm.startswith('token_'):s['radius_max']=max(s['radius_max'],u['distance_from_origin'])
    operating={}
    for arm,records in rows.items():
        cropped=[dict(r,times=r['times'][r['valid']],scores=r['scores'][r['valid']]) for r in records]
        assert score_fixed_test('EAST',cropped,protocol['threshold'])==full[arm]
        operating[arm]={}
        for period,rs in (('full360',records),('common_suffix340',records[20:])):
            curve=operating_curve(rs,protocol['threshold']);peaks=operational_maxima(rs)
            d,s=peaks['positive'][:,None],peaks['negative'][None,:]
            auc=float(((d>s).astype(float)+.5*(d==s)).mean())
            for key in ('tp','fp','tpr','fpr','J'):
                assert curve['fixed_source_threshold'][key]==completed['metrics'][arm][period][key]
            operating[arm][period]=dict(timely_peak_ranking_auc=auc,**curve)
    torch.set_num_threads(2)
    base,_=load_parent(DEFAULT_PARENT)
    original=PairedFeedback(base,np.ones(23)).cuda();original.set_stage('frozen')
    calibrated=load_calibrated();replays=[]
    for index in (20,179,359):
        sid=order[index];shot=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        signal=transform({k:shot[k] for k in ('values','available')},mapping)
        encoder,er=predict_pairs({'original':original},signal)
        token,tr=token_predictions({'original':original.base,'calibrated':calibrated.base},signal)
        predictions=dict(static=encoder['original_static'],encoder_original=encoder['original_intra'],
                         token_original=token['original'],token_calibrated=token['calibrated'])
        with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
            for arm,value in predictions.items():np.testing.assert_array_equal(value,trace[arm])
        saved=json.loads((root/'shots'/f'{sid}_updates.json').read_text())
        assert er['original']==saved['encoder_original']
        assert tr['original']==saved['token_original'] and tr['calibrated']==saved['token_calibrated']
        replays.append(sid)
    comparisons={f'static_to_{k}':paired_flips(full['static']['predictions'],full[k]['predictions']) for k in ARMS[1:]}
    for op in ('encoder','token'):
        comparisons[f'{op}_head_calibration']=paired_flips(full[f'{op}_original']['predictions'],
                                                         full[f'{op}_calibrated']['predictions'])
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    report=dict(status='VERIFIED',rescored_traces=len(order)*len(ARMS),
        replayed_full_shots=replays,replayed_full_traces=4*len(replays),max_replay_difference=0.,
        calibrated_encoder_arm_verified_by_prior_seals=True,
        all_update_time_boundaries_valid=True,source_archive_unchanged=True,
        update_summary=stats,paired_comparisons=comparisons,
        very_early_over3000ms={k:sum(p['timely'] and p['lead_ms']>3000 for p in r['predictions']) for k,r in full.items()},
        update_receipt_hash_scope='audit-time only; not part of original prediction seals',
        audit_time_update_hashes=audited_update_files)
    _durable_json(root/'verification.json',report)
    _durable_json(root/'operating_diagnostic.json',dict(
        scope='POSTHOC all-period labels used for threshold sweep; not deployed or independent',
        thresholds_written_back=False,results=operating))
    print(json.dumps({k:v for k,v in report.items() if k!='audit_time_update_hashes'}),flush=True)


if __name__=='__main__':main()
